"""
Experiment 1: a frozen, general-purpose VLM (Qwen2.5-VL-7B-Instruct) controlling the Franka through a small harness,
in the style of RoboHarness / MOKA: the VLM never outputs motor commands. It looks at camera images and
  1. says WHERE the object is (a bounding box in the image)        -> we turn it into a 3D point with depth
  2. judges whether the grasp WORKED ("is the robot holding X?")    -> decides done / retry
and deterministic tools do the motion (IK reach above the point, re-locate from the wrist camera, descend, close, lift).

No robot training, no privileged object state: the tools only ever get the VLM's estimate. (The true object position is
used for logging how wrong the estimate was, never for control.)

    OMNIGIBSON_HEADLESS=1 python my_franka/vlm_harness_eval.py pointing --layouts 4      # perception test
    OMNIGIBSON_HEADLESS=1 python my_franka/vlm_harness_eval.py eval --tasks apple --layouts 8
    OMNIGIBSON_HEADLESS=1 python my_franka/vlm_harness_eval.py eval --layouts 8          # all 12 tasks
    OMNIGIBSON_HEADLESS=1 python my_franka/vlm_harness_eval.py eval --tasks apple --layouts 50 --video-layouts

Output: my_franka/data/vlm/ (results_*.json, pointing overlays, videos).
"""
import argparse
import json
import math
import os
import time

import av
import cv2
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T

import pick_common
from pi0_zero_shot_eval import TASKS, aim_exterior_cam, parking_spot
from pick_common import (
    CLOSE, DESCEND_STEP, LIFT_Z, MAX_STEP, OPEN, TABLE_TOP_Z, build_cfg, check_success, place_layout, sample_layout,
    to_base, to_env_action,
)
from step7_eval import EVAL_SEED

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "vlm")
# 16:9, both multiples of Qwen2.5-VL's 28-px patch grid (no internal resize). Resolution matters a lot: on a synthetic
# "red square" test Qwen's box centre was off by up to 56 px at 448x252 (only 144 visual tokens) but <= 28 px at 896x504.
W, H = 896, 504
VW, VH = W // 2, H // 2  # video frame size per camera
HOVER = 0.12  # m above the estimated object surface for the approach and the wrist re-look
GRASP_BELOW_SURFACE = 0.02  # descend this far below the estimated top surface before closing
MIN_GRASP_Z = 0.015  # never descend below this height above the table (base frame z = table top)
MAX_STEPS = 400  # env steps per episode (30 Hz, 13 s), same budget as the diffusion-policy eval
MAX_ATTEMPTS = 3  # v1 used 2; the 400-step budget is the real limit (one attempt takes ~150 steps)


def object_word(prompt):
    return prompt.replace("pick up the ", "")


def build_harness_cfg():
    cfg = build_cfg()  # absolute-pose IK arm + binary gripper, 30 Hz: the same robot setup as the DP
    robot = cfg["robots"][0]
    robot["obs_modalities"] = ["rgb", "depth_linear"]
    robot["sensor_config"] = {"VisionSensor": {"sensor_kwargs": {"image_height": H, "image_width": W}}}
    cfg["objects"] = [
        {"type": "DatasetObject", "name": f"obj_{name}", "category": cat, "model": model, "position": parking_spot(k),
         "orientation": [0, 0, 0, 1]}
        for k, (name, cat, model, _) in enumerate(TASKS)
    ]
    cfg["env"]["external_sensors"] = [{
        "sensor_type": "VisionSensor", "name": "ext_cam", "modalities": ["rgb", "depth_linear"],
        "sensor_kwargs": {"image_height": H, "image_width": W},
        "position": [0.0, 0.0, 2.0], "orientation": [0, 0, 0, 1], "pose_frame": "world", "include_in_obs": False,
    }]
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# Geometry: pixel + depth -> 3D. OmniGibson cameras look down their -z axis, +x right, +y up in the image.
# ---------------------------------------------------------------------------------------------------------------
def project(cam, world_pos):
    K = cam.intrinsic_matrix
    pos, quat = cam.get_position_orientation()
    p = T.quat2mat(quat).T @ (th.as_tensor(world_pos, dtype=th.float32) - pos)
    if p[2] >= 0:
        return None
    return (float(K[0, 0] * p[0] / -p[2] + K[0, 2]), float(K[1, 1] * -p[1] / -p[2] + K[1, 2]))


def backproject(cam, u, v, d):
    K = cam.intrinsic_matrix
    pc = th.tensor([(u - float(K[0, 2])) / float(K[0, 0]) * d, -(v - float(K[1, 2])) / float(K[1, 1]) * d, -d])
    pos, quat = cam.get_position_orientation()
    return T.quat2mat(quat) @ pc + pos


def box_to_point(cam, box, depth):
    """3D point for a VLM bounding box: the box centre pixel at the NEAR depth inside the box (the object, not the
    table behind it). Returns (world point, pixel)."""
    x1, y1, x2, y2 = box
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    # inner half of the box, nearest 25% of the depths there = the object's visible surface
    xi1, xi2 = int(max(0, x1 + (x2 - x1) / 4)), int(min(W - 1, x2 - (x2 - x1) / 4))
    yi1, yi2 = int(max(0, y1 + (y2 - y1) / 4)), int(min(H - 1, y2 - (y2 - y1) / 4))
    patch = depth[yi1:yi2 + 1, xi1:xi2 + 1]
    vals = patch[np.isfinite(patch) & (patch > 0.02)]
    if vals.size == 0:
        return None, (u, v)
    d = float(np.percentile(vals, 25))
    return backproject(cam, u, v, d), (u, v)


def cam_obs(cam):
    o = cam.get_obs()[0]
    return o["rgb"][..., :3].cpu().numpy().astype(np.uint8), o["depth_linear"].cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------
class Harness:
    def __init__(self, env, vlm, args):
        self.env, self.vlm, self.args = env, vlm, args
        self.robot = env.scene.robots[0]
        self.wrist = list(self.robot.sensors.values())[0]
        self.ext = env.external_sensors[0]["ext_cam"]
        self.objs = {name: env.scene.object_registry("name", f"obj_{name}") for name, *_ in TASKS}
        self.steps, self.record, self.boxes = 0, None, {}

    # ---- scene ----
    def reset(self, name, layout):
        env = self.env
        env.reset()
        for k, (n, *_) in enumerate(TASKS):
            o = self.objs[n]
            o.set_position_orientation(position=th.tensor(parking_spot(k)), orientation=th.tensor([0, 0, 0, 1.0]))
            o.keep_still()
        pick_common.TARGET_NAME = f"obj_{name}"
        layout = dict(layout, apple_pos=[layout["apple_pos"][0], layout["apple_pos"][1], TABLE_TOP_Z + 0.08])
        place_layout(env, layout)
        aim_exterior_cam(self.ext, layout)
        if og.sim.viewer_camera is not None:
            og.sim.viewer_camera.set_position_orientation(*self.ext.get_position_orientation())
        self.steps, self.boxes, self.stalls = 0, {}, 0
        hold = self.eef().clone()
        for _ in range(40):  # let the object settle, arm still, gripper open
            self._step(hold, OPEN, count=False)
        obj = self.objs[name]
        return obj, obj.get_position_orientation()[0][2].item()

    # ---- low level ----
    def eef(self):
        return self.robot.get_relative_eef_pose()[0]

    def _step(self, target, grip, count=True):
        act = th.cat([th.as_tensor(target, dtype=th.float32) - self.eef(), th.tensor([grip])])
        self.env.step(to_env_action(self.env, {"eef": self.eef()}, act))
        if count:
            self.steps += 1
            if self.record:
                self.record(self)

    def out_of_time(self):
        return self.steps >= MAX_STEPS

    def move_to(self, goal, grip, lead=MAX_STEP, tol=0.01, max_steps=150, stall_steps=None):
        """
        Same 'carrot' rule as the scripted expert: command a point at most @lead ahead of the gripper.
        Returns "reached", "timeout", or "stalled" (no 2 mm of progress in @stall_steps steps, if given).
        (A stricter rule, "< 1 cm per 20 steps, timeouts count too", was tried: it fired the detours far more often,
        they did not escape the IK dead end, and it scored 1/7 instead of 3/7 on the failed apple layouts.)
        """
        goal = th.as_tensor(goal, dtype=th.float32)
        best, since_best = 1e9, 0
        for _ in range(max_steps):
            if self.out_of_time():
                return "timeout"
            d = goal - self.eef()
            n = float(th.norm(d))
            if n < tol:
                return "reached"
            if n < best - 0.002:
                best, since_best = n, 0
            else:
                since_best += 1
                if stall_steps and since_best >= stall_steps:
                    return "stalled"
            self._step(self.eef() + (d if n <= lead else d * lead / n), grip)
        return "timeout"

    def descend(self, grasp):
        """
        Descend onto @grasp. v3: if the arm stops making progress (the step-by-step IK walks into a joint-limit dead
        end, the same stall the scripted expert had), back up, approach from a point 3 cm to the side, and try again.
        """
        status = self.move_to(grasp, OPEN, lead=DESCEND_STEP, tol=0.006,
                              stall_steps=25 if self.args.stall_recovery else None)
        for dx, dy in ((-0.03, 0.0), (0.0, 0.03), (0.0, -0.03))[: 3 if self.args.stall_recovery else 0]:
            if status != "stalled":
                break
            self.stalls += 1
            self.move_to(self.eef() + th.tensor([0, 0, 0.03]), OPEN)
            self.move_to(th.tensor([float(grasp[0]) + dx, float(grasp[1]) + dy, float(self.eef()[2])]), OPEN)
            status = self.move_to(grasp, OPEN, lead=DESCEND_STEP, tol=0.006, stall_steps=25)
        return status

    def hold(self, grip, n):
        target = self.eef().clone()
        for _ in range(n):
            if self.out_of_time():
                return
            self._step(target, grip)

    # ---- the VLM tools ----
    def locate(self, cam_name, word):
        cam = self.ext if cam_name == "ext" else self.wrist
        rgb, depth = cam_obs(cam)
        box = self.vlm.locate(rgb, word)
        self.boxes[cam_name] = box
        if box is None:
            return None
        p_world, _ = box_to_point(cam, box, depth)
        return None if p_world is None else to_base(self.robot, p_world)

    def holding(self, word):
        ext, _ = cam_obs(self.ext)
        wr, _ = cam_obs(self.wrist)
        return self.vlm.yes_no([ext, wr], f"The first image is a side view of a robot arm, the second is from a camera "
                                          f"on the robot's gripper. Is the robot holding the {word} in its gripper, "
                                          f"lifted off the table?")

    # ---- one pick attempt sequence ----
    def run(self, name, word, layout, record=None):
        obj, rest_z = self.reset(name, layout)
        self.record = record
        calls0, secs0 = self.vlm.calls, self.vlm.seconds
        true_base = lambda: to_base(self.robot, obj.get_position_orientation()[0])
        info = dict(attempts=0, vlm_said_holding=None, est_err_ext_cm=None, est_err_wrist_cm=None, failure=None)
        min_dist = 1e9
        for attempt in range(self.args.max_attempts):
            if self.out_of_time():
                break
            info["attempts"] = attempt + 1
            p = self.locate("ext", word)  # global estimate from the exterior camera
            if p is None:
                p = self.locate("wrist", word)
            if p is None:
                info["failure"] = "VLM could not find the object"
                break
            if attempt == 0:
                info["est_err_ext_cm"] = round(float(th.norm(p - true_base())) * 100, 1)
            self.move_to(p + th.tensor([0, 0, HOVER]), OPEN)  # reach above it
            p2 = self.locate("wrist", word)  # close-up re-look from the wrist camera refines the estimate
            if p2 is not None and float(th.norm(p2[:2] - p[:2])) < 0.10:
                p = p2
                if attempt == 0:
                    info["est_err_wrist_cm"] = round(float(th.norm(p - true_base())) * 100, 1)
            self.move_to(p + th.tensor([0, 0, HOVER]), OPEN)
            grasp = th.tensor([p[0], p[1], max(float(p[2]) - GRASP_BELOW_SURFACE, MIN_GRASP_Z)])
            self.descend(grasp)
            min_dist = min(min_dist, float(th.norm(self.eef() - true_base())))
            self.hold(CLOSE, 25)  # close the gripper (assisted grasping attaches the object if it is between the fingers)
            self.move_to(th.tensor([float(self.eef()[0]), float(self.eef()[1]), LIFT_Z]), CLOSE)
            self.hold(CLOSE, 10)
            said, answer = self.holding(word)
            info["vlm_said_holding"] = said
            # v2: done only if the VLM says yes AND the gripper's own grasp sensor agrees (the DP had this sensor
            # too). In v1 the VLM alone decided, and 6/96 false "yes" answers ended episodes that could have retried.
            gripper_says = self.robot.is_grasping(candidate_obj=obj).name == "TRUE"
            if said and (gripper_says or not self.args.trust_gripper):
                break
            # the VLM thinks it missed: let go, back off upward, try again from a fresh look
            self.hold(OPEN, 15)
            self.move_to(self.eef() + th.tensor([0, 0, 0.08]), OPEN)
        success, lifted = check_success(self.env, rest_z)
        truly_held = success
        if info["vlm_said_holding"] is not None and info["failure"] is None and not success:
            info["failure"] = "missed the grasp" if info["vlm_said_holding"] is False else "VLM wrongly judged success"
        info.update(success=bool(success), steps=self.steps, lifted_cm=round(lifted * 100, 1),
                    closest_cm=round(min_dist * 100, 1) if min_dist < 1e9 else None,
                    stalls_recovered=self.stalls, vlm_calls=self.vlm.calls - calls0, vlm_seconds=round(self.vlm.seconds - secs0, 1),
                    vlm_judgement_correct=None if info["vlm_said_holding"] is None else bool(info["vlm_said_holding"]) == truly_held)
        return info


# ---------------------------------------------------------------------------------------------------------------
def make_recorder(path):
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 2 * VW, VH + 24, "yuv420p"

    def write(h):
        ext, _ = cam_obs(h.ext)
        wr, _ = cam_obs(h.wrist)
        tiles = []
        for img, key in ((ext, "ext"), (wr, "wrist")):
            img = cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA)
            box = h.boxes.get(key)
            if box is not None:  # the VLM's last answer for this camera
                cv2.rectangle(img, (int(box[0] / 2), int(box[1] / 2)), (int(box[2] / 2), int(box[3] / 2)), (255, 0, 255), 2)
            tiles.append(img)
        ext, wr = tiles
        bar = np.zeros((24, 2 * VW, 3), np.uint8)
        cv2.putText(bar, f"t={h.steps:3d}  VLM calls {h.vlm.calls}  (magenta box = VLM's latest answer)",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.vstack([np.hstack([ext, wr]), bar])
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)

    def close():
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    return write, close


def pointing_test(h, args, tasks):
    """Perception only: how far is the VLM's 3D estimate from the truth, per camera and task?"""
    os.makedirs(os.path.join(OUT, "pointing"), exist_ok=True)
    rows = []
    for name, cat, mid, prompt in tasks:
        word = object_word(prompt)
        for lid in range(args.layouts):
            obj, _ = h.reset(name, sample_layout(np.random.default_rng(args.seed + lid)))
            truth = obj.get_position_orientation()[0]
            tiles = []
            for cam_name, cam in (("ext", h.ext), ("wrist", h.wrist)):
                rgb, depth = cam_obs(cam)
                box = h.vlm.locate(rgb, word)
                tp = project(cam, truth)
                # geometry check: the TRUE pixel back-projected with our depth rule (error ~ object radius if correct)
                geo = None
                if tp is not None and 0 <= tp[0] < W and 0 <= tp[1] < H:
                    s = 24
                    pt, _ = box_to_point(cam, (tp[0] - s, tp[1] - s, tp[0] + s, tp[1] + s), depth)
                    geo = None if pt is None else float(th.norm(pt - truth))
                err, hit = None, None
                if box is not None:
                    pt, _ = box_to_point(cam, box, depth)
                    err = None if pt is None else float(th.norm(pt - truth))
                    hit = tp is not None and box[0] <= tp[0] <= box[2] and box[1] <= tp[1] <= box[3]
                rows.append(dict(task=name, layout=lid, cam=cam_name, found=box is not None, box_contains_truth=hit,
                                 box=None if box is None else [round(b, 1) for b in box],
                                 true_px=None if tp is None else [round(tp[0], 1), round(tp[1], 1)],
                                 raw_answer=h.vlm.log[-1][1],
                                 err_cm=None if err is None else round(err * 100, 1),
                                 geometry_err_cm=None if geo is None else round(geo * 100, 1),
                                 truth_in_view=tp is not None and 0 <= tp[0] < W and 0 <= tp[1] < H))
                img = rgb.copy()
                if box is not None:
                    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
                if tp is not None:
                    cv2.circle(img, (int(tp[0]), int(tp[1])), 4, (0, 255, 0), -1)
                cv2.putText(img, f"{cam_name}: {'err %.1f cm' % (err * 100) if err is not None else 'not found'}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                tiles.append(img)
            sheet = cv2.resize(np.hstack(tiles), (2 * VW, VH), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(OUT, "pointing", f"{name}_L{lid}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
            r = rows[-2:]
            print(f"[{name:12s} L{lid}] ext: {r[0]['err_cm']} cm (box on object: {r[0]['box_contains_truth']}) | "
                  f"wrist: {r[1]['err_cm']} cm (box on object: {r[1]['box_contains_truth']}, in view: {r[1]['truth_in_view']}) | "
                  f"geometry check {r[0]['geometry_err_cm']}/{r[1]['geometry_err_cm']} cm", flush=True)
    with open(os.path.join(OUT, "pointing_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    for cam_name in ("ext", "wrist"):
        rr = [r for r in rows if r["cam"] == cam_name]
        errs = [r["err_cm"] for r in rr if r["err_cm"] is not None]
        print(f"{cam_name}: found {sum(r['found'] for r in rr)}/{len(rr)}, box on the object "
              f"{sum(bool(r['box_contains_truth']) for r in rr)}/{len(rr)}, 3D error median "
              f"{np.median(errs) if errs else float('nan'):.1f} cm, "
              f"geometry-only error median {np.median([r['geometry_err_cm'] for r in rr if r['geometry_err_cm'] is not None]):.1f} cm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pointing", "eval"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--layouts", type=int, default=8)
    parser.add_argument("--video-layouts", type=int, nargs="*", default=[0])
    parser.add_argument("--tag", default="run")
    parser.add_argument("--trust-gripper", type=int, default=1, help="v2: require the grasp sensor to agree (0 = v1)")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--stall-recovery", type=int, default=1, help="v3: detour when the descent stalls (0 = off)")
    parser.add_argument("--layout-ids", type=int, nargs="*", default=None, help="run exactly these layout numbers")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    args = parser.parse_args()
    tasks = [t for t in TASKS if args.tasks is None or t[0] in args.tasks]
    os.makedirs(os.path.join(OUT, "videos"), exist_ok=True)

    from vlm_backend import QwenVLM

    t0 = time.time()
    vlm = QwenVLM()  # load before the simulator (GPU memory)
    print(f"Qwen2.5-VL-7B loaded in {time.time() - t0:.0f}s, GPU {th.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    env = og.Environment(configs=build_harness_cfg())
    h = Harness(env, vlm, args)

    if args.mode == "pointing":
        pointing_test(h, args, tasks)
        og.shutdown()
        return

    results = {}
    for name, cat, mid, prompt in tasks:
        word = object_word(prompt)
        rows = []
        for lid in (args.layout_ids if args.layout_ids else range(args.layouts)):
            layout = sample_layout(np.random.default_rng(args.seed + lid))
            rec = close = None
            if lid in args.video_layouts:
                rec, close = make_recorder(os.path.join(OUT, "videos", f"vlm_{args.tag}_{name}_L{lid}.mp4"))
            t1 = time.time()
            r = h.run(name, word, layout, rec)
            if close:
                close()
            r["layout"] = lid
            rows.append(r)
            print(f"[{name:12s} L{lid}] {'SUCCESS' if r['success'] else 'fail   '} attempts={r['attempts']} steps={r['steps']:3d} "
                  f"| estimate error ext {r['est_err_ext_cm']} cm, wrist {r['est_err_wrist_cm']} cm | closest {r['closest_cm']} cm "
                  f"| VLM said holding: {r['vlm_said_holding']} (correct: {r['vlm_judgement_correct']}) "
                  f"| {r['vlm_calls']} VLM calls, {r['vlm_seconds']}s | {r['failure'] or ''} | {time.time() - t1:.0f}s", flush=True)
        results[name] = rows
        with open(os.path.join(OUT, f"results_{args.tag}.json"), "w") as f:
            json.dump(dict(args=vars(args), results=results), f, indent=1)
        print(f"== {name}: {sum(r['success'] for r in rows)}/{len(rows)} successes", flush=True)
    total = sum(r["success"] for rows in results.values() for r in rows)
    n = sum(len(rows) for rows in results.values())
    print(f"\nTOTAL: {total}/{n} = {100 * total / max(n, 1):.1f}%", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
