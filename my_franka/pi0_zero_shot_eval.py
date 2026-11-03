"""
Zero-shot pi0.5-DROID on the Franka in our Rs_int table scene, over many "pick up the <object>" tasks.

The model (lerobot/pi05_droid) was trained on the DROID dataset (real Franka arms) and has never seen this simulator.
We feed it what DROID gave it: an exterior camera image, a wrist camera image, 7 joint angles + gripper position and a
language prompt; it returns 15-step chunks of joint velocities + a gripper command, which we turn into joint-position
targets for our Franka (see pi0_common.py for the conventions).

Everything else is as in our regular setup: same table, same robot placement and object placement ranges, same layouts
(eval seeds), same success test (the object is held and lifted at least 5 cm). The only things that change are the
controller (joint positions instead of end-effector IK, 15 Hz instead of 30 Hz) and the cameras (224x224, plus an
exterior camera).

    OMNIGIBSON_HEADLESS=1 python my_franka/pi0_zero_shot_eval.py --layouts 8                 # all tasks, 8 layouts each
    OMNIGIBSON_HEADLESS=1 python my_franka/pi0_zero_shot_eval.py --tasks apple mug --layouts 3 --video-layouts 0 1 2
    python my_franka/pi0_zero_shot_eval.py --tasks apple --layouts 3                          # watch live in the GUI

Output: my_franka/data/pi0/results_<tag>.json, videos in my_franka/data/pi0/videos/.
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
from omnigibson.macros import gm

import pick_common
from pi0_common import JOINT_DELTA_PER_UNIT, PI0_DIR, PI05DroidZeroShot
from pick_common import TABLE_TOP_Z, build_cfg, place_layout, sample_layout
from step2_scene_franka_apple import look_at
from step7_eval import EVAL_SEED

# DROID's cameras are 16:9 (the released data is 320x180); the model letterboxes them to 224x224 itself. Rendering square
# images skips those bars, and on real DROID frames that alone lowered prediction quality (pi0_droid_realdata_check.py).
IMG_W, IMG_H = 320, 180
CONTROL_HZ = 15
FINGER_OPEN = 0.04  # Franka finger joint position when open (m)
SUCCESS_LIFT = 0.05
# DROID's reset pose (from memory of the DROID robot code): arm folded forward over the workspace. Our default Franka pose
# is very different (joint 2 at -1.3 rad; DROID's data covers -0.84..1.39), so for a fair zero-shot test we start here.
DROID_HOME = [0.0, -math.pi / 5, 0.0, -4 * math.pi / 5, 0.0, 3 * math.pi / 5, 0.0]
# (name, BEHAVIOR dataset category, model id, prompt)
TASKS = [
    ("apple", "apple", "agveuv", "pick up the apple"),
    ("orange", "orange", "cwcefi", "pick up the orange"),
    ("banana", "banana", "verqwv", "pick up the banana"),
    ("lemon", "lemon", "asdada", "pick up the lemon"),
    ("pear", "pear", "clvnqe", "pick up the pear"),
    ("potato", "potato", "bpwohr", "pick up the potato"),
    ("carrot", "carrot", "aucrah", "pick up the carrot"),
    ("mug", "mug", "dhnxww", "pick up the mug"),
    ("coffee_cup", "coffee_cup", "ckkwmj", "pick up the coffee cup"),
    ("can_of_soda", "can_of_soda", "bfrzvk", "pick up the can of soda"),
    ("toy_figure", "toy_figure", "acymin", "pick up the toy figure"),
    ("sponge", "sponge", "aewrov", "pick up the sponge"),
]


def parking_spot(k):
    """Unused objects wait far away; OmniGibson has an infinite floor plane, so they simply rest on it."""
    return [40.0 + 2.0 * k, 40.0, 0.2]


def build_pi0_cfg(gripper="robotiq"):
    """
    @gripper: "robotiq" (Robotiq 2F-85, what DROID uses) or "panda" (the Franka hand used everywhere else in this project).
    The Robotiq makes the wrist camera see the same kind of fingers DROID's does, and is ~3 cm longer than the Panda hand.
    """
    cfg = build_cfg()
    cfg["env"]["action_frequency"] = CONTROL_HZ  # DROID controls at 15 Hz
    cfg["env"]["rendering_frequency"] = CONTROL_HZ  # headless runs need rendering dt == step dt
    robot = cfg["robots"][0]
    if gripper == "robotiq":
        robot["end_effector"] = "robotiq"
    robot["obs_modalities"] = ["rgb"]
    robot["sensor_config"] = {"VisionSensor": {"sensor_kwargs": {"image_height": IMG_H, "image_width": IMG_W}}}
    # absolute joint-position targets (7 arm joints), binary gripper
    robot["controller_config"]["arm_0"] = {
        "name": "JointController", "motor_type": "position", "use_delta_commands": False,
        "command_input_limits": None, "command_output_limits": None,
    }
    cfg["objects"] = [
        {"type": "DatasetObject", "name": f"obj_{name}", "category": cat, "model": model, "position": parking_spot(k),
         "orientation": [0, 0, 0, 1]}
        for k, (name, cat, model, _) in enumerate(TASKS)
    ]
    cfg["env"]["external_sensors"] = [{
        "sensor_type": "VisionSensor", "name": "ext_cam", "modalities": ["rgb"],
        "sensor_kwargs": {"image_height": IMG_H, "image_width": IMG_W},
        "position": [0.0, 0.0, 2.0], "orientation": [0, 0, 0, 1], "pose_frame": "world", "include_in_obs": False,
    }]
    return cfg


def to_world(layout, local):
    """Point given in the robot's frame (x forward, y left) -> world coordinates."""
    yaw, (rx, ry, rz) = layout["robot_yaw"], layout["robot_pos"]
    c, s = math.cos(yaw), math.sin(yaw)
    return [rx + c * local[0] - s * local[1], ry + s * local[0] + c * local[1], rz + local[2]]


def aim_exterior_cam(cam, layout):
    """Exterior camera like DROID's: off to the robot's right and slightly ahead, above the table, looking at the workspace.
    (My first guess, 0.85 m ahead and to the left, ended up outside the room and rendered black.)"""
    eye = to_world(layout, [0.20, -0.80, 0.55])
    target = to_world(layout, [0.30, 0.0, 0.10])
    cam.set_position_orientation(position=th.tensor(eye), orientation=look_at(eye, target))


def rgb_of(cam):
    return cam.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype(np.uint8)


class Episode:
    """Holds the handles for one environment; run(task, layout) performs one pick attempt."""

    def __init__(self, env, model, args):
        self.env, self.model, self.args = env, model, args
        self.robot = env.scene.robots[0]
        self.wrist = list(self.robot.sensors.values())[0]
        self.ext = env.external_sensors[0]["ext_cam"]
        self.arm_idx = self.robot.arm_control_idx[self.robot.default_arm]
        self.grip_idx = self.robot.gripper_control_idx[self.robot.default_arm]
        self.objs = {name: env.scene.object_registry("name", f"obj_{name}") for name, *_ in TASKS}
        if args.hfov:  # horizontal field of view in degrees for both cameras (default: OmniGibson's ~63 deg)
            for cam in (self.wrist, self.ext):
                cam.focal_length = cam.horizontal_aperture / (2 * math.tan(math.radians(args.hfov) / 2))
        self.hfov = 2 * math.degrees(math.atan(self.wrist.horizontal_aperture / (2 * self.wrist.focal_length)))

    def q(self):
        return self.robot.get_joint_positions()[self.arm_idx]

    def state8(self):
        """7 joint angles + DROID-style gripper position: 0 = open, 1 = closed."""
        q = self.robot.get_joint_positions()[self.grip_idx]
        lo = self.robot.joint_lower_limits[self.grip_idx]
        hi = self.robot.joint_upper_limits[self.grip_idx]
        frac = float(((q - lo) / (hi - lo)).mean())  # 0 at the lower joint limit, 1 at the upper
        closed = frac if self.args.gripper == "robotiq" else 1.0 - frac  # Robotiq closes upward, Panda fingers close toward 0
        return np.concatenate([self.q().numpy(), [float(np.clip(closed, 0.0, 1.0))]])

    def step(self, target_q, closed):
        self.env.step(th.cat([th.as_tensor(target_q, dtype=th.float32), th.tensor([-1.0 if closed else 1.0])]))

    def reset(self, name, layout):
        env = self.env
        env.reset()
        for k, (n, *_) in enumerate(TASKS):  # everything parked, then the target goes on the table
            o = self.objs[n]
            o.set_position_orientation(position=th.tensor(parking_spot(k)), orientation=th.tensor([0, 0, 0, 1.0]))
            o.keep_still()
        pick_common.TARGET_NAME = f"obj_{name}"
        layout = dict(layout, apple_pos=[layout["apple_pos"][0], layout["apple_pos"][1], TABLE_TOP_Z + 0.08])
        place_layout(env, layout)
        if self.args.home == "droid":
            q_all = self.robot.get_joint_positions()
            q_all[self.arm_idx] = th.tensor(DROID_HOME, dtype=q_all.dtype)
            self.robot.set_joint_positions(q_all)
            self.robot.keep_still()
        aim_exterior_cam(self.ext, layout)
        if og.sim.viewer_camera is not None:  # GUI runs: show the exterior camera's view in the viewport
            og.sim.viewer_camera.set_position_orientation(*self.ext.get_position_orientation())
        hold = self.q().clone()
        for _ in range(30):  # 2 s: let the object settle on the table while the arm holds still
            self.step(hold, closed=False)
        return self.objs[name]

    def run(self, name, prompt, layout, record=None):
        obj = self.reset(name, layout)
        obj_rest_z = obj.get_position_orientation()[0][2].item()
        self.model.last_text = None
        min_dist, max_lift, first_close, close_dist, held_steps = 1e9, -1e9, None, None, 0
        t, action_chunk, k = 0, None, 0
        tracked, commanded = 0.0, 0.0  # how far the joints really moved vs how far we asked them to (controller check)
        while t < self.args.max_steps:
            if action_chunk is None or k >= self.args.open_loop:
                action_chunk = self.model.predict(rgb_of(self.ext), rgb_of(self.wrist), self.state8(), prompt)
                k = 0
            a = action_chunk[k]
            k += 1
            q = self.q()
            target = q + th.as_tensor(np.clip(a[:7], -1, 1), dtype=th.float32) * self.args.joint_delta
            closed = bool(a[7] > 0.5)
            self.step(target, closed)
            tracked += float(th.norm(self.q() - q))
            commanded += float(th.norm(target - q))
            t += 1
            eef = self.robot.get_eef_position()
            opos = obj.get_position_orientation()[0]
            d = float(th.norm(eef - opos))
            min_dist = min(min_dist, d)
            if closed and first_close is None:
                first_close, close_dist = t, d
            lift = float(opos[2]) - obj_rest_z
            max_lift = max(max_lift, lift)
            if record is not None:
                record(rgb_of(self.ext), rgb_of(self.wrist), t, d, closed)
            held = self.robot.is_grasping(candidate_obj=obj).name == "TRUE" and lift > SUCCESS_LIFT
            held_steps = held_steps + 1 if held else 0
            if held_steps >= 8:  # held and lifted for ~0.5 s: success
                break
        success = held_steps >= 8
        return dict(success=bool(success), steps=t, min_dist_cm=round(min_dist * 100, 1), max_lift_cm=round(max_lift * 100, 1),
                    first_close=first_close, close_dist_cm=None if close_dist is None else round(close_dist * 100, 1),
                    tracking=round(tracked / max(commanded, 1e-9), 2))


def make_recorder(path):
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=CONTROL_HZ)
    stream.width, stream.height, stream.pix_fmt = 2 * IMG_W, IMG_H + 24, "yuv420p"

    def write(ext, wrist, t, dist, closed):
        bar = np.zeros((24, 2 * IMG_W, 3), np.uint8)
        cv2.putText(bar, f"t={t:3d}  gripper-object {dist * 100:5.1f} cm  gripper {'CLOSED' if closed else 'open'}",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.vstack([np.hstack([ext, wrist]), bar])
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)

    def close():
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    return write, close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", default=None, help=f"subset of {[t[0] for t in TASKS]} (default: all)")
    parser.add_argument("--layouts", type=int, default=8, help="layouts per task (layout numbers 0..N-1)")
    parser.add_argument("--max-steps", type=int, default=300, help="15 Hz control steps per episode (300 = 20 s)")
    parser.add_argument("--open-loop", type=int, default=8, help="actions executed from each 15-step chunk (openpi DROID: 8)")
    parser.add_argument("--joint-delta", type=float, default=JOINT_DELTA_PER_UNIT, help="rad of joint motion per unit of normalized velocity")
    parser.add_argument("--home", choices=["droid", "ours"], default="droid", help="initial arm pose")
    parser.add_argument("--hfov", type=float, default=None, help="horizontal field of view (deg) for both cameras")
    parser.add_argument("--gripper", choices=["robotiq", "panda"], default="robotiq", help="end effector")
    parser.add_argument("--video-layouts", type=int, nargs="*", default=[0], help="layouts to record as MP4 ([] = none)")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    args = parser.parse_args()

    tasks = [t for t in TASKS if args.tasks is None or t[0] in args.tasks]
    os.makedirs(os.path.join(PI0_DIR, "videos"), exist_ok=True)
    t0 = time.time()
    model = PI05DroidZeroShot()  # load BEFORE the simulator: the weights briefly need a lot of GPU memory
    print(f"pi0.5-DROID loaded in {time.time() - t0:.0f}s", flush=True)
    env = og.Environment(configs=build_pi0_cfg(args.gripper))
    ep = Episode(env, model, args)
    print(f"cameras {IMG_W}x{IMG_H}, horizontal field of view {ep.hfov:.0f} deg, gripper {args.gripper}, "
          f"open gripper state {ep.state8()[-1]:.2f} (should be 0.00)", flush=True)

    results = {}
    for name, cat, mid, prompt in tasks:
        rows = []
        for lid in range(args.layouts):
            layout = sample_layout(np.random.default_rng(args.seed + lid))
            record = close = None
            if lid in args.video_layouts:
                record, close = make_recorder(os.path.join(PI0_DIR, "videos", f"pi05_droid_{args.tag}_{name}_L{lid}.mp4"))
            t1 = time.time()
            r = ep.run(name, prompt, layout, record)
            if close:
                close()
            r["layout"] = lid
            rows.append(r)
            print(f"[{name:12s} L{lid}] {'SUCCESS' if r['success'] else 'fail   '} steps={r['steps']:3d} closest approach "
                  f"{r['min_dist_cm']:5.1f} cm | max lift {r['max_lift_cm']:5.1f} cm | first close {r['first_close']} "
                  f"(at {r['close_dist_cm']} cm) | joints followed {r['tracking']:.2f} of commanded | {time.time() - t1:.0f}s", flush=True)
        results[name] = rows
        with open(os.path.join(PI0_DIR, f"results_{args.tag}.json"), "w") as f:
            json.dump(dict(args=vars(args), prompt={t[0]: t[3] for t in TASKS}, results=results), f, indent=1)
        n_ok = sum(r["success"] for r in rows)
        print(f"== {name}: {n_ok}/{len(rows)} successes | mean closest approach "
              f"{np.mean([r['min_dist_cm'] for r in rows]):.1f} cm | closed the gripper in "
              f"{sum(r['first_close'] is not None for r in rows)}/{len(rows)}", flush=True)

    total = sum(r["success"] for rows in results.values() for r in rows)
    n = sum(len(rows) for rows in results.values())
    print(f"\nTOTAL: {total}/{n} = {100 * total / max(n, 1):.1f}%", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
