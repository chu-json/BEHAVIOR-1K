"""
Record an MP4 showing what the WRIST CAMERA sees during a pick, next to a third-person side view.
It shows how much of the wrist view the apple takes up on the final descent (about 40% at the grasp in the
layouts checked) and lets you compare the drivers below.

    OMNIGIBSON_HEADLESS=1 python my_franka/record_descent_video.py                      # expert, eval layout 1
    OMNIGIBSON_HEADLESS=1 python my_franka/record_descent_video.py --layouts 1 2 6      # several layouts, one video
    OMNIGIBSON_HEADLESS=1 python my_franka/record_descent_video.py --policy data/policy.pt   # BC policy drives
    OMNIGIBSON_HEADLESS=1 python my_franka/record_descent_video.py --side 1             # side camera on the other side
    OMNIGIBSON_HEADLESS=1 python my_franka/record_descent_video.py --vision-ckpt <ckpt> --perturb table_texture   # under a visual change
    OMNIGIBSON_HEADLESS=1 python my_franka/record_descent_video.py --vision-ckpt data/train/diffusion_500/checkpoints/last/pretrained_model
        # the camera-only diffusion policy drives (it sees the 128x128 wrist image shown on the left, upscaled)

Output: my_franka/data/videos/descent_<layouts>.mp4 (H.264, plays in browsers / Slack)
Left: wrist camera. Right: side camera. Overlay: phase, gripper-to-apple distance, and how much of the wrist
image the apple covers.
"""
import argparse
import math
import os

import av
import cv2
import numpy as np
import torch as th

import omnigibson as og
from pick_common import (
    MAX_EPISODE_STEPS, MAX_STEP, Expert, apple_z, apply_layout, build_cfg, check_success, get_obs, get_state,
    sample_layout, to_env_action,
)
from step2_scene_franka_apple import look_at
from step6_train import DATA_DIR, Policy
from step7_eval import EVAL_SEED
from watch_failures import expert_branch

RES = 256  # higher than a policy would use, so the video is easy to watch
HOLD_FRAMES = 30  # repeat the final frame ~1 s


def build_video_cfg(wrist_res=RES):
    cfg = build_cfg()
    robot = cfg["robots"][0]
    robot["obs_modalities"] = ["rgb", "seg_instance"]  # segmentation is only used to measure apple coverage
    robot["sensor_config"] = {"VisionSensor": {"sensor_kwargs": {"image_height": wrist_res, "image_width": wrist_res}}}
    cfg["env"]["external_sensors"] = [{
        "sensor_type": "VisionSensor",
        "name": "side_cam",
        "modalities": ["rgb"],
        "sensor_kwargs": {"image_height": RES, "image_width": RES},
        "position": [0.0, 0.0, 2.0],  # placeholder, re-aimed at every layout
        "orientation": [0, 0, 0, 1],
        "pose_frame": "world",
        "include_in_obs": False,
    }]
    return cfg


def aim_side_cam(cam, layout, side):
    """Look at the apple from the robot's left (+1) or right (-1), slightly above table height."""
    ax, ay, az = layout["apple_pos"]
    yaw = layout["robot_yaw"]
    perp = (-math.sin(yaw) * side, math.cos(yaw) * side)  # unit vector to the robot's left/right
    eye = [ax + 0.32 * perp[0] - 0.08 * math.cos(yaw), ay + 0.32 * perp[1] - 0.08 * math.sin(yaw), az + 0.12]
    target = [ax, ay, az + 0.07]
    cam.set_position_orientation(position=th.tensor(eye), orientation=look_at(eye, target))
    for _ in range(3):  # the new camera pose only shows up after a render
        og.sim.render()


def rgb(cam):
    img = cam.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
    return img if img.shape[0] == RES else cv2.resize(img, (RES, RES), interpolation=cv2.INTER_NEAREST)


def apple_coverage(wrist, apple):
    """Fraction of wrist-camera pixels that belong to the apple (from semantic segmentation)."""
    obs, info = wrist.get_obs()
    seg = obs["seg_instance"].cpu().numpy()
    ids = [i for i, name in info["seg_instance"].items() if name == apple.name or str(name).endswith("/apple")]
    return float(np.isin(seg, ids).mean()) if ids else 0.0


BAR_H = 64  # info bar under the two images


def title(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def info_bar(lines, width):
    bar = np.zeros((BAR_H, width, 3), dtype=np.uint8)
    for k, text in enumerate(lines):
        cv2.putText(bar, text, (8, 18 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, nargs="*", default=[1])
    parser.add_argument("--policy", default=None, help="BC policy checkpoint; default: the scripted expert drives")
    parser.add_argument("--side", type=int, choices=[1, -1], default=-1, help="side camera on robot's left (1) / right (-1)")
    parser.add_argument("--vision-ckpt", default=None, help="LeRobot checkpoint of a camera-only policy to drive")
    parser.add_argument("--perturb", default=None, help="visual perturbation to apply, e.g. table_texture, red_apple, "
                        "wall_texture, light_0.5x (see perturbations.py)")
    args = parser.parse_args()

    policy = vision = None
    if args.policy:
        policy = Policy()
        policy.load_state_dict(th.load(args.policy, map_location="cpu")["state_dict"])
        policy.eval()
    if args.vision_ckpt:
        from step11_eval_vision import TASK, load_policy
        from step9_collect_vision import IMG_RES, proprio
        vision, pre, post = load_policy(args.vision_ckpt, "cuda")
    driver = "camera diffusion policy" if vision else ("state BC policy" if policy else "scripted expert")
    if args.perturb:
        driver += f" [{args.perturb}]"

    # the camera policy must see exactly what it was trained on: a native 128x128 render
    env = og.Environment(configs=build_video_cfg(IMG_RES if vision else RES))
    if args.perturb:
        from perturbations import build_perturbations

        build_perturbations()[args.perturb].apply(env)
    robot = env.scene.robots[0]
    apple = env.scene.object_registry("name", "apple")
    wrist = list(robot.sensors.values())[0]
    side_cam = env.external_sensors[0]["side_cam"]

    out_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(out_dir, exist_ok=True)
    tag = ("vision_" if vision else ("bc_" if policy else "")) + (f"{args.perturb}_" if args.perturb else "")
    out_path = os.path.join(out_dir, f"descent_{tag}{'_'.join(map(str, args.layouts))}.mp4")
    container = av.open(out_path, mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = 2 * RES, RES + BAR_H, "yuv420p"

    def write(frame):
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)

    for layout_id in args.layouts:
        layout = sample_layout(np.random.default_rng(EVAL_SEED + layout_id))
        apply_layout(env, layout)
        aim_side_cam(side_cam, layout, args.side)
        rest_z = apple_z(env)
        expert, prev_grip, frame = Expert(), 1.0, None
        if vision:
            vision.reset()
        print(f"recording layout {layout_id} ({driver})", flush=True)
        for t in range(MAX_EPISODE_STEPS):
            s = get_state(env, prev_grip)
            if vision:
                img = th.from_numpy(wrist.get_obs()[0]["rgb"][..., :3].cpu().numpy()).permute(2, 0, 1).float().div(255)[None]
                obs = {"observation.images.wrist": img, "observation.state": th.from_numpy(proprio(s))[None], "task": [TASK]}
                with th.inference_mode():
                    out = post(vision.select_action(pre(obs)))[0].cpu()
                a = th.cat([out[:3].clamp(-1, 1) * MAX_STEP, th.tensor([1.0 if out[3] >= 0 else -1.0])])
            else:
                a = expert.act(s) if policy is None else policy.act(get_obs(s))
            env.step(to_env_action(env, s, a))
            prev_grip = float(a[3])

            d = s["apple"] - s["eef"]
            cover = apple_coverage(wrist, apple)
            images = np.hstack([title(rgb(wrist), "WRIST CAMERA"), title(rgb(side_cam), "SIDE VIEW")])
            frame = np.vstack([images, info_bar([
                f"layout {layout_id} | {driver} | t={t:3d} | phase: {expert_branch(s)}",
                f"gripper -> apple centre: {th.norm(d[:2]).item() * 100:4.1f} cm horizontal, {-d[2].item() * 100:+5.1f} cm vertical",
                f"apple covers {cover * 100:3.0f}% of the wrist image",
            ], 2 * RES)])
            write(frame)
            if t % 30 == 0:
                print(f"  t={t:3d} phase={expert_branch(s):<26} apple covers {cover * 100:3.0f}% of wrist image", flush=True)
            if check_success(env, rest_z)[0] and s["eef"][2] > 0.17:
                break
        success, _ = check_success(env, rest_z)
        print(f"  layout {layout_id}: {'SUCCESS' if success else 'FAIL'}", flush=True)
        final = frame.copy()
        final[RES:] = info_bar([f"layout {layout_id} | {driver} | RESULT: {'SUCCESS' if success else 'FAIL'}"], 2 * RES)
        for _ in range(HOLD_FRAMES):
            write(final)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"saved {out_path}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
