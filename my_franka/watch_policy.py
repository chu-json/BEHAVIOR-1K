"""
Watch the trained policy (or the scripted expert) pick apples live in the GUI, one random layout after another.

    python my_franka/watch_policy.py                         # policy, endless random layouts
    python my_franka/watch_policy.py --expert                # the scripted teacher instead
    python my_franka/watch_policy.py --layouts 0 13 30 38    # replay specific eval layouts (the known failures)
    python my_franka/watch_policy.py --policy data/policy_rl.pt   # a different checkpoint (e.g. after RL)

Layout numbers are the same as step7_eval.py's episode numbers, so "eval ep 13" here is the exact same scene.
The camera re-aims at every new layout. Ctrl+C to quit.
"""
import argparse
import itertools
import math
import os

import numpy as np
import torch as th

import omnigibson as og
from pick_common import (
    MAX_EPISODE_STEPS, Expert, apple_z, apply_layout, build_cfg, check_success, get_obs, get_state,
    sample_layout, to_env_action,
)
from step2_scene_franka_apple import look_at
from step6_train import DATA_DIR, Policy
from step7_eval import EVAL_SEED

PAUSE_STEPS = 45  # ~1.5 s of holding still after each episode, so you can see the result


def aim_camera(layout):
    """Put the camera behind-left of the robot (relative to its yaw), looking at the apple."""
    if og.sim.viewer_camera is None:  # headless runs disable the viewer camera
        return
    rx, ry, rz = layout["robot_pos"]
    yaw = layout["robot_yaw"]
    back, side, up = -0.8, -0.6, 0.65  # in the robot's own frame
    eye = [rx + back * math.cos(yaw) - side * math.sin(yaw), ry + back * math.sin(yaw) + side * math.cos(yaw), rz + up]
    target = layout["apple_pos"]
    og.sim.viewer_camera.set_position_orientation(position=th.tensor(eye), orientation=look_at(eye, target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--policy", default=os.path.join(DATA_DIR, "policy.pt"))
    parser.add_argument("--layouts", type=int, nargs="*", help="eval layout numbers to replay (default: 0, 1, 2, ...)")
    args = parser.parse_args()

    policy = None
    if not args.expert:
        policy = Policy()
        policy.load_state_dict(th.load(args.policy, map_location="cpu")["state_dict"])
        policy.eval()
    who = "expert" if args.expert else f"policy ({os.path.basename(args.policy)})"

    env = og.Environment(configs=build_cfg())
    layout_ids = args.layouts if args.layouts else itertools.count()
    n_ok = n = 0
    try:
        for i in layout_ids:
            layout = sample_layout(np.random.default_rng(EVAL_SEED + i))
            apply_layout(env, layout)
            aim_camera(layout)
            rest_z = apple_z(env)
            print(f"\n--- layout {i}: apple {math.dist(layout['robot_pos'][:2], layout['apple_pos'][:2]) * 100:.0f} cm "
                  f"from the base | {who} driving ---", flush=True)

            expert, prev_grip = Expert(), 1.0
            for t in range(MAX_EPISODE_STEPS):
                s = get_state(env, prev_grip)
                a = expert.act(s) if args.expert else policy.act(get_obs(s))
                if float(a[3]) < 0 <= prev_grip:
                    print(f"  t={t}: gripper closes", flush=True)
                env.step(to_env_action(env, s, a))
                prev_grip = float(a[3])
                if check_success(env, rest_z)[0] and s["eef"][2] > 0.17:
                    break  # lifted and holding: no need to wait out the full episode

            success, lifted = check_success(env, rest_z)
            n_ok, n = n_ok + success, n + 1
            print(f"  {'SUCCESS' if success else 'FAIL'} (apple lifted {lifted * 100:.1f} cm) | "
                  f"total {n_ok}/{n}", flush=True)
            s = get_state(env, prev_grip)
            for _ in range(PAUSE_STEPS):  # hold the final pose for a moment
                env.step(to_env_action(env, s, th.tensor([0.0, 0.0, 0.0, prev_grip])))
    except KeyboardInterrupt:
        pass
    og.shutdown()


if __name__ == "__main__":
    main()
