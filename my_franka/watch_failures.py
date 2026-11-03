"""
Replay the layouts where BOTH the scripted teacher and the BC policy failed, so you can see why.

It reads data/eval_expert.log and data/eval_policy.log (written by step7_eval.py), finds the layouts that failed
in both, and on each one runs the teacher, then the BC policy, in the GUI. The terminal narrates what is
happening: which branch of the expert's rule applies right now, and how far the gripper is from the apple.

    python my_franka/watch_failures.py                   # teacher then policy on every shared failure
    python my_franka/watch_failures.py --who expert      # only the teacher
    python my_franka/watch_failures.py --layouts 13      # just one layout
"""
import argparse
import math
import os
import re

import numpy as np
import torch as th

import omnigibson as og
from pick_common import (
    CLOSE_TOL, MAX_EPISODE_STEPS, PREGRASP_CLEARANCE, XY_RECOVER, XY_TOL, Expert, apple_z, apply_layout, build_cfg,
    check_success, get_obs, get_state, sample_layout, to_env_action,
)
from step6_train import DATA_DIR, Policy
from step7_eval import EVAL_SEED
from watch_policy import PAUSE_STEPS, aim_camera

REPORT_EVERY = 30  # steps between distance printouts


def failed_layouts(log_path):
    with open(log_path, errors="ignore") as f:
        return {int(m.group(1)) for m in re.finditer(r"eval ep (\d+): success=False", f.read())}


def expert_branch(s):
    """Which branch of the expert's rule applies in state @s (mirrors Expert.act in pick_common.py)."""
    eef, apple = s["eef"], s["apple"]
    if s["prev_grip"] < 0:
        return "lift (grasped)" if s["grasping"] > 0 else "closed, waiting for grasp"
    hover_z = s["apple_top_z"] + PREGRASP_CLEARANCE
    xy_err = th.norm(apple[:2] - eef[:2]).item()
    if not (xy_err < XY_TOL or (eef[2] < hover_z - 0.02 and xy_err < XY_RECOVER)):
        return "fly to hover point"
    return "close" if th.norm(apple - eef).item() < CLOSE_TOL else "descend"


def run(env, layout_id, who, policy):
    layout = sample_layout(np.random.default_rng(EVAL_SEED + layout_id))
    apply_layout(env, layout)
    aim_camera(layout)
    rest_z = apple_z(env)
    print(f"\n=== layout {layout_id} | {who.upper()} | apple {math.dist(layout['robot_pos'][:2], layout['apple_pos'][:2]) * 100:.0f}"
          f" cm from base, {math.degrees(layout['robot_yaw']):+.0f} deg robot yaw ===", flush=True)
    expert, prev_grip, last_branch = Expert(), 1.0, None
    for t in range(MAX_EPISODE_STEPS):
        s = get_state(env, prev_grip)
        branch = expert_branch(s)
        if branch != last_branch:  # narrate every change of phase
            print(f"  t={t:3d}  phase -> {branch}", flush=True)
            last_branch = branch
        if t % REPORT_EVERY == 0:
            d = s["apple"] - s["eef"]
            print(f"  t={t:3d}  gripper->apple: horizontal {th.norm(d[:2]).item() * 100:5.1f} cm, "
                  f"vertical {-d[2].item() * 100:+5.1f} cm | gripper height {s['eef'][2].item() * 100:5.1f} cm", flush=True)
        a = expert.act(s) if who == "expert" else policy.act(get_obs(s))
        env.step(to_env_action(env, s, a))
        prev_grip = float(a[3])
        if check_success(env, rest_z)[0] and s["eef"][2] > 0.17:
            break
    success, lifted = check_success(env, rest_z)
    print(f"  RESULT: {'SUCCESS' if success else 'FAIL'} (apple moved {lifted * 100:+.1f} cm vertically)", flush=True)
    s = get_state(env, prev_grip)
    for _ in range(PAUSE_STEPS):
        env.step(to_env_action(env, s, th.tensor([0.0, 0.0, 0.0, prev_grip])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, nargs="*", help="default: layouts that failed for BOTH expert and BC")
    parser.add_argument("--who", choices=["both", "expert", "policy"], default="both")
    parser.add_argument("--policy", default=os.path.join(DATA_DIR, "policy.pt"))
    args = parser.parse_args()

    layouts = args.layouts
    if not layouts:
        shared = failed_layouts(os.path.join(DATA_DIR, "eval_expert.log")) & failed_layouts(
            os.path.join(DATA_DIR, "eval_policy.log"))
        layouts = sorted(shared)
        print(f"layouts failed by both teacher and BC policy: {layouts}", flush=True)

    policy = Policy()
    policy.load_state_dict(th.load(args.policy, map_location="cpu")["state_dict"])
    policy.eval()
    env = og.Environment(configs=build_cfg())
    try:
        for i in layouts:
            for who in (["expert", "policy"] if args.who == "both" else [args.who]):
                run(env, i, who, policy)
    except KeyboardInterrupt:
        pass
    og.shutdown()


if __name__ == "__main__":
    main()
