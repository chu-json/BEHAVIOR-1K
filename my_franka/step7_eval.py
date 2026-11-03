"""
Step 7: let the trained policy drive the robot on NEW random layouts and measure the success rate.

    OMNIGIBSON_HEADLESS=1 python my_franka/step7_eval.py --episodes 50      # numbers only
    python my_franka/step7_eval.py --episodes 5                             # watch it in the GUI
    OMNIGIBSON_HEADLESS=1 python my_franka/step7_eval.py --expert           # same layouts, scripted expert

Eval seeds start at 50,000,000 so they never overlap the training seeds (worker_id * 1,000,000 + episode).
"""
import argparse
import os

import numpy as np
import torch as th

import omnigibson as og
from pick_common import (
    MAX_EPISODE_STEPS, Expert, apple_z, apply_layout, build_cfg, check_success, get_obs, get_state,
    sample_layout, to_env_action,
)
from step6_train import DATA_DIR, Policy

EVAL_SEED = 50_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--policy", default=os.path.join(DATA_DIR, "policy.pt"))
    parser.add_argument("--expert", action="store_true", help="run the scripted expert instead (baseline)")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="layout seed base (default: the standard eval set)")
    args = parser.parse_args()

    policy = None
    if not args.expert:
        policy = Policy()
        policy.load_state_dict(th.load(args.policy, map_location="cpu")["state_dict"])
        policy.eval()

    env = og.Environment(configs=build_cfg())
    n_ok = 0
    for ep in range(args.episodes):
        layout = sample_layout(np.random.default_rng(args.seed + ep))
        apply_layout(env, layout)
        rest_z = apple_z(env)
        expert = Expert()
        # Diagnostics: where was the gripper when it first closed, how often did it flip open/close,
        # and did the apple get pushed before the grasp?
        apple0 = get_state(env, 1.0)["apple"].clone()
        first_close, n_switches, prev_grip = None, 0, 1.0
        for t in range(MAX_EPISODE_STEPS):
            s = get_state(env, prev_grip)
            a = expert.act(s) if args.expert else policy.act(get_obs(s))
            grip = float(a[3])
            if (grip < 0) != (prev_grip < 0):
                n_switches += 1
            if grip < 0 and first_close is None:
                d = s["apple"] - s["eef"]
                first_close = dict(t=t, dxy=th.norm(d[:2]).item(), dz=-d[2].item(),
                                   pushed=th.norm(s["apple"][:2] - apple0[:2]).item())
            prev_grip = grip
            env.step(to_env_action(env, s, a))
            if args.expert and expert.done:
                break
        success, lifted = check_success(env, rest_z)
        n_ok += success
        fc = (f"first close t={first_close['t']} xy-off={first_close['dxy'] * 100:.1f}cm "
              f"z-above={first_close['dz'] * 100:.1f}cm apple-pushed={first_close['pushed'] * 100:.1f}cm"
              if first_close else "never closed")
        print(f"eval ep {ep}: success={success} lifted={lifted * 100:.1f}cm | {fc} | grip switches={n_switches} "
              f"| running {n_ok}/{ep + 1} = {100 * n_ok / (ep + 1):.0f}%", flush=True)
    print(f"FINAL {'expert' if args.expert else 'policy'} success rate: {n_ok}/{args.episodes} "
          f"= {100 * n_ok / args.episodes:.1f}%", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
