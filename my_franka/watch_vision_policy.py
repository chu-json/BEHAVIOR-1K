"""
Watch the final camera-only diffusion policy pick apples live in the GUI, one random layout after another.
The policy sees ONLY the 128x128 wrist-camera image plus 7 proprio numbers (no apple position).

    python my_franka/watch_vision_policy.py                       # layouts 0, 1, 2, ... until Ctrl+C
    python my_franka/watch_vision_policy.py --layouts 1 0 38      # specific eval layouts (same numbers as step11_eval_vision.py)
    python my_franka/watch_vision_policy.py --failures            # replay the 5 layouts the final policy fails on
    python my_franka/watch_vision_policy.py --perturb table_texture --layouts 1 2 3     # under a visual change
    python my_franka/watch_vision_policy.py --perturb red_apple --layouts 3 17 31       # layouts that failed with a red apple

--perturb choices (see perturbations.py): light_0.5x, light_2x, light_0.25x, light_4x, wall_texture, table_texture,
red_apple. The policy never saw any of these in training; table_texture is the one that breaks it completely.
    python my_franka/watch_vision_policy.py --ckpt data/train/diffusion_500/checkpoints/last/pretrained_model

For the wrist camera's own view next to a side view, record a video instead:
    OMNIGIBSON_HEADLESS=1 python my_franka/record_descent_video.py --vision-ckpt <ckpt> --layouts 1 0 38
"""
import argparse
import itertools
import math
import os

import numpy as np
import torch as th

import omnigibson as og
from pick_common import MAX_EPISODE_STEPS, MAX_STEP, apple_z, apply_layout, check_success, get_state, sample_layout, to_env_action
from step6_train import DATA_DIR
from step7_eval import EVAL_SEED
from step9_collect_vision import build_vision_cfg, proprio, wrist_rgb
from step11_eval_vision import TASK, load_policy
from perturbations import build_perturbations
from watch_policy import PAUSE_STEPS, aim_camera

DEFAULT_CKPT = os.path.join(DATA_DIR, "train", "diffusion_all", "checkpoints", "last", "pretrained_model")
FINAL_POLICY_FAILURES = [8, 13, 30, 33, 38]  # eval layouts the 962-demo, 50k-step policy fails


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--layouts", type=int, nargs="*", help="eval layout numbers (default: 0, 1, 2, ...)")
    parser.add_argument("--failures", action="store_true", help="replay the layouts this policy is known to fail")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--perturb", default=None, help="visual change to apply (see the list above)")
    args = parser.parse_args()

    policy, pre, post = load_policy(args.ckpt, args.device)
    print(f"loaded {policy.config.type} policy from {args.ckpt}", flush=True)

    env = og.Environment(configs=build_vision_cfg(num_envs=1))
    if args.perturb:
        perts = build_perturbations()
        assert args.perturb in perts, f"unknown --perturb {args.perturb}; choose from {list(perts)}"
        perts[args.perturb].apply(env)
        print(f"visual change applied: {perts[args.perturb].description}", flush=True)
    layout_ids = FINAL_POLICY_FAILURES if args.failures else (args.layouts or itertools.count())
    n_ok = n = 0
    try:
        for i in layout_ids:
            layout = sample_layout(np.random.default_rng(EVAL_SEED + i))
            apply_layout(env, layout)
            aim_camera(layout)
            rest_z = apple_z(env)
            policy.reset()
            print(f"\n--- layout {i}: apple {math.dist(layout['robot_pos'][:2], layout['apple_pos'][:2]) * 100:.0f} cm "
                  f"from the base | camera diffusion policy driving"
                  f"{' [' + args.perturb + ']' if args.perturb else ''} ---", flush=True)

            prev_grip, first_close = 1.0, None
            for t in range(MAX_EPISODE_STEPS):
                s = get_state(env, prev_grip)
                img = th.from_numpy(wrist_rgb(env, 0)).permute(2, 0, 1).float().div(255)[None]
                obs = {"observation.images.wrist": img, "observation.state": th.from_numpy(proprio(s))[None], "task": [TASK]}
                with th.inference_mode():
                    action = post(policy.select_action(pre(obs)))[0].cpu()
                grip = 1.0 if action[3] >= 0 else -1.0
                if grip < 0 and first_close is None:
                    first_close = t
                    print(f"  t={t}: gripper closes", flush=True)
                env.step(to_env_action(env, s, th.cat([action[:3].clamp(-1, 1) * MAX_STEP, th.tensor([grip])])))
                prev_grip = grip
                if check_success(env, rest_z)[0] and s["eef"][2] > 0.17:
                    break

            success, lifted = check_success(env, rest_z)
            n_ok, n = n_ok + success, n + 1
            print(f"  {'SUCCESS' if success else 'FAIL'} (apple moved {lifted * 100:+.1f} cm vertically"
                  f"{'; gripper never closed' if first_close is None else ''}) | total {n_ok}/{n}", flush=True)
            s = get_state(env, prev_grip)
            for _ in range(PAUSE_STEPS):  # hold the final pose for a moment
                env.step(to_env_action(env, s, th.tensor([0.0, 0.0, 0.0, prev_grip])))
    except KeyboardInterrupt:
        pass
    og.shutdown()


if __name__ == "__main__":
    main()
