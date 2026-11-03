"""
Step 11: evaluate a LeRobot-trained vision policy (diffusion / ACT) on the same 50 unseen layouts as step 7.
The policy sees ONLY the wrist camera + proprio (no apple position).

    OMNIGIBSON_HEADLESS=1 python my_franka/step11_eval_vision.py --ckpt <train_dir>/checkpoints/last/pretrained_model
    python my_franka/step11_eval_vision.py --ckpt ... --episodes 5          # watch in the GUI

LeRobot policies keep their own observation history (n_obs_steps) and action queue (they predict a chunk of
`horizon` actions, execute `n_action_steps` of them, then re-plan), so we just call select_action every step.
"""
import argparse
import time

import numpy as np
import torch as th

import omnigibson as og
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from pick_common import MAX_EPISODE_STEPS, MAX_STEP, apple_z, apply_layout, check_success, get_state, sample_layout, to_env_action
from step7_eval import EVAL_SEED
from step9_collect_vision import build_vision_cfg, proprio, wrist_rgb

TASK = "pick up the apple"


def load_policy(ckpt, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.device = device
    policy: PreTrainedPolicy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help=".../checkpoints/<step>/pretrained_model")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy, pre, post = load_policy(args.ckpt, args.device)
    print(f"loaded {policy.config.type} policy from {args.ckpt}", flush=True)
    env = og.Environment(configs=build_vision_cfg(num_envs=1))

    n_ok, infer_time, n_infer = 0, 0.0, 0
    for ep in range(args.episodes):
        layout = sample_layout(np.random.default_rng(args.seed + ep))
        apply_layout(env, layout)
        rest_z = apple_z(env)
        policy.reset()
        prev_grip, first_close = 1.0, None
        for t in range(MAX_EPISODE_STEPS):
            s = get_state(env, prev_grip)
            img = th.from_numpy(wrist_rgb(env, 0)).permute(2, 0, 1).float().div(255)[None]  # (1, 3, H, W) in [0, 1]
            obs = {"observation.images.wrist": img, "observation.state": th.from_numpy(proprio(s))[None], "task": [TASK]}
            t0 = time.time()
            with th.inference_mode():
                action = post(policy.select_action(pre(obs)))[0].cpu()
            infer_time, n_infer = infer_time + time.time() - t0, n_infer + 1
            grip = 1.0 if action[3] >= 0 else -1.0
            if grip < 0 and first_close is None:
                first_close = t
            env.step(to_env_action(env, s, th.cat([action[:3].clamp(-1, 1) * MAX_STEP, th.tensor([grip])])))
            prev_grip = grip
            if check_success(env, rest_z)[0] and s["eef"][2] > 0.17:
                break
        success, lifted = check_success(env, rest_z)
        n_ok += success
        print(f"eval ep {ep}: success={success} lifted={lifted * 100:.1f}cm steps={t + 1} first_close={first_close} "
              f"| running {n_ok}/{ep + 1} = {100 * n_ok / (ep + 1):.0f}% | avg policy call {1000 * infer_time / n_infer:.0f} ms",
              flush=True)
    print(f"FINAL vision policy success rate: {n_ok}/{args.episodes} = {100 * n_ok / args.episodes:.1f}%", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
