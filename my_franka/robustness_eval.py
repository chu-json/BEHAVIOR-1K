"""
Robustness test: how much does the camera-only diffusion policy degrade when the visuals change?

Runs the policy on the SAME 50 unseen layouts under each condition (one sim process, perturbations applied and
undone in turn), and compares each condition with the unperturbed baseline:

    baseline          what the policy was trained on
    light_0.5x        all lights at half intensity        (slightly darker)
    light_2x          all lights at double intensity      (slightly brighter)
    wall_texture      walls covered in a brick pattern
    table_texture     table covered in a wood-grain pattern
    red_apple         apple tinted red
    baseline_repeat   baseline again at the end: shows run-to-run noise AND that every perturbation was undone

Extra conditions (pass them with --conditions): light_0.25x, light_4x

    OMNIGIBSON_HEADLESS=1 python my_franka/robustness_eval.py                       # all default conditions, 50 layouts
    OMNIGIBSON_HEADLESS=1 python my_franka/robustness_eval.py --conditions baseline red_apple --episodes 20

NOTE the diffusion policy samples its actions from noise, so even an unchanged run varies a little from run to run;
baseline_repeat measures how much. Output: my_franka/data/robustness/ (results.json, one wrist image per condition,
and a montage of them).
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch as th

import omnigibson as og
from perturbations import build_perturbations
from pick_common import MAX_EPISODE_STEPS, MAX_STEP, apple_z, apply_layout, check_success, get_state, sample_layout, to_env_action
from step6_train import DATA_DIR
from step7_eval import EVAL_SEED
from step9_collect_vision import build_vision_cfg, proprio, wrist_rgb
from step11_eval_vision import TASK, load_policy

DEFAULT_CKPT = os.path.join(DATA_DIR, "train", "diffusion_all", "checkpoints", "last", "pretrained_model")
OUT_DIR = os.path.join(DATA_DIR, "robustness")
DEFAULT_CONDITIONS = ["baseline", "light_0.5x", "light_2x", "wall_texture", "table_texture", "red_apple", "baseline_repeat"]


def run_episode(env, policy, pre, post, layout):
    """One pick attempt. Returns (success, failure_reason or None, first_close_step, wrist image at the start)."""
    apply_layout(env, layout)
    start_img = wrist_rgb(env, 0)  # exactly what the policy sees on its first step (after the env reset)
    rest_z = apple_z(env)
    policy.reset()
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
        env.step(to_env_action(env, s, th.cat([action[:3].clamp(-1, 1) * MAX_STEP, th.tensor([grip])])))
        prev_grip = grip
        if check_success(env, rest_z)[0] and s["eef"][2] > 0.17:
            break
    success, lifted = check_success(env, rest_z)
    if success:
        return True, None, first_close, start_img
    if lifted < -0.10:
        return False, "apple knocked off the table", first_close, start_img
    if first_close is None:
        return False, "never closed the gripper", first_close, start_img
    return False, "closed but did not lift the apple", first_close, start_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--start", type=int, default=0, help="first layout number; layouts start..start+episodes-1 are used")
    parser.add_argument("--tag", default=None, help="output files get this suffix (needed when several runs go in parallel)")
    parser.add_argument("--conditions", nargs="*", default=DEFAULT_CONDITIONS)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    policy, pre, post = load_policy(args.ckpt, args.device)
    env = og.Environment(configs=build_vision_cfg(num_envs=1))
    perts = build_perturbations()
    unknown = [c for c in args.conditions if c not in perts and not c.startswith("baseline")]
    assert not unknown, f"unknown conditions {unknown}; choose from baseline, baseline_repeat, {list(perts)}"

    results, images = {}, []
    for cond in args.conditions:
        pert = perts.get(cond)  # None for baseline / baseline_repeat
        if pert:
            pert.apply(env)
        print(f"\n===== {cond}: {pert.description if pert else 'no change'} =====", flush=True)
        outcomes, reasons, first_closes, t0 = [], {}, [], time.time()
        for ep in range(args.start, args.start + args.episodes):
            layout = sample_layout(np.random.default_rng(args.seed + ep))
            success, reason, first_close, start_img = run_episode(env, policy, pre, post, layout)
            outcomes.append(bool(success))
            first_closes.append(first_close)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            if ep == args.start:  # what the policy sees at the start of the first layout under this condition
                cv2.imwrite(os.path.join(OUT_DIR, f"{cond}.png"), cv2.cvtColor(start_img, cv2.COLOR_RGB2BGR))
                images.append((cond, start_img))
                print(f"  first frame (layout 0, before the policy acts): mean brightness {start_img.mean():.0f}", flush=True)
            done = ep - args.start + 1
            if done % 10 == 0:
                print(f"  {done}/{args.episodes}: {sum(outcomes)} successes so far "
                      f"({(time.time() - t0) / done:.0f} s/episode)", flush=True)
        results[cond] = dict(successes=sum(outcomes), episodes=args.episodes, start=args.start, outcomes=outcomes,
                             failure_reasons=reasons,
                             description=pert.description if pert else "no change")
        if pert:
            pert.restore(env)
        failed = [i for i, ok in enumerate(outcomes) if not ok]
        print(f"  {cond}: {sum(outcomes)}/{args.episodes} = {100 * sum(outcomes) / args.episodes:.0f}% | "
              f"failed layouts {failed} | {reasons}", flush=True)
        suffix = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(OUT_DIR, f"results{suffix}.json"), "w") as f:  # saved after every condition
            json.dump(dict(ckpt=args.ckpt, results=results), f, indent=1)

    # ---- summary ----
    base = results.get("baseline")
    print("\n===== SUMMARY (same layouts for every row) =====", flush=True)
    print(f"{'condition':<18}{'success':>9}{'  lost vs baseline':>20}{'  gained':>9}   failure reasons", flush=True)
    for cond, r in results.items():
        rate = f"{r['successes']}/{r['episodes']} ({100 * r['successes'] / r['episodes']:.0f}%)"
        if base and cond != "baseline":
            lost = sum(b and not o for b, o in zip(base["outcomes"], r["outcomes"]))
            gained = sum(o and not b for b, o in zip(base["outcomes"], r["outcomes"]))
            print(f"{cond:<18}{rate:>9}{lost:>20}{gained:>9}   {r['failure_reasons']}", flush=True)
        else:
            print(f"{cond:<18}{rate:>9}{'-':>20}{'-':>9}   {r['failure_reasons']}", flush=True)

    if images:  # montage of what the policy saw
        tiles = []
        for name, img in images:
            tile = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST).copy()
            cv2.putText(tile, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        cv2.imwrite(os.path.join(OUT_DIR, f"montage{'_' + args.tag if args.tag else ''}.png"), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
    print(f"\nsaved results to {OUT_DIR}", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
