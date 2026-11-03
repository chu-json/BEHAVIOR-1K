"""Run N rollouts of a websocket policy (e.g. Comet via steering/comet_server.py) and log one CSV row each.

Uses this fork's BatchedEvaluator (num_envs=1) in-process, so Isaac Sim launches once for all rollouts.
Start the policy server first, then (behavior env, from the repo root):

    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/run_rollouts.py \
        --instances 242 --seeds 0 --max-steps 4299 --resolution full --output-dir steering/runs/one

Instances are BEHAVIOR task-instance ids loaded with mode="train" (the 2025 validation ids for
turning_on_radio live in the train range; their files are identical in the 2025 and 2026 instance sets).
Rollout i uses instance instances[i % len(instances)] and seed seeds[i] (or base_seed + i).
"""

import argparse
import csv
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)  # so hydra can import steering.compat_wrapper

# 2025 challenge validation instances for turning_on_radio: first 10 of the 20 in
# 2025-challenge-task-instances/metadata/test_instances.csv (Comet's reported "validation" protocol).
RADIO_2025_VAL = [242, 295, 211, 203, 109, 181, 197, 187, 214, 139]
WRAPPERS = {
    "full": "steering.compat_wrapper.CompatFullResWrapper",
    "224": "steering.compat_wrapper.CompatWrapper",
}
CSV_FIELDS = [
    "rollout",
    "task",
    "instance_id",
    "seed",
    "success",
    "q_score",
    "success_step",
    "steps",
    "max_steps",
    "wall_time_s",
    "steps_per_sec",
    "policy_time_s",
    "resolution",
    "world_frame_base_qvel",
    "video_path",
    "json_path",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="turning_on_radio")
    ap.add_argument("--instances", type=int, nargs="+", default=RADIO_2025_VAL)
    ap.add_argument("-n", "--num-rollouts", type=int, default=None, help="Default: one per instance.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="One per rollout. Default: base_seed + i.")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=4299, help="Episode timeout. 4299 = 2025 protocol for radio.")
    ap.add_argument("--resolution", choices=sorted(WRAPPERS), default="full", help="Camera render resolution.")
    ap.add_argument(
        "--robot-frame-base-qvel",
        action="store_true",
        help="Disable compat fix #3 (keep the fork's robot-frame base_qvel).",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-fps", type=int, default=30)
    return ap.parse_args()


def main():
    args = parse_args()
    n = args.num_rollouts or len(args.instances)
    seeds = args.seeds or [args.base_seed + i for i in range(n)]
    assert len(seeds) == n, f"got {len(seeds)} seeds for {n} rollouts"
    write_video = not args.no_video

    from omegaconf import OmegaConf

    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import seed_everything
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False  # same as eval.py headless default

    out_dir = os.path.abspath(args.output_dir)
    json_dir, video_dir = os.path.join(out_dir, "json"), os.path.join(out_dir, "videos")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "rollouts.csv")
    new_csv = not os.path.exists(csv_path)

    seed_everything(seeds[0])
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": WRAPPERS[args.resolution]},
            "policy_name": "websocket",
            "model": {
                "_target_": "omnigibson.eval.policies.WebsocketPolicy",
                "host": args.host,
                "port": args.port,
                "action_chunk_size": 0,
            },
            "headless": True,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": write_video,
            "mode": "train",
            "seed": seeds[0],
            "num_envs": 1,
            "task": {"name": args.task},
            "robot": None,
        }
    )

    with BatchedEvaluator(cfg) as evaluator, open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_csv:
            writer.writeheader()
        evaluator.env.world_frame_base_qvel = not args.robot_frame_base_qvel

        # Time spent waiting on the policy server (vs. simulating), per rollout.
        policy_time = [0.0]
        forward = evaluator.policy.forward

        def timed_forward(*a, **kw):
            t = time.perf_counter()
            out = forward(*a, **kw)
            policy_time[0] += time.perf_counter() - t
            return out

        evaluator.policy.forward = timed_forward

        for i in range(n):
            instance, seed = args.instances[i % len(args.instances)], seeds[i]
            seed_everything(seed)
            evaluator.env.rollout_seed = seed
            policy_time[0] = 0.0
            logging.info(f"=== rollout {i + 1}/{n}: instance {instance}, seed {seed}")
            t0 = time.perf_counter()
            # rollout_id=seed so file names are <task>_<instance>_<seed>.{json,mp4}
            res = evaluator.run(
                [instance],
                write_video=write_video,
                video_path=video_dir,
                metrics_dir=json_dir,
                rollout_id=seed,
                video_fps=args.video_fps,
            )[instance]
            wall = time.perf_counter() - t0
            success = bool(res["success"])
            stem = f"{args.task}_{instance}_{seed}"
            row = {
                "rollout": i,
                "task": args.task,
                "instance_id": instance,
                "seed": seed,
                "success": int(success),
                "q_score": res.get("q_score", {}).get("final"),
                # Episodes terminate on success, so the success step is the episode length.
                "success_step": res["steps"] if success else "",
                "steps": res["steps"],
                "max_steps": args.max_steps,
                "wall_time_s": round(wall, 1),
                "steps_per_sec": round(res["steps"] / wall, 2),
                "policy_time_s": round(policy_time[0], 1),
                "resolution": args.resolution,
                "world_frame_base_qvel": int(not args.robot_frame_base_qvel),
                "video_path": os.path.join(video_dir, stem + ".mp4") if write_video else "",
                "json_path": os.path.join(json_dir, stem + ".json"),
            }
            writer.writerow(row)
            f.flush()
            logging.info(f"=== rollout {i + 1}/{n} done: {row}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
