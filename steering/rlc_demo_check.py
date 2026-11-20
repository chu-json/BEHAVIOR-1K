"""Open-loop check of the RLC policy on a 2025 radio demo: feed ground-truth observations (images + state from the
demo) through the same path our server uses and compare the predicted chunk with the demo's recorded actions.

    cd /home/behavior/ext/rlc/solution
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu XLA_FLAGS=--xla_disable_hlo_passes=while-loop-invariant-code-motion \
        .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/rlc_demo_check.py --frames 0 300 900

Image sources: "full" = the 720/480 demo videos resize_with_pad'ed by RLC's wrapper (what our full-res client sends),
"224" = RLC's own training images (IliaLarchenko/behavior_224_rgb). State: the 256-d demo state's fields placed at the
fork's 61-d indices (RLC only reads base_qvel, trunk/arm/gripper qpos).
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EP = 40
DEMO_FULL = "/home/behavior/ext/rlc/demofull/videos/task-0000"
DEMO_224 = "/home/behavior/ext/rlc/demo224/videos/task-0000"
PARQUET = "/home/behavior/ext/data/2025-challenge-demos/data/task-0000/episode_{:08d}.parquet"
CAMS = {  # obs key -> dataset video key
    "robot_r1::robot_r1:zed_link:Camera:0::rgb": "head",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": "left_wrist",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": "right_wrist",
}
# v3.7.2 256-d indices (steering/norm_stats_check.py) -> fork 61-d indices (comet_server.PROPRIOCEPTION_INDICES)
MAP_256_TO_61 = [
    (np.s_[253:256], np.s_[0:3]),  # base_qvel
    (np.s_[158:165], np.s_[3:10]),  # arm_left_qpos
    (np.s_[193:195], np.s_[24:26]),  # gripper_left_qpos
    (np.s_[197:204], np.s_[28:35]),  # arm_right_qpos
    (np.s_[232:234], np.s_[49:51]),  # gripper_right_qpos
    (np.s_[236:240], np.s_[53:57]),  # trunk_qpos
]
GROUPS = {"base": np.s_[0:3], "torso": np.s_[3:7], "l_arm": np.s_[7:14], "l_grip": np.s_[14:15],
          "r_arm": np.s_[15:22], "r_grip": np.s_[22:23]}  # fmt: skip


def frame(path, t):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, bgr = cap.read()
    assert ok, (path, t)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to61(s256):
    p = np.zeros(61, dtype=np.float32)
    for a, b in MAP_256_TO_61:
        p[b] = s256[a]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=EP)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 150, 300, 600, 900, 1200, 1500])
    ap.add_argument("--sources", nargs="+", default=["full", "224"])
    ap.add_argument("--stage", default="true", help="'true' (timestamp-based, as in training) or an int")
    ap.add_argument("--replay", type=int, default=0, help="Replay frames 0..N through the full server path instead.")
    args, rest = ap.parse_known_args()

    import rlc_server

    server_policy, openpi_policy = rlc_server.build_policy(rlc_server.make_parser().parse_args(rest))
    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper

    rlc = B1KPolicyWrapper(openpi_policy, task_id=0)
    df = pq.read_table(PARQUET.format(args.episode)).to_pandas()
    S = np.stack(df["observation.state"]).astype(np.float32)
    A = np.stack(df["action"]).astype(np.float64)
    T = len(df)
    print(f"episode {args.episode}: {T} frames")
    if args.replay:
        return replay(server_policy, S, A, args.episode, args.replay)
    for t in args.frames:
        stage = min(int(t / (T / 5)), 4) if args.stage == "true" else int(args.stage)
        demo = A[t : t + 26]
        for src in args.sources:
            obs = {"robot_r1::proprio": to61(S[t]), "task_id": np.array([0])}
            for k, cam in CAMS.items():
                root = DEMO_FULL if src == "full" else DEMO_224
                obs[k] = frame(f"{root}/observation.images.rgb.{cam}/episode_{args.episode:08d}.mp4", t)
            rlc.current_stage = stage
            batch = rlc.prepare_batch_for_pi_behavior(rlc.process_obs(obs))
            out = openpi_policy.infer(batch)
            pred = np.asarray(out["actions"], dtype=np.float64)[: len(demo)]
            err = {g: float(np.abs(pred[:, s] - demo[:, s]).mean()) for g, s in GROUPS.items()}
            print(
                f"t={t:4d} stage={stage} src={src:4s} pred_stage={out['predicted_stage']}  MAE "
                + " ".join(f"{g}={v:.3f}" for g, v in err.items())
            )
            print(f"   base vel demo mean {np.round(demo[:, 0:3].mean(0), 3).tolist()}  pred mean "
                  f"{np.round(pred[:, 0:3].mean(0), 3).tolist()}")  # fmt: skip
            print(f"   grip demo L/R {demo[[0, -1]][:, [14, 22]].round(2).tolist()} pred "
                  f"{pred[[0, -1]][:, [14, 22]].round(2).tolist()}; |arm move| demo "
                  f"{np.abs(demo[-1, 7:22] - demo[0, 7:22]).max():.3f} pred {np.abs(pred[-1, 7:22] - pred[0, 7:22]).max():.3f}",
                  flush=True)  # fmt: skip


def replay(policy, S, A, ep, n):
    """Open-loop replay: demo obs at every step in the client's batched format (full-res frames) -> BatchedRLCPolicy
    (inpainting, compression, stage voting, correction rules). Prints the executed actions next to the demo's."""
    caps = {k: cv2.VideoCapture(f"{DEMO_FULL}/observation.images.rgb.{c}/episode_{ep:08d}.mp4") for k, c in CAMS.items()}
    policy.reset()
    out = []
    for t in range(n):
        obs = {k: cv2.cvtColor(c.read()[1], cv2.COLOR_BGR2RGB)[None] for k, c in caps.items()}
        obs["robot_r1::proprio"] = to61(S[t])[None]
        obs["task_id"] = np.array([0])
        obs["steering_seed"] = np.array([0])
        out.append(np.asarray(policy.act(obs))[0])
        if (t + 1) % 20 == 0:
            a = np.stack(out[-20:])
            d = A[t - 19 : t + 1]
            print(f"steps {t - 19:4d}-{t:4d} stage {policy.rlc.current_stage}  base exec {a[:, 0:3].mean(0).round(3).tolist()} "
                  f"demo {d[:, 0:3].mean(0).round(3).tolist()}  r_arm j3 exec {a[-1, 17]:.2f} demo {d[-1, 17]:.2f}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, force=True)
    main()
