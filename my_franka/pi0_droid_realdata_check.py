"""
Sanity check for the zero-shot pi0.5-DROID pipeline WITHOUT our simulator: feed it real recorded DROID frames and
compare its predicted actions with the actions the robot actually executed.

If our wrapper (pi0_common.py: image/state/prompt conventions, normalization, tokenizer, LeRobot port of the weights)
is right, predictions on real DROID data should track the recorded joint velocities well above chance. If they don't,
the 0/96 in simulation could be our bug rather than the sim-to-real gap.

Also runs the same frames as SQUARE center crops (what our simulator effectively gave the model) to measure whether
the missing 16:9 letterbox bars matter.

    python my_franka/pi0_droid_realdata_check.py --episodes 8
"""
import argparse
import json
import os

import av
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from pi0_common import PI0_DIR, PI05DroidZeroShot

REPO = "cadene/droid_1.0.1"
CACHE = os.path.join(PI0_DIR, "droid_real")


def fetch(path):
    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=CACHE)


def frames(path):
    with av.open(path) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def square_crop(img):
    """Center square crop of a 16:9 frame: same content area our square sim cameras give, no letterbox bars."""
    h, w = img.shape[:2]
    x0 = (w - h) // 2
    return img[:, x0:x0 + h]


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--scan", type=int, default=60, help="how many episode files to look through for good episodes")
    parser.add_argument("--every", type=int, default=10, help="predict at every Nth frame")
    args = parser.parse_args()

    picked = []
    for ep in range(args.scan):
        try:
            df = pd.read_parquet(fetch(f"data/chunk-000/episode_{ep:06d}.parquet"))
        except Exception as e:  # missing episode index
            continue
        instr = str(df["language_instruction"].iloc[0]).strip()
        if bool(df["is_episode_successful"].iloc[0]) and instr:
            picked.append((ep, instr, df))
        if len(picked) >= args.episodes:
            break
    print(f"picked {len(picked)} successful episodes with instructions: {[p[0] for p in picked]}", flush=True)

    model = PI05DroidZeroShot()
    rows = []
    for ep, instr, df in picked:
        ext = frames(fetch(f"videos/chunk-000/observation.images.exterior_1_left/episode_{ep:06d}.mp4"))
        wr = frames(fetch(f"videos/chunk-000/observation.images.wrist_left/episode_{ep:06d}.mp4"))
        n = min(len(df), len(ext), len(wr))
        jp = np.stack(df["observation.state.joint_position"].values)
        gp = np.stack(df["observation.state.gripper_position"].values).reshape(-1)
        jv = np.stack(df["action.joint_velocity"].values)
        ga = np.stack(df["action.gripper_position"].values).reshape(-1)
        for t in range(0, n - 1, args.every):
            state8 = np.concatenate([jp[t], [gp[t]]])
            pred_lb = model.predict(ext[t], wr[t], state8, instr)[0]                              # 16:9 -> letterboxed
            pred_sq = model.predict(square_crop(ext[t]), square_crop(wr[t]), state8, instr)[0]    # square, no bars
            rows.append(dict(ep=ep, t=t, true_v=jv[t].tolist(), true_g=float(ga[t]),
                             pred_lb_v=pred_lb[:7].tolist(), pred_lb_g=float(pred_lb[7]),
                             pred_sq_v=pred_sq[:7].tolist(), pred_sq_g=float(pred_sq[7])))
        print(f"episode {ep}: {n} frames, '{instr[:60]}'", flush=True)

    tv = np.array([r["true_v"] for r in rows])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    print(f"\n{len(rows)} predictions on real DROID frames")
    for tag in ("lb", "sq"):
        pv = np.array([r[f"pred_{tag}_v"] for r in rows])
        pg = np.array([r[f"pred_{tag}_g"] for r in rows])
        tg = np.array([r["true_g"] for r in rows])
        per_joint = [corr(pv[:, j], tv[:, j]) for j in range(7)]
        shuffled = [corr(pv[perm, j], tv[:, j]) for j in range(7)]
        cos = np.sum(pv * tv, 1) / (np.linalg.norm(pv, axis=1) * np.linalg.norm(tv, axis=1) + 1e-9)
        moving = np.linalg.norm(tv, axis=1) > 0.05
        name = "letterboxed 16:9 (as DROID)" if tag == "lb" else "square crop (like our sim)"
        print(f"\n[{name}]")
        print("  correlation predicted vs recorded joint velocity, per joint:", [round(c, 2) for c in per_joint],
              f"| mean {np.mean(per_joint):.2f}")
        print("  same, predictions SHUFFLED across frames (chance level):  ", [round(c, 2) for c in shuffled],
              f"| mean {np.mean(shuffled):.2f}")
        print(f"  mean cosine similarity of the 7-D velocity direction (frames where the robot moves, n={moving.sum()}): "
              f"{cos[moving].mean():.2f}  (shuffled: {(np.sum(pv[perm] * tv, 1) / (np.linalg.norm(pv[perm], axis=1) * np.linalg.norm(tv, axis=1) + 1e-9))[moving].mean():.2f})")
        print(f"  gripper open/closed agreement: {np.mean((pg > 0.5) == (tg > 0.5)) * 100:.0f}%  "
              f"(always predicting the majority class: {max(np.mean(tg > 0.5), 1 - np.mean(tg > 0.5)) * 100:.0f}%)")
    with open(os.path.join(PI0_DIR, "realdata_check.json"), "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()
