"""Norm stats for a delta-joint wiring of pi0.5: joint actions as offsets from the current joint position.

Base pi0.5 was pre-trained with joint actions expressed relative to the current state (openpi's ALOHA / DROID configs
convert absolute joint targets to deltas before training). This computes, from the 2025 radio demos, stats of
  delta[t, k] = action[t + k, joint] - state[t, joint]     for k = 0..31 (the 32-step action chunk)
for torso (action 3:7) and both arms (7:14, 15:22), and writes a norm_stats.json identical to the absolute one except
for those action dims. Base velocity and grippers stay absolute (as in openpi's conventions).

    python steering/delta_norm_stats.py --demos /home/behavior/ext/data/2025-challenge-demos \
        --absolute /home/behavior/ext/normstats/pt12/norm_stats.json --out <dir>/norm_stats.json
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

# action dims -> 256-d demo state dims (v3.7.2 PROPRIOCEPTION_INDICES)
JOINTS = [(slice(3, 7), slice(236, 240)), (slice(7, 14), slice(158, 165)), (slice(15, 22), slice(197, 204))]
H = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--absolute", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10, help="Use every k-th frame as a chunk start.")
    args = ap.parse_args()
    deltas = []
    for f in sorted(glob.glob(os.path.join(args.demos, "data", "task-0000", "*.parquet"))):
        t = pq.read_table(f, columns=["observation.state", "action"])
        s = np.stack(t["observation.state"].to_numpy(zero_copy_only=False))
        a = np.stack(t["action"].to_numpy(zero_copy_only=False))
        for i in range(0, len(a) - H, args.stride):
            d = np.zeros((H, 23), dtype=np.float32)
            for asl, ssl in JOINTS:
                d[:, asl] = a[i : i + H, asl] - s[i, ssl]
            deltas.append(d)
    d = np.concatenate(deltas)  # (chunks*H, 23)
    ns = json.load(open(args.absolute))
    act = ns["norm_stats"]["actions"]
    for asl, _ in JOINTS:
        for j in range(asl.start, asl.stop):
            col = d[:, j]
            act["mean"][j] = float(col.mean())
            act["std"][j] = float(col.std())
            act["q01"][j] = float(np.quantile(col, 0.01))
            act["q99"][j] = float(np.quantile(col, 0.99))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(ns, open(args.out, "w"), indent=2)
    print(f"{len(deltas)} chunks; delta q01/q99 torso {np.round(act['q01'][3:7], 3)} {np.round(act['q99'][3:7], 3)}; "
          f"left arm {np.round(act['q01'][7:14], 2)} {np.round(act['q99'][7:14], 2)}")


if __name__ == "__main__":
    main()
