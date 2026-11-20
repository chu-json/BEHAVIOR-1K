"""Which frame is proprio base_qvel in, in a demo dataset? (2025 demos: world; 2026 demos: robot. NOTES "CyberWAM setup")

The base action [0:3] is a robot-frame velocity command (HolonomicBaseJointController, output limit 0.75 m/s), so
in a robot-frame dataset base_qvel_xy ~= k * action_xy with k ~= 0.75; in a world-frame dataset it matches only after
rotating the action by the heading (integrated base_qvel yaw rate + a fitted initial yaw).

    python steering/check_qvel_frame.py --parquet <2026 data/chunk-000/file-000.parquet>          # 61-d state
    python steering/check_qvel_frame.py --parquet <2025 data/task-0000/episode_00000010.parquet>  # 256-d state
"""

import argparse

import numpy as np
import pyarrow.parquet as pq


def rel_err(s, pred):
    k = (pred * s).sum() / max((pred * pred).sum(), 1e-9)
    return np.linalg.norm(s - k * pred) / np.linalg.norm(s), k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    for path in args.parquet:
        t = pq.read_table(path)
        S = np.stack(t["observation.state"].to_numpy(zero_copy_only=False))
        A = np.stack(t["action"].to_numpy(zero_copy_only=False))
        ep = np.array(t["episode_index"]) if "episode_index" in t.column_names else np.zeros(len(S), int)
        qv = {61: slice(0, 3), 256: slice(253, 256)}[S.shape[1]]
        for e in np.unique(ep)[: args.episodes]:
            m = ep == e
            s, a = S[m][:, qv], A[m][:, 0:3]
            yaw = np.cumsum(s[:, 2]) / args.fps
            er, kr = rel_err(s[:, :2], a[:, :2])
            best = min(
                rel_err(
                    s[:, :2],
                    np.stack(
                        [
                            np.cos(yaw + y0) * a[:, 0] - np.sin(yaw + y0) * a[:, 1],
                            np.sin(yaw + y0) * a[:, 0] + np.cos(yaw + y0) * a[:, 1],
                        ],
                        1,
                    ),
                )
                for y0 in np.linspace(-np.pi, np.pi, 73)
            )
            verdict = "ROBOT" if er < best[0] else "WORLD"
            print(
                f"{path.split('/')[-1]} ep {e}: robot-frame err {er:.3f} (k={kr:.2f}) | world-frame err "
                f"{best[0]:.3f} (k={best[1]:.2f}) -> {verdict} frame"
            )


if __name__ == "__main__":
    main()
