"""Recompute Comet's norm stats from the turning_on_radio demos and compare to the published file.

Comet's scripts/compute_norm_stats.py (B1K branch) needs the v3.7.2 `omnigibson.learning` + lerobot stack,
which this fork doesn't have. This script reproduces that branch's math exactly:
  1. take per-episode stats from meta/episodes_stats.jsonl for the radio episodes (task index 0),
  2. aggregate with omnigibson v3.7.2 `lerobot_utils.aggregate_feature_stats` (copied below),
  3. apply openpi-comet `b1k_policy.extract_state_from_proprio` to each stat vector, pad to 32.
It also computes exact per-frame stats from the raw parquet as an independent ground truth.

Usage (behavior env):
    python steering/norm_stats_check.py --demos /home/behavior/ext/data/2025-challenge-demos \
        --published /home/behavior/ext/normstats/pt12/norm_stats.json
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

# v3.7.2 PROPRIOCEPTION_INDICES["R1Pro"] for the 256-d dataset state (only the fields Comet uses).
IDX = {
    "base_qvel": np.s_[253:256],
    "trunk_qpos": np.s_[236:240],
    "arm_left_qpos": np.s_[158:165],
    "arm_right_qpos": np.s_[197:204],
    "gripper_left_qpos": np.s_[193:195],
    "gripper_right_qpos": np.s_[232:234],
}
STATE_NAMES = (
    [f"base_qvel{i}" for i in range(3)]
    + [f"trunk{i}" for i in range(4)]
    + [f"arm_L{i}" for i in range(7)]
    + [f"arm_R{i}" for i in range(7)]
    + ["grip_L", "grip_R"]
)
ACTION_NAMES = (
    [f"base{i}" for i in range(3)]
    + [f"torso{i}" for i in range(4)]
    + [f"arm_L{i}" for i in range(7)]
    + ["grip_L"]
    + [f"arm_R{i}" for i in range(7)]
    + ["grip_R"]
)
RADIO_TASK_INDEX = 0
MODEL_ACTION_DIM = 32


def extract_state_from_proprio(p):
    """Same as openpi-comet b1k_policy.extract_state_from_proprio (gripper width = sum of 2 fingers)."""
    return np.concatenate(
        [
            p[..., IDX["base_qvel"]],
            p[..., IDX["trunk_qpos"]],
            p[..., IDX["arm_left_qpos"]],
            p[..., IDX["arm_right_qpos"]],
            p[..., IDX["gripper_left_qpos"]].sum(axis=-1, keepdims=True),
            p[..., IDX["gripper_right_qpos"]].sum(axis=-1, keepdims=True),
        ],
        axis=-1,
    )


def aggregate_feature_stats(stats_ft_list):
    """Copied from omnigibson v3.7.2 learning/utils/lerobot_utils.py."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    q01 = np.stack([s["q01"] for s in stats_ft_list])
    q99 = np.stack([s["q99"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means**2) * counts).sum(axis=0) / total_count
    return {
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "q01": np.percentile(q01, 1, axis=0),
        "q99": np.percentile(q99, 99, axis=0),
    }


def script_stats(demos):
    """Reproduces compute_norm_stats.py's B1K branch for tasks=["turning_on_radio"]."""
    state, action = [], []
    with open(os.path.join(demos, "meta", "episodes_stats.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d["episode_index"] // 10000 != RADIO_TASK_INDEX:
                continue
            s = d["stats"]
            state.append({k: np.asarray(v, dtype=np.float64) for k, v in s["observation.state"].items()})
            action.append({k: np.asarray(v, dtype=np.float64) for k, v in s["action"].items()})
    st, ac = aggregate_feature_stats(state), aggregate_feature_stats(action)
    out = {"state": {}, "actions": {}}
    for k in ("mean", "std", "q01", "q99"):
        out["state"][k] = extract_state_from_proprio(st[k])
        out["actions"][k] = ac[k]
    return out, len(state)


def exact_stats(demos):
    """Exact per-frame stats over all radio demo frames."""
    states, actions = [], []
    for f in sorted(glob.glob(os.path.join(demos, "data", "task-0000", "*.parquet"))):
        t = pq.read_table(f, columns=["observation.state", "action"])
        states.append(extract_state_from_proprio(np.stack(t["observation.state"].to_numpy(zero_copy_only=False))))
        actions.append(np.stack(t["action"].to_numpy(zero_copy_only=False)))
    out = {}
    for name, x in (("state", np.concatenate(states)), ("actions", np.concatenate(actions))):
        out[name] = {
            "mean": x.mean(0),
            "std": x.std(0),
            "q01": np.quantile(x, 0.01, axis=0),
            "q99": np.quantile(x, 0.99, axis=0),
        }
    return out, np.concatenate(states), np.concatenate(actions)


def qnorm(x, s):
    return (x - s["q01"]) / (s["q99"] - s["q01"] + 1e-6) * 2.0 - 1.0


def compare(label, ref, other, frames, names):
    """Print per-dim q01/q99 and how much the normalized model input shifts if `other` replaced `ref`."""
    shift = np.abs(qnorm(frames, ref) - qnorm(frames, other)).mean(0)
    print(f"\n{label}: mean |normalized(x; published) - normalized(x; recomputed)| per dim, over radio demo frames")
    print(f"  {'dim':10s} {'pub q01':>9s} {'new q01':>9s} {'pub q99':>9s} {'new q99':>9s} {'shift':>7s}")
    for i, n in enumerate(names):
        print(
            f"  {n:10s} {ref['q01'][i]:9.4f} {other['q01'][i]:9.4f} {ref['q99'][i]:9.4f} {other['q99'][i]:9.4f}"
            f" {shift[i]:7.3f}"
        )
    print(f"  => mean shift {shift.mean():.3f}, max shift {shift.max():.3f} ({names[int(shift.argmax())]})")
    return shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--out", default=None, help="Optionally write recomputed (script-method) stats here.")
    args = ap.parse_args()

    pub_raw = json.load(open(args.published))["norm_stats"]
    pub = {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in pub_raw.items()}
    scr, n_eps = script_stats(args.demos)
    ex, st_frames, ac_frames = exact_stats(args.demos)
    print(f"radio episodes: {n_eps}, frames: {len(st_frames)}")

    for name, frames, names in (("state", st_frames, STATE_NAMES), ("actions", ac_frames, ACTION_NAMES)):
        d = len(names)
        ref = {k: pub[name][k][:d] for k in ("q01", "q99")}
        compare(f"[{name}] published vs script-method recompute", ref, scr[name], frames, names)
        compare(f"[{name}] published vs exact per-frame", ref, ex[name], frames, names)
        print(f"  padding dims {d}:{MODEL_ACTION_DIM} in published: q01 {pub[name]['q01'][d:].tolist()[:3]}...")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pad = lambda v: np.pad(v, (0, MODEL_ACTION_DIM - len(v))).tolist()  # noqa: E731
        json.dump(
            {"norm_stats": {k: {kk: pad(vv) for kk, vv in v.items()} for k, v in scr.items()}},
            open(args.out, "w"),
            indent=2,
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
