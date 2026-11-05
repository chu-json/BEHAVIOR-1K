"""Check that the 23-d state Comet extracts from this fork's 61-d proprio means the same thing as in the 2025 data.

1. Live layout: set the R1Pro's joints to a random pose, read proprio from the env, extract Comet's 23-d state with
   the fork's PROPRIOCEPTION_INDICES (what steering/comet_server.py uses) and compare to the joints set, by name.
2. Data consistency: for each of the 23 dims, compare the min/max seen in the 2025 radio demos with the current
   robot's limits for the joint now at that position. A reordered joint would typically show demo values
   outside the new joint's limits.

    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/check_state_layout.py \
        --demos /home/behavior/ext/data/2025-challenge-demos
"""

import argparse
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from steering.comet_server import PROPRIOCEPTION_INDICES  # noqa: E402
from steering.norm_stats_check import STATE_NAMES  # noqa: E402
from steering.norm_stats_check import extract_state_from_proprio as extract_2025  # noqa: E402

FORK_IDX = PROPRIOCEPTION_INDICES["R1Pro"]


def extract_fork(p):
    """Comet's extract_state_from_proprio, but with the fork's 61-d indices (as served by comet_server.py)."""
    return np.concatenate(
        [
            p[FORK_IDX["base_qvel"]],
            p[FORK_IDX["trunk_qpos"]],
            p[FORK_IDX["arm_left_qpos"]],
            p[FORK_IDX["arm_right_qpos"]],
            p[FORK_IDX["gripper_left_qpos"]].sum(keepdims=True),
            p[FORK_IDX["gripper_right_qpos"]].sum(keepdims=True),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    args = ap.parse_args()

    import torch as th
    from omegaconf import OmegaConf

    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.eval.utils.eval_utils import flatten_obs_dict
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False
    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.DefaultWrapper"},
            "policy_name": "local",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True,
            "partial_scene_load": True,
            "max_steps": 10_000,
            "write_video": False,
            "mode": "train",
            "seed": 0,
            "num_envs": 1,
            "task": {"name": "turning_on_radio"},
            "robot": None,
        }
    )
    lines = []
    with BatchedEvaluator(cfg) as ev:
        ev.load_batch({0: 242})
        robot = ev.instance_eval_states[0].env_accessor.robot
        names = list(robot.joints.keys())
        lo, hi = robot.joint_lower_limits, robot.joint_upper_limits

        # Joint indices in the order Comet's 23-d state expects them (base qvel has no joint-limit meaning).
        groups = [
            robot.trunk_control_idx,
            robot.arm_control_idx["left"],
            robot.arm_control_idx["right"],
        ]
        ordered = th.cat(groups).tolist()
        fingers = {arm: robot.gripper_control_idx[arm].tolist() for arm in robot.arm_names}

        # 1. Live layout check at a random pose (well inside the limits).
        g = th.Generator().manual_seed(0)
        q = robot.get_joint_positions().clone()
        for j in ordered:
            q[j] = lo[j] + (hi[j] - lo[j]) * (0.2 + 0.6 * th.rand(1, generator=g)).item()
        for arm in robot.arm_names:
            for j in fingers[arm]:
                q[j] = 0.02
        robot.set_joint_positions(q, drive=False)
        obs = flatten_obs_dict(ev.env.get_obs()[0][0])
        proprio = obs[f"{robot.name}::proprio"].detach().cpu().numpy()
        got = extract_fork(proprio)
        expected = np.concatenate([q[ordered].numpy(), [0.04, 0.04]])
        err = np.abs(got[3:] - expected)
        lines.append(f"proprio size: {proprio.shape[-1]}")
        lines.append(f"live layout check: max |extracted - set| = {err.max():.2e} over 20 joint dims")
        for k, (n, j) in enumerate(zip(STATE_NAMES[3:21], ordered)):
            lines.append(f"  {n:7s} <- joint '{names[j]}' set {q[j].item():+.3f} read {got[3 + k]:+.3f}")

        # 2. 2025 demo ranges vs. current joint limits, per dim.
        states = []
        for f in sorted(glob.glob(os.path.join(args.demos, "data", "task-0000", "*.parquet"))):
            t = pq.read_table(f, columns=["observation.state"])
            states.append(extract_2025(np.stack(t["observation.state"].to_numpy(zero_copy_only=False))))
        s = np.concatenate(states)
        lines.append(f"2025 radio demo frames: {len(s)}")
        lines.append("  dim     joint now at this position        limits              demo min..max     outside")
        for k, (n, j) in enumerate(zip(STATE_NAMES[3:21], ordered)):
            col = s[:, 3 + k]
            outside = np.mean((col < lo[j].item() - 1e-3) | (col > hi[j].item() + 1e-3))
            lines.append(
                f"  {n:7s} {names[j]:32s} [{lo[j].item():+.2f},{hi[j].item():+.2f}]  "
                f"[{col.min():+.2f},{col.max():+.2f}]  {100 * outside:5.1f}%"
            )
        print("STATE_LAYOUT_REPORT\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
