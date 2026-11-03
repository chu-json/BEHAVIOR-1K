"""
Check that our simulated Franka's joint angles mean the same as the real DROID Franka's.

Takes recorded DROID frames (joint angles + the end-effector pose DROID logged for them), puts our robot in the same
joint angles, and compares where our gripper ends up (in the robot-base frame). If joint order, signs or zero offsets
differed, the positions would be off by tens of centimetres or on the wrong side. A constant offset ALONG the gripper
axis only means the two logged points sit at different places on the hand.

Result: sideways error 0.0-0.4 cm in 26 of 32 poses (the rest put our arm into our table), with DROID's point a constant
10.5 cm BEHIND our Panda fingertip point (13.5 cm behind our Robotiq one). The Franka flange is 10.34 cm behind the
Panda fingertip point, so DROID's cartesian_position is most likely the flange pose: the joint conventions match.

    OMNIGIBSON_HEADLESS=1 python my_franka/pi0_joint_convention_check.py
"""
import glob
import os

import numpy as np
import pandas as pd
import torch as th

import sys

import omnigibson as og
import omnigibson.utils.transform_utils as T
from pi0_common import PI0_DIR
from pi0_zero_shot_eval import build_pi0_cfg


def main():
    files = sorted(glob.glob(os.path.join(PI0_DIR, "droid_real", "data", "chunk-000", "episode_*.parquet")))[:8]
    samples = []
    for f in files:
        df = pd.read_parquet(f)
        for t in np.linspace(0, len(df) - 1, 4).astype(int):
            samples.append((os.path.basename(f), t, np.array(df["observation.state.joint_position"].iloc[t]),
                            np.array(df["observation.state.cartesian_position"].iloc[t])))
    gripper = sys.argv[1] if len(sys.argv) > 1 else "robotiq"
    print(f"gripper: {gripper}", flush=True)
    env = og.Environment(configs=build_pi0_cfg(gripper))
    robot = env.scene.robots[0]
    arm_idx = robot.arm_control_idx[robot.default_arm]
    errs, axis_offsets = [], []
    for name, t, q, cart in samples:
        q_all = robot.get_joint_positions()
        q_all[arm_idx] = th.tensor(q, dtype=q_all.dtype)
        robot.set_joint_positions(q_all)
        robot.keep_still()
        env.step(th.cat([th.tensor(q, dtype=th.float32), th.tensor([1.0])]))  # one control step holding these angles
        pos, quat = robot.get_relative_eef_pose()
        ours = pos.numpy()
        droid = cart[:3]
        # gripper pointing direction in the base frame (eef z-axis), to separate "along the gripper" from sideways error
        z_axis = T.quat2mat(quat)[:, 2].numpy()
        diff = droid - ours
        along = float(diff @ z_axis)
        sideways = float(np.linalg.norm(diff - along * z_axis))
        errs.append(sideways)
        axis_offsets.append(along)
        print(f"{name} t={t:3d}  DROID eef {np.round(droid, 3)}  ours {np.round(ours, 3)}  "
              f"| offset along gripper axis {along * 100:+5.1f} cm, sideways {sideways * 100:4.1f} cm", flush=True)
    print(f"\nover {len(samples)} poses: sideways error mean {np.mean(errs) * 100:.1f} cm (max {np.max(errs) * 100:.1f}); "
          f"offset along the gripper axis mean {np.mean(axis_offsets) * 100:+.1f} cm (std {np.std(axis_offsets) * 100:.1f})")
    og.shutdown()


if __name__ == "__main__":
    main()
