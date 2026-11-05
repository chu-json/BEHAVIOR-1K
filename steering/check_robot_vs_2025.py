"""Compare the R1Pro as loaded by this fork's evaluator against what the 2025 data says Comet was trained on.

Reports: camera poses relative to the base at the reset pose (vs. 2025 demo frame 0), camera intrinsics (vs. the
2025 CAMERA_INTRINSICS), base mass before/after the evaluator override, and finger-link collision geometry
(which the fork's toggle check needs as meshes).

    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/check_robot_vs_2025.py \
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

# 2025 omnigibson.learning.utils.eval_utils.CAMERA_INTRINSICS (fx at the data-collection resolution).
FX_2025 = {"head": (306.0, 720), "left_wrist": (388.6639, 480), "right_wrist": (388.6639, 480)}
CAM_ORDER = ["left_wrist", "right_wrist", "head"]  # order of the 21-d cam_rel_poses in both versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True)
    args = ap.parse_args()

    from omegaconf import OmegaConf

    import omnigibson.eval.evaluator as evaluator_mod
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False

    masses = {}
    orig_apply = evaluator_mod.BatchedEvaluator._apply_robot_eval_settings

    def spy_apply(self):
        r = self.instance_eval_states[0].env_accessor.robot
        masses["default"] = r.base_footprint_link.mass
        orig_apply(self)
        masses["eval"] = r.base_footprint_link.mass

    evaluator_mod.BatchedEvaluator._apply_robot_eval_settings = spy_apply

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": "steering.compat_wrapper.CompatFullResWrapper"},
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
    out = []
    with evaluator_mod.BatchedEvaluator(cfg) as ev:
        ev.load_batch({0: 242})
        st = ev.instance_eval_states[0]
        robot = st.env_accessor.robot
        out.append(f"base_footprint_link mass: default {masses['default']:.1f} kg -> eval {masses['eval']:.1f} kg")
        total = sum(link.mass for link in robot.links.values())
        out.append(f"robot total link mass (after override): {total:.1f} kg")

        # Camera poses relative to base at the reset pose.
        live = st.obs[f"{robot.name}::cam_rel_poses"].detach().cpu().numpy().reshape(-1)
        demo = []
        for f in sorted(glob.glob(os.path.join(args.demos, "data", "task-0000", "*.parquet")))[:50]:
            t = pq.read_table(f, columns=["observation.cam_rel_poses"])
            demo.append(np.stack(t["observation.cam_rel_poses"].to_numpy(zero_copy_only=False))[0])
        demo = np.mean(demo, axis=0)
        for k, cam in enumerate(CAM_ORDER):
            lp, lq, dp, dq = (
                live[7 * k : 7 * k + 3],
                live[7 * k + 3 : 7 * k + 7],
                demo[7 * k : 7 * k + 3],
                demo[7 * k + 3 : 7 * k + 7],
            )
            ang = 2 * np.degrees(
                np.arccos(min(1.0, abs(float(np.dot(lq / np.linalg.norm(lq), dq / np.linalg.norm(dq))))))
            )
            out.append(
                f"{cam:11s} pos live {np.round(lp, 4)} vs 2025 {np.round(dp, 4)}  |dpos| {100 * np.linalg.norm(lp - dp):.2f} cm"
                f"  rot diff {ang:.2f} deg"
            )

        # Intrinsics.
        for cam in CAM_ORDER:
            sensor = robot.sensors[ev.robot_camera_names[cam].split("::")[1]]
            w = sensor.image_width
            fx = w * sensor.focal_length / sensor.horizontal_aperture
            fx25, w25 = FX_2025[cam]
            hfov = 2 * np.degrees(np.arctan(w / (2 * fx)))
            hfov25 = 2 * np.degrees(np.arctan(w25 / (2 * fx25)))
            out.append(
                f"{cam:11s} res {sensor.image_height}x{w}  focal {sensor.focal_length:.3f} aperture "
                f"{sensor.horizontal_aperture:.3f}  fx {fx:.1f} (2025 {fx25 * w / w25:.1f} at this res)  "
                f"hfov {hfov:.1f} deg (2025 {hfov25:.1f})"
            )

        # Finger links and their collision geometry.
        for arm, links in robot.finger_links.items():
            for link in links:
                kinds = [type(c).__name__ + ":" + c.prim.GetTypeName() for c in link.collision_meshes.values()]
                out.append(f"finger {arm:5s} {link.name:28s} collision: {kinds}")
        print("ROBOT_VS_2025\n" + "\n".join(out), flush=True)


if __name__ == "__main__":
    main()
