"""Is the radio's toggle-button marker reachable from outside the radio's collision geometry?

Casts PhysX rays from 40 cm outside the radio toward the marker center along the radio's local axes (+-x, +-y,
+z). The first hit on the radio is its surface; depth = distance from that surface point to the marker center.
If depth > marker radius, a finger touching that face cannot overlap the marker without penetrating the radio.

    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/button_reachability.py --instance 214
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

RADIO = "radio_receiver.n.01_1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, default=214)
    ap.add_argument("--out", default="/home/behavior/ext/runs/button_reachability.json")
    args = ap.parse_args()

    import torch as th
    from omegaconf import OmegaConf

    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    from omnigibson.eval.evaluator import BatchedEvaluator
    from omnigibson.macros import gm
    from omnigibson.object_states import ToggledOn

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
    with BatchedEvaluator(cfg) as ev:
        ev.load_batch({0: args.instance})
        acc = ev.instance_eval_states[0].env_accessor
        entity = acc.object_scope[RADIO]
        radio = getattr(entity, "wrapped_obj", entity)
        marker = radio.states[ToggledOn].marker
        mpos = marker.get_position_orientation()[0]
        radius = float(th.min(marker.extent * marker.scale))
        rp, rq = radio.get_position_orientation()
        rot = T.quat2mat(rq)
        radio_paths = {link.prim_path for link in radio.links.values()}
        res = {"instance": args.instance, "marker_radius_m": radius, "faces": {}}
        for name, axis in {
            "+x": [1, 0, 0],
            "-x": [-1, 0, 0],
            "+y": [0, 1, 0],
            "-y": [0, -1, 0],
            "+z": [0, 0, 1],
        }.items():
            d = rot @ th.tensor(axis, dtype=th.float32)
            origin = mpos + 0.40 * d
            hits = []

            def report(hit):
                hits.append((hit.distance, hit.rigid_body, list(hit.position)))
                return True

            og.sim.psqi.raycast_all(origin.tolist(), (-d).tolist(), 0.40, report)
            on_radio = sorted(h for h in hits if h[1] in radio_paths)
            if on_radio:
                surface_dist_from_origin = on_radio[0][0]
                depth = 0.40 - surface_dist_from_origin  # >0: marker center is behind the surface
                res["faces"][name] = {
                    "marker_center_depth_behind_surface_cm": round(100 * depth, 2),
                    "sphere_protrudes_cm": round(100 * (radius - depth), 2),
                    "reachable_by_touching_face": bool(depth < radius),
                }
            else:
                res["faces"][name] = {"hit": "no radio surface between origin and marker (marker outside?)"}
        print("REACH", json.dumps(res), flush=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
