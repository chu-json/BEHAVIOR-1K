"""Where is the radio's toggle button, per instance? Writes button/radio world poses after loading each instance.

Used to test whether hard instances have the button facing away from where the robot ends up.

    OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/button_geometry.py \
        --out /home/behavior/ext/runs/button_geometry.json
"""

import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from steering.run_rollouts import RADIO_2025_VAL  # noqa: E402

RADIO = "radio_receiver.n.01_1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, nargs="+", default=RADIO_2025_VAL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from omegaconf import OmegaConf

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
    out = {}
    with BatchedEvaluator(cfg) as ev:
        acc = ev.instance_eval_states[0].env_accessor
        for inst in args.instances:
            ev.load_batch({0: inst})
            entity = acc.object_scope[RADIO]
            radio = getattr(entity, "wrapped_obj", entity)
            rp, rq = radio.get_position_orientation()
            bp = radio.states[ToggledOn].marker.get_position_orientation()[0]
            x, y, z, w = (float(v) for v in rq)
            out[inst] = {
                "radio_pos": [float(v) for v in rp],
                "radio_yaw": math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
                "button_pos": [float(v) for v in bp],
                "button_offset_world": [float(b - r) for b, r in zip(bp, rp)],
                "radio_aabb_extent": [float(v) for v in radio.aabb_extent],
            }
            print("BUTTON", inst, json.dumps(out[inst]), flush=True)
            # og.shutdown() in the evaluator's __exit__ ends the process, so write after every instance.
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
