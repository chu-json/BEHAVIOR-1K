"""Generate scripted radio demos (ApproachAssist-free PressAssist v4 in scripted mode: drive toward the button, torso+arm
IK press) in perturbed scenes, recorded in the steering/ft radio_ft format (steering/demo_recorder.py). Only
successful episodes are kept.

    python3 steering/gen_scripted_demos.py [--per-config 30] [--configs yaw0,yaw45,...] [--steps 900]

Instances: training-side task instances (never the 2025 val/test instances used for evaluation). Each config writes
to /home/behavior/ext/data/scripted_demos/<config>/ and runs/scripted_demos/<config>/rollouts.csv.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from steer_campaign import EXT, REPO, run  # noqa: E402

EVAL = {242, 295, 211, 203, 109, 181, 197, 187, 214, 139}  # 2025 val10 (our 20 eval pairs)
TRAIN_INSTANCES = [i for i in range(1, 81) if i not in EVAL]
CONFIGS = {
    "yaw0": [],
    "yaw45": ["--radio-yaw-deg", "45"],
    "yaw-45": ["--radio-yaw-deg", "-45"],
    "yaw90": ["--radio-yaw-deg", "90"],
    "yaw-90": ["--radio-yaw-deg", "-90"],
    "clutter6": ["--clutter", "6"],
    "occluder": ["--occluder"],
}
RELEASED = f"{EXT}/mentor/released_pi05_radio/pi05_turn_on_the_radio"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--per-config", type=int, default=30)
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()
    # A policy server must be up for the harness; the scripted assist drives, the policy only acts after a failed press.
    server = {
        "cmd": [".venv/bin/python", f"{REPO}/steering/mentor_server.py", "--ckpt", RELEASED, "--port", "8000"],
        "cwd": f"{EXT}/mentor/openpi",
    }
    for k, cfg in enumerate(args.configs.split(",")):
        pairs = [(TRAIN_INSTANCES[(j + 7 * k) % len(TRAIN_INSTANCES)], 1000 + j) for j in range(args.per_config)]
        out = f"{EXT}/data/scripted_demos/{cfg}"
        rflags = ["--press-assist", "--record-demos", out, "--record-only-success", *CONFIGS[cfg]]
        if cfg != "yaw0":
            rflags += ["--perturb-seed", str(k)]
        run(cfg, [], rflags, pairs, args.steps, f"{EXT}/runs/scripted_demos", {"STEER_PRESS_SCRIPTED": "1"}, server)


if __name__ == "__main__":
    main()
