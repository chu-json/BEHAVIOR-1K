"""Closed-loop radio eval of a fine-tuned checkpoint (steering/ft) with the campaign harness.

    python3 steering/ft/ft_closed_loop.py CKPT NAME [N_PAIRS] [STEPS] [--mentor]
--mentor serves CKPT with steering/mentor_server.py (wensi-ai/openpi checkpoints, e.g. the released baseline).
Runs the first N_PAIRS of steer_campaign.PAIRS (val10 x seeds) and writes runs/ft/NAME/rollouts.csv.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from steer_campaign import EXT, PAIRS, REPO, run  # noqa: E402

mentor = "--mentor" in sys.argv
sys.argv = [a for a in sys.argv if a != "--mentor"]
ckpt, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
server = {
    "cmd": [".venv/bin/python", f"{REPO}/steering/comet_server.py", "--ckpt", ckpt,
            "--config", "steering_ft:radio_lora", "--port", "8000"],  # fmt: skip
    "cwd": f"{EXT}/openpi-comet",
}
if mentor:
    server = {
        "cmd": [".venv/bin/python", f"{REPO}/steering/mentor_server.py", "--ckpt", ckpt, "--port", "8000"],
        "cwd": f"{EXT}/mentor/openpi",
    }
run(name, [], [], PAIRS[:n], steps, f"{EXT}/runs/ft", None, server)
