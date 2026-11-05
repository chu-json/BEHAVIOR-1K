"""Send fake observations to the Comet policy server to warm it up / measure its GPU memory, no simulator needed.

Mimics the fork evaluator's batched obs (num_envs=1) at a given render resolution.

    python steering/probe_server.py --resolution full --n 40
"""

import argparse
import time

import numpy as np

from omnigibson.eval.utils.network_utils import WebsocketClientPolicy

RES = {"full": (720, 480), "224": (224, 224)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resolution", choices=sorted(RES), default="full")
    ap.add_argument("--n", type=int, default=40, help="Number of act() calls (inference runs every 32).")
    args = ap.parse_args()
    head, wrist = RES[args.resolution]
    rng = np.random.default_rng(0)
    obs = {
        "robot_r1::proprio": np.zeros((1, 61), dtype=np.float32),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": rng.integers(0, 255, (1, head, head, 4), dtype=np.uint8),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": rng.integers(0, 255, (1, wrist, wrist, 4), np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": rng.integers(0, 255, (1, wrist, wrist, 4), np.uint8),
        "steering_seed": np.zeros((1, 1), dtype=np.int64),
    }
    client = WebsocketClientPolicy(host=args.host, port=args.port)
    client.reset()
    for i in range(args.n):
        t = time.perf_counter()
        a = client.act(obs)
        print(f"act {i}: shape {tuple(a.shape)} {1000 * (time.perf_counter() - t):.0f} ms", flush=True)
    client.reset()


if __name__ == "__main__":
    main()
