"""CPU smoke test for steering/rlc_server.py on real observations saved by comet_server.py --dump-obs.

Runs a short episode (several inference calls, so both the plain and the inpainting sampling paths run) and checks
every action is (1, 23), finite, gripper in [-1, 1], and torso/arm targets near the current joints.

    cd /home/behavior/ext/rlc/solution
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/rlc_smoke_test.py \
        --mode direct
    # websocket round trip against a server already running: rlc_server.py --port 8010
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/rlc_smoke_test.py \
        --mode ws --port 8010
"""

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DUMPS = "/home/behavior/ext/runs/day/obs_record/A_radio_sft_P0_s2/obs"
CAMS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)
# 23-d action layout -> fork 61-d proprio
JOINTS = {"torso": (slice(3, 7), slice(53, 57)), "left_arm": (slice(7, 14), slice(3, 10)),
          "right_arm": (slice(15, 22), slice(28, 35))}  # fmt: skip


def make_obs(npz, seed):
    d = np.load(npz)
    obs = {k: d["images"][i][None] for i, k in enumerate(CAMS)}  # 224x224 (the client may also send 720/480)
    obs["robot_r1::proprio"] = d["proprio"][None].astype(np.float32)
    obs["task_id"] = np.array([0], dtype=np.int64)
    obs["steering_seed"] = np.array([seed], dtype=np.int64)
    obs["steering_target"] = np.array([[0.6, 0.0, 0.8]], dtype=np.float32)
    return obs, d["proprio"].astype(np.float64)


def check(a, q, step):
    a = np.asarray(a, dtype=np.float64)
    assert a.shape == (1, 23), a.shape
    assert np.isfinite(a).all()
    a = a[0]
    # gripper: [-1, 1] nominal; model/cubic-interpolation overshoot is clipped by the controller's input limits
    assert abs(a[14]) <= 1.5 and abs(a[22]) <= 1.5, (a[14], a[22])
    assert np.abs(a[0:3]).max() < 1.5, a[0:3]
    dev = {k: float(np.abs(a[asl] - q[qsl]).max()) for k, (asl, qsl) in JOINTS.items()}
    assert max(dev.values()) < 0.5, (dev, np.round(a, 3).tolist(), np.round(q[53:57], 3).tolist())
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["direct", "ws"], default="direct")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--steps", type=int, default=45, help="45 = 3 inferences (steps 0, 20, 40)")
    ap.add_argument("--dumps", default=DUMPS)
    ap.add_argument("--episode", default="ep000")
    args, rest = ap.parse_known_args()

    files = sorted(glob.glob(os.path.join(args.dumps, f"{args.episode}_step*.npz")))
    assert files, f"no dumps in {args.dumps}"
    if args.mode == "direct":
        import rlc_server

        policy, _ = rlc_server.build_policy(rlc_server.make_parser().parse_args(rest))
    else:
        from comet_server import install_omnigibson_shim

        net = install_omnigibson_shim()
        policy = net.WebsocketClientPolicy(host="localhost", port=args.port)

    policy.reset()
    q_first = np.load(files[0])["proprio"]
    for step in range(args.steps):
        # dumps are every 32 sim steps of a Comet rollout; walk through them so the scene changes a bit
        obs, q = make_obs(files[min(step // 20, len(files) - 1)], seed=7)
        t = time.time()
        a = policy.act(obs)
        a = a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
        dt = time.time() - t
        dev = check(a, q, step)
        if step % 10 == 0 or dt > 1.0:
            print(f"step {step:3d} {dt:7.2f}s  max|a-q| " + " ".join(f"{k}={v:.3f}" for k, v in dev.items()))
            print("   action", np.round(a[0], 3).tolist())
    print("proprio[0] joints: torso", np.round(q_first[53:57], 3).tolist())
    print(f"SMOKE TEST OK ({args.mode}, {args.steps} steps)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
