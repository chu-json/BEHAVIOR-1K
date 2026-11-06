"""Counterfactual prompt probing: how much does the prompt change a policy's predicted actions on fixed observations?

For every saved observation (from `comet_server.py --dump-obs`), query the policy once per (prompt, noise seed),
resetting the sampling RNG to the same seed for every prompt, so prompts are compared under identical diffusion
noise. Saves all predicted action chunks; steering/probe_report.py turns them into effect sizes:
  language effect = mean |chunk(prompt, seed) - chunk(P0, seed)|        (same noise, different prompt)
  noise floor     = mean |chunk(P0, seed_a) - chunk(P0, seed_b)|       (same prompt, different noise)
per joint group. Run with Comet's uv venv from the openpi-comet checkout:

    cd /home/behavior/ext/openpi-comet
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 .venv/bin/python \
        /home/behavior/BEHAVIOR-1K/steering/prompt_probe.py --model base --obs-dir <dumps> --out <file.npz>
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comet_server import install_omnigibson_shim  # noqa: E402
from probe_prompts import MODELS, PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--obs-dir", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-obs", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None, help="Name shown in reports (default: --model).")
    args = ap.parse_args()

    install_omnigibson_shim()
    import jax

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    m = MODELS[args.model]
    policy = _policy_config.create_trained_policy(_config.get_config(m["config"]), m["ckpt"])

    files = sorted(f for d in args.obs_dir for f in glob.glob(os.path.join(d, "*.npz")))
    if len(files) > args.max_obs:  # spread evenly over episodes/time
        files = [files[i] for i in np.linspace(0, len(files) - 1, args.max_obs).astype(int)]
    logging.info(f"{args.model}: {len(files)} observations x {len(args.prompts)} prompts x {len(args.seeds)} seeds")

    chunks = np.zeros((len(files), len(args.prompts), len(args.seeds), 32, 23), dtype=np.float32)
    proprio = np.zeros((len(files), 61), dtype=np.float32)
    t0 = time.time()
    for i, f in enumerate(files):
        d = np.load(f)
        imgs = d["images"]
        proprio[i] = d["proprio"]
        for j, pid in enumerate(args.prompts):
            batch = {
                "observation/egocentric_camera": imgs[0],
                "observation/wrist_image_left": imgs[1],
                "observation/wrist_image_right": imgs[2],
                "observation/state": d["proprio"],
                "prompt": PROMPTS[pid],
            }
            for k, seed in enumerate(args.seeds):
                policy._rng = jax.random.key(seed * 100003 + i)  # same noise for every prompt at this obs+seed
                chunks[i, j, k] = np.asarray(policy.infer(batch)["actions"])[:32, :23]
        if i % 20 == 0:
            el = time.time() - t0
            logging.info(
                f"{i + 1}/{len(files)} obs, {el / 60:.1f} min elapsed, ~{el / (i + 1) * (len(files) - i - 1) / 60:.0f} min left"
            )
    np.savez_compressed(
        args.out,
        chunks=chunks,
        proprio=proprio,
        files=np.array(files),
        prompts=np.array(args.prompts),
        prompt_text=np.array([PROMPTS[p] for p in args.prompts]),
        seeds=np.array(args.seeds),
        model=args.label or args.model,
    )
    json.dump(
        {"model": args.model, "n_obs": len(files), "minutes": (time.time() - t0) / 60}, open(args.out + ".json", "w")
    )
    logging.info(f"wrote {args.out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
