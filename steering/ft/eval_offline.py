"""Offline check of a fine-tuned checkpoint on held-out radio demos (ft_config.HOLDOUT): predicted vs recorded
action chunks, through the exact serving transform stack (openpi create_trained_policy + our config).

    cd /home/behavior/ext/openpi-comet
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/ft/eval_offline.py \
        --ckpt /home/behavior/ext/ckpt/ft/<exp>/<step> --config radio_lora
Reports mean |pred - demo| per joint group over the 32-step chunk, next to a "hold the current joints" baseline.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import ft_loader  # noqa: E402,F401  (omnigibson shim)
import ft_config  # noqa: E402
import numpy as np  # noqa: E402
import radio_data  # noqa: E402

GROUPS = {"base": np.s_[0:3], "torso": np.s_[3:7], "left_arm": np.s_[7:14], "left_grip": np.s_[14:15],
          "right_arm": np.s_[15:22], "right_grip": np.s_[22:23]}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="radio_lora")
    ap.add_argument("--every", type=int, default=60, help="evaluate every k-th frame of each held-out episode")
    ap.add_argument("--episodes", default=None, help="comma list (default: ft_config.HOLDOUT)")
    ap.add_argument("--out", default=None, help="write results json here (default <ckpt>/eval_offline.json)")
    args = ap.parse_args()

    import jax
    from openpi.policies import policy_config

    cfg = ft_config.get_config(args.config)
    policy = policy_config.create_trained_policy(cfg, args.ckpt)
    eps = [int(e) for e in args.episodes.split(",")] if args.episodes else list(ft_config.HOLDOUT)
    ds = radio_data.RadioDataset(action_horizon=cfg.model.action_horizon, episodes=eps)
    err, hold = {k: [] for k in GROUPS}, {k: [] for k in GROUPS}
    policy._rng = jax.random.key(0)
    for i, (e, n) in enumerate(zip(ds.episodes, ds.lengths)):
        for t in range(0, n - 1, args.every):
            x = ds[int(ds.starts[i] + t)]
            gt = x["actions"]
            pred = np.asarray(policy.infer({k: v for k, v in x.items() if k != "actions"})["actions"])
            assert pred.shape == gt.shape, (pred.shape, gt.shape)
            s = x["observation/state"]
            for k, sl in GROUPS.items():
                err[k].append(np.abs(pred[:, sl] - gt[:, sl]).mean())
                if k == "base":  # baseline: zero velocity
                    hold[k].append(np.abs(gt[:, sl]).mean())
                elif "grip" in k:  # state is finger width, action is a [-1, 1] command: no hold baseline
                    hold[k].append(np.nan)
                else:
                    hold[k].append(np.abs(s[sl][None] - gt[:, sl]).mean())
    res = {k: {"pred": float(np.mean(err[k])), "hold": float(np.nanmean(hold[k])) if "grip" not in k else None} for k in GROUPS}
    res["n"] = len(err["base"])
    print(f"{res['n']} chunks from episodes {eps}")
    for k in GROUPS:
        print(f"  {k:10s} |pred-demo| {res[k]['pred']:.4f}   hold-current baseline {res[k]['hold']}")
    json.dump(res, open(args.out or os.path.join(args.ckpt, "eval_offline.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
