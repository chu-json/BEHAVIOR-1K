"""Fine-tune base pi0.5 on the 200 radio demos with openpi's (openpi-comet's) JAX training loop.

Builds the TrainConfig in code (steering/ft/ft_config.py), swaps openpi's BEHAVIOR dataset for ours
(steering/ft/ft_loader.py), then calls openpi-comet scripts/train.py main(). No openpi/openpi-comet edits.

    cd /home/behavior/ext/openpi-comet
    # once: norm stats (state + delta actions) from the preprocessed data
    .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/ft/train_radio.py --compute-norm-stats
    # train (checkpoints -> /home/behavior/ext/ckpt/ft/<exp>/<step>/)
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/ft/train_radio.py \
        --exp radio_lora_v1 --variant lora --batch-size 16 --steps 20000
"""

import argparse
import importlib.util
import json
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import ft_loader  # noqa: E402  (installs the omnigibson shim before openpi.training.config is imported)
import ft_config  # noqa: E402
import radio_data  # noqa: E402

OPENPI_TRAIN = "/home/behavior/ext/openpi-comet/scripts/train.py"


def compute_norm_stats(out_dir=ft_config.ASSETS_DIR):
    from openpi.shared import normalize

    stats = radio_data.compute_norm_stats()
    d = os.path.join(out_dir, ft_config.ASSET_ID)
    normalize.save(d, stats)
    print(f"wrote {d}/norm_stats.json")
    for k, s in stats.items():
        print(k, "q01", [round(float(v), 3) for v in s.q01])
        print(k, "q99", [round(float(v), 3) for v in s.q99])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compute-norm-stats", action="store_true")
    ap.add_argument("--exp", default="debug", help="checkpoint dir name under /home/behavior/ext/ckpt/ft")
    ap.add_argument("--variant", default="lora", choices=ft_config.VARIANTS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=2.5e-5, help="peak LR (cosine decay to lr/10, 1k warmup)")
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument("--save-interval", type=int, default=2_000)
    ap.add_argument("--keep-period", type=int, default=6_000, help="keep checkpoints at multiples of this (+ latest)")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-aug", default="none")
    ap.add_argument("--image-aug", default="none")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.compute_norm_stats:
        compute_norm_stats()
        return

    cfg = ft_config.make_config(
        variant=args.variant,
        exp_name=args.exp,
        batch_size=args.batch_size,
        num_train_steps=args.steps,
        peak_lr=args.lr,
        warmup_steps=args.warmup,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        log_interval=args.log_interval,
        num_workers=args.workers,
        seed=args.seed,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    # next to (not inside) the checkpoint dir: openpi's --overwrite wipes that dir
    os.makedirs(ft_config.CKPT_BASE, exist_ok=True)
    with open(os.path.join(ft_config.CKPT_BASE, f"{args.exp}.args.json"), "w") as f:
        json.dump({**vars(args), "config_name": cfg.name}, f, indent=2)

    from openpi.training import data_loader as _data_loader

    _data_loader.create_behavior_data_loader = ft_loader.make_create_fn(args.prompt_aug, args.image_aug)
    spec = importlib.util.spec_from_file_location("openpi_train", OPENPI_TRAIN)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    logging.info(f"steering fine-tune: {cfg.name} exp={cfg.exp_name} bs={cfg.batch_size} steps={cfg.num_train_steps}")
    try:
        train.main(cfg)
    finally:
        import jax

        st = jax.devices()[0].memory_stats() or {}
        logging.info(f"GPU memory: peak_bytes_in_use {st.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB, "
                     f"bytes_limit {st.get('bytes_limit', 0) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
