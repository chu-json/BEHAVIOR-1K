"""Continue fine-tuning the organisers' released pi0.5 turning_on_radio checkpoint on our 195 radio demos with the
MENTOR's openpi fork (github.com/wensi-ai/openpi branch behavior @0cc8e35, /home/behavior/ext/mentor/openpi), so the
checkpoints are served unchanged by steering/mentor_server.py (config pi05_b1k, <ckpt>/params,
<ckpt>/assets/turning_on_radio/norm_stats.json).

Recipe = behavior26's control arm (their pi05_b1k_frozen_vlm, scripts/shot_one.sh INIT_FROM=released
STATS_FROM=released): init from the released params, the released norm stats, VLM frozen (SigLIP + PaliGemma expert 0;
trainable = action expert `*_1` + action_in/out_proj + time_mlp; frozen params bf16), no EMA, AdamW, cosine LR
(1000 warmup, peak 2.5e-5 -> 2.5e-6 at the run length), batch 32, seed 42. No openpi edits: their trainer
scripts/b1k/train_b1k.py main() is called with this config, `create_b1k_dataset` is swapped for our
mentor_data.MentorRadioDataset (their full transform pipeline still runs), and optionally (--prefix-stopgrad)
Pi0.compute_loss is replaced by an exactly equivalent two-pass version: the frozen prefix (images + prompt + state
tokens) is run once into a KV cache under stop_gradient, then the suffix (noisy actions) attends to it, as in
Pi0.sample_actions. Only valid because nothing in the prefix is trainable (asserted).

    cd /home/behavior/ext/mentor/openpi
    # validation 1: released ckpt, training-side transforms vs mentor_server's path, offline error vs hold
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/ft/mentor_ft.py check-repr
    # stop-gradient parity (loss + expert grads, patched vs unpatched, same batch/rng)
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/ft/mentor_ft.py parity
    # train
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 .venv/bin/python -u /home/behavior/BEHAVIOR-1K/steering/ft/mentor_ft.py train \
        --exp ft_cont --steps 6000 --batch-size 32 --prefix-stopgrad
Checkpoints: /home/behavior/ext/ckpt/ft_cont/<exp>/<step>/{params,train_state,assets/turning_on_radio}.
"""

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (HERE, os.path.dirname(HERE)) if p not in sys.path]

import numpy as np  # noqa: E402

import mentor_data  # noqa: E402
import radio_data  # noqa: E402

MENTOR = "/home/behavior/ext/mentor/openpi"
RELEASED = "/home/behavior/ext/mentor/released_pi05_radio/pi05_turn_on_the_radio"
ASSET_ID = "turning_on_radio"
CKPT_BASE = "/home/behavior/ext/ckpt/ft_cont"
CONFIG_NAME = "pi05_b1k_frozen_vlm"  # checkpoint dir = CKPT_BASE/CONFIG_NAME/<exp>
GROUPS = {"base": np.s_[0:3], "torso": np.s_[3:7], "left_arm": np.s_[7:14], "left_grip": np.s_[14:15],
          "right_arm": np.s_[15:22], "right_grip": np.s_[22:23]}  # fmt: skip


def radio_prompt():
    from openpi.configs.tasks import TASK_REGISTRY

    p = TASK_REGISTRY["b1k"]["turning_on_radio"]
    assert p == radio_data.RADIO_PROMPT, (p, radio_data.RADIO_PROMPT)
    return p


def freeze_vlm_filter():
    """behavior26 patch 0002 freeze_vlm_filter(): SigLIP + PaliGemma expert 0 frozen, action expert (`_1`) trains."""
    import flax.nnx as nnx
    from openpi.shared import nnx_utils

    return nnx.All(nnx_utils.PathRegex(".*(llm|img).*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))


def make_config(exp="debug", batch_size=32, steps=6000, peak_lr=2.5e-5, warmup=1000, decay_lr=2.5e-6,
                save_interval=1000, keep_period=2000, log_interval=10, workers=1, seed=42, resume=False,
                overwrite=False, init_params=RELEASED + "/params"):  # fmt: skip
    from openpi.training import config as _config
    from openpi.training import optimizer as _optimizer
    from openpi.training import weight_loaders

    base = _config.get_config("pi05_b1k")  # the config the released ckpt was trained with / mentor_server serves
    data = dataclasses.replace(
        base.data,
        repo_id=ASSET_ID,
        robot_config_name="b1k/R1Pro",
        assets=_config.AssetsConfig(assets_dir=RELEASED + "/assets", asset_id=ASSET_ID),
    )
    return dataclasses.replace(
        base,
        name=CONFIG_NAME,
        exp_name=exp,
        data=data,
        weight_loader=weight_loaders.CheckpointWeightLoader(init_params),
        freeze_filter=freeze_vlm_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(warmup, steps - 1), peak_lr=peak_lr, decay_steps=steps, decay_lr=decay_lr
        ),
        batch_size=batch_size,
        num_train_steps=steps,
        save_interval=save_interval,
        keep_period=keep_period,
        log_interval=log_interval,
        num_workers=workers,
        seed=seed,
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=False,
        checkpoint_base_dir=CKPT_BASE,
        assets_base_dir=CKPT_BASE + "/assets",
        val_log_interval=None,
    )


def install_dataset(exclude=mentor_data.HOLDOUT):
    """Swap the mentor's LeRobot B1K dataset for ours; their transform_dataset/TorchDataLoader still run."""
    from openpi.training import data_loader as _dl

    prompt = radio_prompt()

    def create_b1k_dataset(data_config, action_horizon, *a, **k):
        ds = mentor_data.MentorRadioDataset(prompt, action_horizon=action_horizon, exclude=exclude)
        logging.info(f"MentorRadioDataset: {len(ds.ds.episodes)} episodes, {len(ds)} frames (held out {exclude})")
        return ds

    _dl.create_b1k_dataset = create_b1k_dataset


def compute_loss_prefix_stopgrad(self, rng, observation, actions, *, train=False):
    """Pi0.compute_loss (pi0.py) with the prefix run separately into a stop_gradient KV cache (as sample_actions).

    Forward is the same function: prefix tokens never attend to suffix tokens (ar_mask), so the prefix stream is
    unchanged, and the suffix sees the same keys/values, positions and mask. Gradients w.r.t. parameters that are
    used only by the suffix (action expert, action/time projections) are unchanged; prefix params get none.
    """
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    kv_cache = jax.lax.stop_gradient(kv_cache)

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_to_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_to_suffix, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )  # fmt: skip
    assert prefix_out is None
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
    return jnp.mean(jnp.square(v_t - u_t), axis=-1)


def install_prefix_stopgrad(config):
    from openpi.models import pi0

    # every parameter the prefix uses must be frozen; check the prefix-only paths by name
    import flax.nnx as nnx
    import jax

    model_shape = nnx.eval_shape(lambda: config.model.create(jax.random.key(0)))
    state = nnx.state(model_shape)
    trainable = state.filter(config.trainable_filter)
    paths = ["/".join(map(str, k)) for k in trainable.flat_state().keys()] if hasattr(trainable, "flat_state") else \
        ["/".join(map(str, k)) for k, _ in nnx.traversals.flatten_mapping(trainable.to_pure_dict()).items()]
    bad = [p for p in paths if p.startswith("PaliGemma/img") or ("llm" in p and "_1" not in p)]
    assert not bad, f"prefix params are trainable, stop-gradient would change the gradient: {bad[:5]}"
    n = sum(int(np.prod(v.shape)) for v in jax.tree.leaves(trainable.to_pure_dict()))
    logging.info(f"prefix stop-gradient installed; {len(paths)} trainable leaves, {n / 1e9:.3f}B params, e.g. "
                 f"{sorted({p.split('/')[0] + '/' + p.split('/')[1] if '/' in p else p for p in paths})[:12]}")
    pi0.Pi0.compute_loss = compute_loss_prefix_stopgrad


def load_train_module():
    import importlib.util

    path = os.path.join(MENTOR, "scripts", "b1k", "train_b1k.py")
    spec = importlib.util.spec_from_file_location("train_b1k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prune_old_train_states(ckpt_dir):
    """Keep train_state only in the newest finished checkpoint (resume needs just that); older kept checkpoints
    (multiples of keep_period) keep params + assets, which is all serving needs (~7.6 GB instead of ~11 GB)."""
    if not os.path.isdir(ckpt_dir):
        return
    steps = sorted(int(d) for d in os.listdir(ckpt_dir) if d.isdigit())
    for s in steps[:-1]:
        ts = os.path.join(ckpt_dir, str(s), "train_state")
        if os.path.isdir(ts):
            shutil.rmtree(ts, ignore_errors=True)
            logging.info(f"pruned {ts}")


# ----------------------------------------------------------------------------------------------------------------
def cmd_train(args):
    cfg = make_config(exp=args.exp, batch_size=args.batch_size, steps=args.steps, peak_lr=args.lr,
                      warmup=args.warmup, save_interval=args.save_interval, keep_period=args.keep_period,
                      log_interval=args.log_interval, workers=args.workers, seed=args.seed, resume=args.resume,
                      overwrite=args.overwrite)  # fmt: skip
    os.makedirs(CKPT_BASE, exist_ok=True)
    with open(os.path.join(CKPT_BASE, f"{args.exp}.args.json"), "w") as f:
        json.dump({**vars(args), "config": repr(cfg)}, f, indent=2, default=str)
    install_dataset()
    if args.prefix_stopgrad:
        install_prefix_stopgrad(cfg)
    train = load_train_module()
    from openpi.training import checkpoints as _ck

    orig_save = _ck.save_state

    def save_state(manager, state, data_loader, step):
        manager.wait_until_finished()
        prune_old_train_states(str(cfg.checkpoint_dir))
        if args.params_only_save:
            # Drop the optimizer state from the saved train_state: the full save pushed host RAM over the edge and
            # systemd-oomd killed the run (2026-11-27). Checkpoints stay servable; --resume is not possible.
            import dataclasses

            state = dataclasses.replace(state, opt_state=None)
        return orig_save(manager, state, data_loader, step)

    train._checkpoints.save_state = save_state
    logging.info(f"ft-cont: {cfg.name}/{cfg.exp_name} bs={cfg.batch_size} steps={cfg.num_train_steps} "
                 f"lr={cfg.lr_schedule} stopgrad={args.prefix_stopgrad}")
    try:
        train.main(cfg)
    finally:
        import jax

        st = jax.devices()[0].memory_stats() or {}
        logging.info(f"GPU memory: peak {st.get('peak_bytes_in_use', 0) / 2**30:.2f} GiB of "
                     f"{st.get('bytes_limit', 0) / 2**30:.2f} GiB")
    prune_old_train_states(str(cfg.checkpoint_dir))
    print("TRAIN_DONE")


def serving_policy(ckpt):
    """Exactly mentor_server.py's policy + wrapper construction."""
    from openpi.policies import policy_config as _policy_config
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.training import config as _config

    prompt = radio_prompt()
    config = _config.get_config("pi05_b1k")
    config = dataclasses.replace(
        config, data=dataclasses.replace(config.data, repo_id=ASSET_ID, robot_config_name="b1k/R1Pro")
    )
    policy = _policy_config.create_trained_policy(config, ckpt, default_prompt=prompt)
    wrapper = B1KPolicyWrapper(policy=policy, robot="b1k/R1Pro", text_prompt=prompt, control_mode="receding_horizon",
                               action_horizon=16, max_len=config.model.action_horizon)  # fmt: skip
    return policy, wrapper


def train_side_policy(model, ckpt):
    """The training data pipeline (make_config + their DataConfig: repack -> B1KInputs -> MappedDeltaActions ->
    Normalize -> model transforms) applied to our dataset rows, inverse transforms on the output."""
    from openpi import transforms as T
    from openpi.policies import policy as _policy
    from openpi.training import checkpoints as _ck

    cfg = make_config()
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ns_ckpt = _ck.load_norm_stats(os.path.join(ckpt, "assets"), ASSET_ID)
    for k in ("state", "actions"):
        assert np.allclose(dc.norm_stats[k].mean, ns_ckpt[k].mean) and np.allclose(dc.norm_stats[k].std, ns_ckpt[k].std)
    return _policy.Policy(
        model,
        transforms=[*dc.repack_transforms.inputs, *dc.data_transforms.inputs,
                    T.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs],
        output_transforms=[*dc.model_transforms.outputs,
                           T.Unnormalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm), *dc.data_transforms.outputs],
    ), dc  # fmt: skip


def cmd_check_repr(args):
    import jax

    ckpt = args.ckpt
    policy, wrapper = serving_policy(ckpt)
    tpol, dc = train_side_policy(policy._model, ckpt)
    logging.info(f"use_quantile_norm={dc.use_quantile_norm}; data transforms {dc.data_transforms.inputs}")
    eps = [int(e) for e in args.episodes.split(",")] if args.episodes else list(mentor_data.HOLDOUT)
    ds = mentor_data.MentorRadioDataset(radio_prompt(), episodes=eps, exclude=())
    rng = np.random.default_rng(0)
    err = {k: [] for k in GROUPS}
    hold = {k: [] for k in GROUPS}
    max_path_diff, n = 0.0, 0
    for i, (e, L) in enumerate(zip(ds.ds.episodes, ds.ds.lengths)):
        for t in range(0, L - 1, args.every):
            idx = int(ds.ds.starts[i] + t)
            row = ds[idx]
            obs, gt, s23 = ds.as_sim_obs(idx)
            noise = rng.standard_normal((32, 32)).astype(np.float32)
            # training-side path: the full dataset row (incl. the demo chunk, which the model never sees at inference;
            # RepackTransform needs every mapped key)
            a_train = np.asarray(tpol.infer(row, noise=noise)["actions"])
            # serving path: mentor_server -> B1KPolicyWrapper.process_input -> policy.infer
            inp = wrapper.process_input(obs)[0]
            a_serve = np.asarray(policy.infer(inp, noise=noise)["actions"])
            max_path_diff = max(max_path_diff, float(np.abs(a_train - a_serve).max()))
            for k, sl in GROUPS.items():
                err[k].append(np.abs(a_serve[:, sl] - gt[:, sl]).mean())
                if k == "base":
                    hold[k].append(np.abs(gt[:, sl]).mean())
                elif "grip" in k:
                    hold[k].append(np.nan)
                else:
                    hold[k].append(np.abs(s23[sl][None] - gt[:, sl]).mean())
            n += 1
    res = {k: {"pred": float(np.mean(err[k])), "hold": None if "grip" in k else float(np.mean(hold[k]))}
           for k in GROUPS}  # fmt: skip
    res.update(n=n, episodes=eps, max_abs_diff_train_vs_serve=max_path_diff, ckpt=ckpt)
    print(f"{n} chunks from episodes {eps}; max |train-side - serve-side| = {max_path_diff:.2e}")
    for k in GROUPS:
        print(f"  {k:10s} |pred-demo| {res[k]['pred']:.4f}   hold/zero baseline {res[k]['hold']}")
    out = args.out or os.path.join(CKPT_BASE, f"check_repr_{os.path.basename(ckpt.rstrip('/'))}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out)
    del jax


def cmd_parity(args):
    """Loss + gradient parity of the stop-gradient compute_loss vs the original, on a real batch, released init."""
    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0
    from openpi.models import model as _model
    from openpi.training import data_loader as _dl
    from openpi.training import sharding

    cfg = make_config(batch_size=args.batch_size, workers=0)
    install_dataset()
    train = load_train_module()
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    ds_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = _dl.create_b1k_data_loader(cfg, sharding=ds_sh, shuffle=True)
    batch = next(iter(loader))
    state, _ = train.init_train_state(cfg, jax.random.key(1), mesh, resume=False)
    model = nnx.merge(state.model_def, state.params)
    model.train()
    obs, actions = batch
    diff_state = nnx.DiffState(0, cfg.trainable_filter)
    orig = pi0.Pi0.compute_loss

    def grads_of(fn):
        pi0.Pi0.compute_loss = fn

        @nnx.jit
        def f(model, obs, actions):
            def loss_fn(m):
                return jnp.mean(m.compute_loss(jax.random.key(7), obs, actions, train=True))

            return nnx.value_and_grad(loss_fn, argnums=diff_state)(model)

        return f(model, obs, actions)

    l0, g0 = grads_of(orig)
    l1, g1 = grads_of(compute_loss_prefix_stopgrad)
    pi0.Pi0.compute_loss = orig
    f0 = traverse_util.flatten_dict(g0.to_pure_dict())
    f1 = traverse_util.flatten_dict(g1.to_pure_dict())
    assert f0.keys() == f1.keys()
    rel = {}
    for k in f0:
        a, b = np.asarray(f0[k], np.float64), np.asarray(f1[k], np.float64)
        rel["/".join(map(str, k))] = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
    gn0 = float(np.sqrt(sum(np.sum(np.asarray(v, np.float64) ** 2) for v in f0.values())))
    gn1 = float(np.sqrt(sum(np.sum(np.asarray(v, np.float64) ** 2) for v in f1.values())))
    cos = float(sum(np.sum(np.asarray(f0[k], np.float64) * np.asarray(f1[k], np.float64)) for k in f0) / (gn0 * gn1))
    worst = sorted(rel.items(), key=lambda kv: -kv[1])[:8]
    print(f"loss orig {float(l0):.6f} stopgrad {float(l1):.6f}; grad norm {gn0:.5f} vs {gn1:.5f}; cosine {cos:.6f}")
    print(f"{len(rel)} trainable leaves; median rel diff {np.median(list(rel.values())):.2e}; worst:")
    for k, v in worst:
        print(f"   {v:.2e}  {k}")
    del _model


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--exp", default="debug")
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=2.5e-5)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--save-interval", type=int, default=1000)
    t.add_argument("--keep-period", type=int, default=2000)
    t.add_argument("--log-interval", type=int, default=10)
    t.add_argument("--workers", type=int, default=1)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--prefix-stopgrad", action="store_true")
    t.add_argument("--params-only-save", action="store_true", help="save params (+step) only, no optimizer state")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("check-repr")
    c.add_argument("--ckpt", default=RELEASED)
    c.add_argument("--every", type=int, default=60)
    c.add_argument("--episodes", default=None)
    c.add_argument("--out", default=None)
    p = sub.add_parser("parity")
    p.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    {"train": cmd_train, "check-repr": cmd_check_repr, "parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    main()
