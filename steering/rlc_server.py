"""Serve the RLC policy (1st place, 2025 BEHAVIOR Challenge; IliaLarchenko/behavior-1k-solution) to this fork's
BEHAVIOR evaluator, the same way steering/comet_server.py serves Comet.

Run with RLC's uv venv (setup: steering/NOTES.md "RLC setup"):

    cd /home/behavior/ext/rlc/solution
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 .venv/bin/python \
        /home/behavior/BEHAVIOR-1K/steering/rlc_server.py --ckpt /home/behavior/ext/rlc/ckpt/checkpoint_2 --port 8000

Replaces RLC's scripts/serve_b1k.py without editing RLC, openpi or BEHAVIOR code. Defaults = serve_b1k.py's defaults
(their challenge settings): 30-action chunks, execute 26 compressed into 20 steps by cubic interpolation (off when the
gripper moves a lot), 4 actions kept for soft inpainting of the next chunk, 20 flow steps, stage voting
(history 3, 2 votes), eval tricks on (gripper-correction rules incl. the radio stage-4 rule).
Shims (see comet_server.py):
  1. `omnigibson.learning.*` -> stand-ins: the fork's network_utils and its 61-d PROPRIOCEPTION_INDICES (RLC only
     reads base_qvel, trunk/arm qpos and gripper qpos, all present).
  2. Batched obs (leading num_envs dim) -> unbatched; RLC's wrapper returns a (23,) action -> (1, 23).
  3. task_id: RLC reads obs["task_id"][0] (2025 index; radio = 0, same in the fork's 2026 list). We set it from
     --task-id so the model always gets the right task embedding.
  4. Checkpoint loading: openpi.models.model.restore_params is replaced by restore_params_lowmem (same bf16 params,
     host-RAM peak ~7 GB instead of > 15 GB). --orbax-restore uses the original.
Per-episode seeding of the sampling RNG from `steering_seed`; `steering_target` is popped (used by --best-of-n).

CPU-only (tests): CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu XLA_FLAGS=--xla_disable_hlo_passes=while-loop-invariant-code-motion
(without that flag XLA:CPU hoists fp32 copies of all bf16 weights out of the layer scan and needs > 20 GB RAM).
"""

import argparse
import copy
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comet_server import SEED_KEY, TARGET_KEY, install_omnigibson_shim  # noqa: E402

RADIO_TASK_ID = 0  # 2025 challenge task index (b1k TASK_NUM_STAGES[0] = 5 stages)


def restore_params_lowmem(params_path, *, restore_type=None, dtype=None, sharding=None):
    """Drop-in for openpi.models.model.restore_params that reads the orbax/ocdbt checkpoint one zarr chunk at a time
    with tensorstore and casts each chunk to `dtype` immediately. openpi's version has orbax read every fp32 array
    (12.6 GB) before casting to bf16, which OOMs a CPU-only run on this 31 GB host (and doubles host RAM on GPU).
    Returns the same pure dict (no 'params' / 'value' levels), as jax arrays on the default device (or `sharding`)."""
    import ast
    import itertools
    import json

    import jax
    import ml_dtypes
    import tensorstore as ts
    from flax import traverse_util

    params_path = os.path.abspath(str(params_path))
    meta = json.load(open(os.path.join(params_path, "_METADATA")))["tree_metadata"]
    ctx = ts.Context({"cache_pool": {"total_bytes_limit": 0}})
    np_dtype = None if dtype is None else np.dtype(ml_dtypes.bfloat16 if "bfloat16" in str(dtype) else dtype)
    flat = {}
    for key in meta:
        path = ast.literal_eval(key)  # ('params', 'PaliGemma', ..., 'value')
        spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{params_path}", "path": ".".join(path)}}
        arr = ts.open(spec, open=True, context=ctx).result()
        src = arr.dtype.numpy_dtype
        cast = np_dtype if (np_dtype is not None and np.issubdtype(src, np.floating)) else src
        out = np.empty(arr.shape, dtype=cast)
        chunk = arr.chunk_layout.read_chunk.shape or arr.shape
        for idx in itertools.product(*(range(0, n, c) for n, c in zip(arr.shape, chunk))):
            sl = tuple(slice(i0, min(i0 + c, n)) for i0, c, n in zip(idx, chunk, arr.shape))
            out[sl] = arr[sl].read().result().astype(cast)
        flat[path] = jax.device_put(out, sharding) if sharding is not None else jax.device_put(out)
        del out
    params = traverse_util.unflatten_dict(flat)["params"]
    flat = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat):
        flat = {kp[:-1]: v for kp, v in flat.items()}
    logging.info(f"low-memory restore: {len(flat)} arrays from {params_path} as {np_dtype}")
    return traverse_util.unflatten_dict(flat)


class BestOfN:
    """Sample N action chunks (different flow noise) and keep the one whose predicted hand path (URDF FK) gets closest
    to `self.target` (radio button in the robot base frame, from the client's --inject-target). Wraps the openpi
    policy, so RLC's wrapper applies its correction rules / interpolation / inpainting to the chosen chunk."""

    def __init__(self, policy, n):
        from urdf_fk import R1ProFK

        self.policy = policy
        self.n = n
        self.fk = R1ProFK()
        self.target = None
        self.last_scores = None

    @property
    def metadata(self):
        return self.policy.metadata

    def reset(self):
        self.policy.reset()

    def score(self, a):
        best = np.inf
        for t in (6, 13, 20, 25):  # within the 26 executed actions of the 30-action chunk
            h = self.fk.hands(a[t, 3:7], a[t, 7:14], a[t, 15:22])
            best = min(best, *(np.linalg.norm(h[s] - self.target) for s in ("left", "right")))
        return float(best)

    def infer(self, obs, **kwargs):
        if self.n <= 1 or self.target is None:
            return self.policy.infer(obs, **kwargs)
        # Fresh copies per candidate: openpi's DeltaActions subtracts the state from `initial_actions` IN PLACE, so
        # reusing one array would shift the inpainting targets by -state on every extra sample.
        cands = [
            self.policy.infer(copy.deepcopy(obs), **{k: np.array(v, copy=True) for k, v in kwargs.items()})
            for _ in range(self.n)
        ]
        scores = [self.score(np.asarray(c["actions"], dtype=np.float64)) for c in cands]
        self.last_scores = scores
        logging.info(f"best-of-{self.n}: hand-to-button {min(scores):.3f} m (worst {max(scores):.3f})")
        return cands[int(np.argmin(scores))]


class BatchedRLCPolicy:
    """Adapts RLC's B1KPolicyWrapper (unbatched obs, (23,) action) to the fork's batched protocol, num_envs == 1."""

    def __init__(self, rlc_policy, openpi_policy, task_id, best_of_n=None, no_inpaint=False, log_dir=None):
        self.rlc = rlc_policy
        self.no_inpaint = no_inpaint
        self.log_dir = log_dir  # save obs + chunk at every inference (debugging closed-loop behaviour)
        self._episode = -1
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.openpi_policy = openpi_policy
        self.task_id = task_id
        self.best_of_n = best_of_n
        self._need_seed = True

    def reset(self):
        self.rlc.reset()
        self._need_seed = True
        self._episode += 1

    def act(self, obs):
        unbatched = {}
        for k, v in obs.items():
            v = np.asarray(v)
            assert v.shape[:1] == (1,), f"only num_envs=1 supported; obs[{k!r}] has shape {v.shape}"
            unbatched[k] = v[0]
        seed = unbatched.pop(SEED_KEY, None)
        target = unbatched.pop(TARGET_KEY, None)
        unbatched["task_id"] = np.array([self.task_id], dtype=np.int64)  # RLC reads obs["task_id"][0]
        if self.best_of_n is not None and target is not None:
            self.best_of_n.target = np.asarray(target, dtype=np.float64).reshape(3)
        if self._need_seed and seed is not None:
            import jax

            s = int(np.asarray(seed).reshape(-1)[0])
            # PiBehaviorPolicy splits self._rng on every infer; the model draws its (correlated) flow noise from it.
            self.openpi_policy._rng = jax.random.key(s)
            logging.info(f"Episode start: policy RNG seeded with {s}")
        self._need_seed = False
        n_pred = self.rlc.prediction_count
        action = self.rlc.act(unbatched)  # torch (23,)
        if self.log_dir and self.rlc.prediction_count != n_pred:  # this step ran an inference
            from b1k.policies.b1k_policy import extract_state_from_proprio
            from openpi_client.image_tools import resize_with_pad

            cams = [k for k in unbatched if k.endswith("::rgb")]
            np.savez_compressed(
                os.path.join(self.log_dir, f"ep{self._episode:03d}_step{self.rlc.step_count - 1:05d}.npz"),
                proprio=np.asarray(unbatched["robot_r1::proprio"], dtype=np.float32),
                state23=extract_state_from_proprio(np.asarray(unbatched["robot_r1::proprio"], dtype=np.float64)),
                executed=np.asarray(self.rlc.last_actions, dtype=np.float32),  # after rules/compression
                next_initial=np.zeros((0, 23)) if self.rlc.next_initial_actions is None else self.rlc.next_initial_actions,
                stage=self.rlc.current_stage,
                images=np.stack([resize_with_pad(np.asarray(unbatched[k])[..., :3], 224, 224) for k in sorted(cams)]),
                cams=np.array(sorted(cams)),
            )
        if self.no_inpaint:
            self.rlc.next_initial_actions = None  # each chunk sampled freely (A/B knob; RLC default = inpaint 4)
        action = action.reshape(1, -1)
        assert tuple(action.shape) == (1, 23), f"unexpected action shape {tuple(action.shape)}"
        return action


def build_policy(args):
    """Load the checkpoint and build the full RLC inference stack. Returns (BatchedRLCPolicy, openpi policy)."""
    install_omnigibson_shim()
    if not args.orbax_restore:
        import openpi.models.model as _model

        _model.restore_params = restore_params_lowmem  # create_trained_policy calls _model.restore_params(...)
    from b1k.policies import policy_config as _policy_config
    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
    from b1k.training import config as _config

    train_config = _config.get_config(args.config)
    # Norm stats (+ action correlation matrix) come from <ckpt>/assets/IliaLarchenko/behavior_224_rgb/norm_stats.json.
    openpi_policy = _policy_config.create_trained_policy(
        train_config, args.ckpt, sample_kwargs={"num_steps": args.num_steps}
    )
    inner = openpi_policy
    bon = None
    if args.best_of_n > 1:
        bon = inner = BestOfN(openpi_policy, args.best_of_n)
        logging.info(f"best-of-{args.best_of_n} chunk selection enabled (needs client --inject-target)")
    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        num_steps=args.num_steps,
        apply_eval_tricks=not args.no_eval_tricks,
    )
    logging.info(f"RLC wrapper config: {wrapper_config}")
    rlc = B1KPolicyWrapper(inner, task_id=args.task_id, config=wrapper_config)
    return (
        BatchedRLCPolicy(rlc, openpi_policy, args.task_id, best_of_n=bon, no_inpaint=args.no_inpaint, log_dir=args.log_chunks),
        openpi_policy,
    )


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/home/behavior/ext/rlc/ckpt/checkpoint_2", help="RLC checkpoint dir.")
    ap.add_argument("--config", default="pi_behavior_b1k_fast", help="RLC TrainConfig name.")
    ap.add_argument("--task-id", type=int, default=RADIO_TASK_ID, help="2025 task index (turning_on_radio = 0).")
    ap.add_argument("--port", type=int, default=8000)
    # serve_b1k.py defaults
    ap.add_argument("--actions-to-execute", type=int, default=26)
    ap.add_argument("--actions-to-keep", type=int, default=4)
    ap.add_argument("--execute-in-n-steps", type=int, default=20)
    ap.add_argument("--history-len", type=int, default=3)
    ap.add_argument("--votes-to-promote", type=int, default=2)
    ap.add_argument("--num-steps", type=int, default=20, help="Flow-matching denoising steps.")
    ap.add_argument("--no-eval-tricks", action="store_true", help="Disable correction rules + gripper check.")
    ap.add_argument("--log-chunks", default=None, help="Dir: save obs (224 imgs, proprio) + executed chunk per inference.")
    ap.add_argument("--no-inpaint", action="store_true", help="Don't condition each chunk on the previous chunk's tail.")
    ap.add_argument("--best-of-n", type=int, default=1, help="Best-of-N chunk selection with the FK critic.")
    ap.add_argument(
        "--orbax-restore", action="store_true", help="Use openpi's own restore_params (needs ~2x more host RAM)."
    )
    return ap


def main():
    args = make_parser().parse_args()
    policy, openpi_policy = build_policy(args)
    from omnigibson.learning.utils.network_utils import WebsocketPolicyServer  # shim -> fork's network_utils

    server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=openpi_policy.metadata)
    logging.info(f"Serving RLC {args.ckpt} ({args.config}), task_id {args.task_id}, on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
