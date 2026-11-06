"""Serve the Comet (openpi-comet) pi0.5 policy to this fork's BEHAVIOR evaluator.

Run with Comet's uv venv, from the openpi-comet checkout (B1KPolicyWrapper reads scripts/task_mapping.json
relative to cwd):

    cd /home/behavior/ext/openpi-comet
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/comet_server.py \
        --ckpt /home/behavior/ext/ckpt/pi05-turning_on_radio-sft --task turning_on_radio --port 8000

This replaces openpi-comet's scripts/serve_b1k.py without editing Comet or BEHAVIOR code. It fixes two
version mismatches (see steering/NOTES.md):
  1. Comet imports `omnigibson.learning.*` (BEHAVIOR v3.7); this fork renamed it to `omnigibson.eval.*`, and
     installing OmniGibson into Comet's venv is unnecessary. We register small stand-in modules instead:
     - network_utils: the fork's own file, loaded by path (same websocket/msgpack protocol as the client).
     - eval_utils.PROPRIOCEPTION_INDICES: the fork's 61-d proprio layout. Comet only reads base_qvel,
       trunk_qpos, arm_{left,right}_qpos and gripper_{left,right}_qpos from it, all present in this layout.
  2. The fork's evaluator sends every observation with a leading num_envs dim; Comet expects unbatched obs.
It also re-seeds the policy's sampling RNG at the start of each episode from the `steering_seed` obs key
(set by steering/compat_wrapper.py), so each rollout's diffusion noise is reproducible.
"""

import argparse
import importlib.util
import logging
import os
import sys
import types
from collections import OrderedDict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORK_NETWORK_UTILS = os.path.join(REPO_ROOT, "OmniGibson", "omnigibson", "eval", "utils", "network_utils.py")
SEED_KEY = "steering_seed"

# Copied from OmniGibson/omnigibson/eval/utils/eval_utils.py (fork, v3.9.3). Must match eval/r1pro.yaml
# proprio_obs order; steering/compat_wrapper.py asserts the proprio size at runtime.
PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
            "base_qvel": np.s_[0:3],
            "arm_left_qpos": np.s_[3:10],
            "arm_left_qvel": np.s_[10:17],
            "eef_left_pos": np.s_[17:20],
            "eef_left_quat": np.s_[20:24],
            "gripper_left_qpos": np.s_[24:26],
            "gripper_left_qvel": np.s_[26:28],
            "arm_right_qpos": np.s_[28:35],
            "arm_right_qvel": np.s_[35:42],
            "eef_right_pos": np.s_[42:45],
            "eef_right_quat": np.s_[45:49],
            "gripper_right_qpos": np.s_[49:51],
            "gripper_right_qvel": np.s_[51:53],
            "trunk_qpos": np.s_[53:57],
            "trunk_qvel": np.s_[57:61],
        }
    ),
}


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def install_omnigibson_shim():
    """Register the minimal `omnigibson.*` modules Comet's server-side code imports."""
    og = _module("omnigibson")
    og.__path__ = []
    _module("omnigibson.macros", gm=types.SimpleNamespace(DEBUG=False))
    _module("omnigibson.learning").__path__ = []
    _module("omnigibson.learning.utils").__path__ = []
    _module("omnigibson.learning.utils.eval_utils", PROPRIOCEPTION_INDICES=PROPRIOCEPTION_INDICES)
    spec = importlib.util.spec_from_file_location("omnigibson.learning.utils.network_utils", FORK_NETWORK_UTILS)
    net = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = net
    spec.loader.exec_module(net)
    return net


class BatchedCometPolicy:
    """Adapts Comet's B1KPolicyWrapper (unbatched) to the fork's batched obs, num_envs == 1 only."""

    def __init__(self, comet_policy, openpi_policy):
        self.comet = comet_policy
        self.openpi_policy = openpi_policy
        self._need_seed = True

    def reset(self):
        self.comet.reset()
        self._need_seed = True

    def act(self, obs):
        unbatched = {}
        for k, v in obs.items():
            v = np.asarray(v)
            assert v.shape[:1] == (1,), f"only num_envs=1 supported; obs[{k!r}] has shape {v.shape}"
            unbatched[k] = v[0]
        seed = unbatched.pop(SEED_KEY, None)
        if self._need_seed and seed is not None:
            import jax

            # openpi Policy samples diffusion noise from self._rng (split on every infer call).
            self.openpi_policy._rng = jax.random.key(int(np.asarray(seed).reshape(-1)[0]))
            logging.info(f"Episode start: policy RNG seeded with {int(np.asarray(seed).reshape(-1)[0])}")
        self._need_seed = False
        action = self.comet.act(unbatched)  # torch (1, 23)
        assert tuple(action.shape) == (1, 23), f"unexpected action shape {tuple(action.shape)}"
        return action


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="openpi checkpoint dir (contains params/ and assets/).")
    ap.add_argument(
        "--config",
        default="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft",
        help="openpi-comet TrainConfig name (the radio SFT config, as used in openpi-comet issue #1).",
    )
    ap.add_argument("--task", default="turning_on_radio")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--control-mode", default="receeding_horizon")
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument(
        "--prompt",
        default=None,
        help="Override the task prompt (default: Comet's scripts/task_mapping.json entry for --task).",
    )
    args = ap.parse_args()

    net = install_omnigibson_shim()

    from openpi.policies import policy_config as _policy_config
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    # Norm stats come from <ckpt>/assets/<asset_id>/norm_stats.json (the radio checkpoint ships its own).
    openpi_policy = _policy_config.create_trained_policy(train_config, args.ckpt)
    comet_policy = B1KPolicyWrapper(
        openpi_policy, task_name=args.task, control_mode=args.control_mode, max_len=args.max_len
    )
    if args.prompt is not None:
        # B1KPolicyWrapper sends self.task_prompt as the "prompt" on every inference call.
        comet_policy.task_prompt = args.prompt
    logging.info(f"PROMPT in use: {comet_policy.task_prompt!r}")
    server = net.WebsocketPolicyServer(
        policy=BatchedCometPolicy(comet_policy, openpi_policy),
        host="0.0.0.0",
        port=args.port,
        metadata=openpi_policy.metadata,
    )
    logging.info(f"Serving {args.ckpt} ({args.config}) for {args.task} on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
