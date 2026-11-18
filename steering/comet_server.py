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
TARGET_KEY = "steering_target"

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


class ActionWiring:
    """Optional re-wiring of a policy's predicted action chunk before Comet's wrapper queues it.

    delta_joints: the policy predicts torso/arm joints as offsets from the current joint positions (base pi0.5's
      pre-training convention); add the current joints back. Requires a checkpoint whose action norm stats for those
      dims are delta stats (steering/delta_norm_stats.py).
    freeze_torso: hold the torso at its episode-start position (removes base pi0.5's torso folding).
    freeze_base: zero the base velocity (the policy only moves torso/arms; a client-side approach assist still drives).
    """

    # 61-d proprio (fork layout) -> 23-d action layout
    JOINTS = [(slice(3, 7), slice(53, 57)), (slice(7, 14), slice(3, 10)), (slice(15, 22), slice(28, 35))]

    def __init__(self, policy, delta_joints=False, freeze_torso=False, best_of_n=1, freeze_base=False):
        self.policy = policy
        self.freeze_base = freeze_base
        self.delta_joints = delta_joints
        self.freeze_torso = freeze_torso
        self.torso0 = None
        # best-of-N: sample N chunks (different diffusion noise) and keep the one whose predicted hand path (URDF FK)
        # gets closest to `self.target` (button position in the robot base frame, sent by the client each step).
        self.best_of_n = best_of_n
        self.target = None
        self.last_scores = None
        if best_of_n > 1:
            from urdf_fk import R1ProFK

            self.fk = R1ProFK()

    def reset(self):
        self.torso0 = None

    def infer(self, batch):
        if self.best_of_n <= 1 or self.target is None:
            return self._one(batch)
        cands = [self._one(batch) for _ in range(self.best_of_n)]
        scores = [self.score(c["actions"]) for c in cands]
        self.last_scores = scores
        logging.info(f"best-of-{self.best_of_n}: hand-to-button {min(scores):.3f} m (worst {max(scores):.3f})")
        return cands[int(np.argmin(scores))]

    def score(self, a):
        """Closest predicted end-effector distance to the target over the chunk (checked at 4 points)."""
        best = np.inf
        for t in (7, 15, 23, 31):
            h = self.fk.hands(a[t, 3:7], a[t, 7:14], a[t, 15:22])
            best = min(best, *(np.linalg.norm(h[s] - self.target) for s in ("left", "right")))
        return float(best)

    def _one(self, batch):
        out = self.policy.infer(batch)
        a = np.array(out["actions"], dtype=np.float64, copy=True)
        q = np.asarray(batch["observation/state"], dtype=np.float64)
        if self.delta_joints:
            for asl, qsl in self.JOINTS:
                a[:, asl] += q[qsl]
        if self.freeze_torso:
            if self.torso0 is None:
                self.torso0 = q[53:57].copy()
            a[:, 3:7] = self.torso0
        if self.freeze_base:
            a[:, 0:3] = 0.0
        return {**out, "actions": a}


class BatchedCometPolicy:
    """Adapts Comet's B1KPolicyWrapper (unbatched) to the fork's batched obs, num_envs == 1 only."""

    def __init__(self, comet_policy, openpi_policy, dump_dir=None, wiring=None):
        self.comet = comet_policy
        self.wiring = wiring
        self.openpi_policy = openpi_policy
        self._need_seed = True
        # Optional: save the exact (preprocessed) observation at every inference step, for offline prompt probing
        # (steering/prompt_probe.py). Comet's receding-horizon wrapper infers when its action queue is empty.
        self.dump_dir = dump_dir
        self._episode = -1
        self._step = 0
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    def reset(self):
        self.comet.reset()
        if self.wiring is not None:
            self.wiring.reset()
        self._need_seed = True
        self._episode += 1
        self._step = 0

    def act(self, obs):
        unbatched = {}
        for k, v in obs.items():
            v = np.asarray(v)
            assert v.shape[:1] == (1,), f"only num_envs=1 supported; obs[{k!r}] has shape {v.shape}"
            unbatched[k] = v[0]
        seed = unbatched.pop(SEED_KEY, None)
        target = unbatched.pop(TARGET_KEY, None)
        if self.wiring is not None and target is not None:
            self.wiring.target = np.asarray(target, dtype=np.float64).reshape(3)
        if self._need_seed and seed is not None:
            import jax

            # openpi Policy samples diffusion noise from self._rng (split on every infer call).
            self.openpi_policy._rng = jax.random.key(int(np.asarray(seed).reshape(-1)[0]))
            logging.info(f"Episode start: policy RNG seeded with {int(np.asarray(seed).reshape(-1)[0])}")
        self._need_seed = False
        if self.dump_dir and len(self.comet.action_queue) == 0:
            po = self.comet.process_obs(unbatched)
            os.makedirs(self.dump_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(self.dump_dir, f"ep{self._episode:03d}_step{self._step:05d}.npz"),
                images=np.asarray(po["observation"][0], dtype=np.uint8),  # (3 cams, 224, 224, 3): head, left, right
                proprio=np.asarray(po["proprio"][0], dtype=np.float32),  # (61,)
                seed=-1 if seed is None else int(np.asarray(seed).reshape(-1)[0]),
                prompt=self.comet.task_prompt,
            )
        self._step += 1
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
    ap.add_argument("--dump-obs", default=None, help="Directory to save the observation at every inference step.")
    ap.add_argument(
        "--prompt",
        default=None,
        help="Override the task prompt (default: Comet's scripts/task_mapping.json entry for --task).",
    )
    ap.add_argument("--delta-joints", action="store_true", help="See ActionWiring (needs delta norm stats).")
    ap.add_argument("--freeze-torso", action="store_true", help="Hold the torso at its episode-start position.")
    ap.add_argument("--freeze-base", action="store_true", help="Zero the policy's base velocity commands.")
    ap.add_argument(
        "--best-of-n",
        type=int,
        default=1,
        help="Best-of-N chunk selection with the FK hand-to-button "
        "critic (needs the client to run with --inject-target).",
    )
    args = ap.parse_args()

    net = install_omnigibson_shim()

    from openpi.policies import policy_config as _policy_config
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.training import config as _config

    train_config = _config.get_config(args.config)
    # Norm stats come from <ckpt>/assets/<asset_id>/norm_stats.json (the radio checkpoint ships its own).
    openpi_policy = _policy_config.create_trained_policy(train_config, args.ckpt)
    wiring = None
    inner = openpi_policy
    if args.delta_joints or args.freeze_torso or args.freeze_base or args.best_of_n > 1:
        wiring = inner = ActionWiring(
            openpi_policy,
            delta_joints=args.delta_joints,
            freeze_torso=args.freeze_torso,
            best_of_n=args.best_of_n,
            freeze_base=args.freeze_base,
        )
        logging.info(
            f"ACTION WIRING: delta_joints={args.delta_joints} freeze_torso={args.freeze_torso} "
            f"freeze_base={args.freeze_base} best_of_n={args.best_of_n}"
        )
    comet_policy = B1KPolicyWrapper(inner, task_name=args.task, control_mode=args.control_mode, max_len=args.max_len)
    if args.prompt is not None:
        # B1KPolicyWrapper sends self.task_prompt as the "prompt" on every inference call.
        comet_policy.task_prompt = args.prompt
    logging.info(f"PROMPT in use: {comet_policy.task_prompt!r}")
    server = net.WebsocketPolicyServer(
        policy=BatchedCometPolicy(comet_policy, openpi_policy, dump_dir=args.dump_obs, wiring=wiring),
        host="0.0.0.0",
        port=args.port,
        metadata=openpi_policy.metadata,
    )
    logging.info(f"Serving {args.ckpt} ({args.config}) for {args.task} on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
