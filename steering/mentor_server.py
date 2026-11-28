"""Serve a checkpoint trained with the mentor's openpi fork (github.com/wensi-ai/openpi, branch `behavior`, e.g. the
organisers' released pi0.5 turning_on_radio baseline) to this fork's evaluator / steering/run_rollouts.py.

Run with that fork's uv venv, from its checkout:

    cd /home/behavior/ext/mentor/openpi
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/mentor_server.py \
        --ckpt /home/behavior/ext/mentor/released_pi05_radio/<dir> --port 8000

Mirrors their scripts/b1k/serve_b1k.py (config pi05_b1k, robot b1k/R1Pro, task prompt from TASK_REGISTRY,
B1KPolicyWrapper receding horizon, 16 of 32 actions executed per inference). Differences, none of which change the
model's inputs:
  - websocket server = this fork's network_utils (the protocol run_rollouts.py speaks), loaded by path;
  - obs keys renamed robot_r1::* -> robot::* (their R1Pro config uses robot name "robot");
  - the policy's sampling RNG is reseeded each episode from the `steering_seed` obs key (reproducible pairs);
  - `--prompt` overrides the task prompt (language steering).
Their state = proprio[base_qvel, trunk, left arm, sum(left fingers), right arm, sum(right fingers)] from the same
61-d proprio layout as comet_server.PROPRIOCEPTION_INDICES.
"""

import argparse
import dataclasses
import importlib.util
import logging
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FORK_NETWORK_UTILS = os.path.join(REPO_ROOT, "OmniGibson", "omnigibson", "eval", "utils", "network_utils.py")
SEED_KEY, TARGET_KEY = "steering_seed", "steering_target"


def load_network_utils():
    spec = importlib.util.spec_from_file_location("fork_network_utils", FORK_NETWORK_UTILS)
    net = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(net)
    return net


class Adapter:
    def __init__(self, wrapper, openpi_policy):
        self.wrapper, self.openpi_policy = wrapper, openpi_policy
        self._need_seed = True

    def reset(self):
        self.wrapper.reset()
        self._need_seed = True

    def act(self, obs):
        seed = obs.get(SEED_KEY)
        obs = {k.replace("robot_r1", "robot"): np.asarray(v) for k, v in obs.items() if k not in (SEED_KEY, TARGET_KEY)}
        if self._need_seed and seed is not None:
            import jax

            s = int(np.asarray(seed).reshape(-1)[0])
            self.openpi_policy._rng = jax.random.key(s)
            logging.info(f"Episode start: policy RNG seeded with {s}")
        self._need_seed = False
        action = self.wrapper.act(obs)  # torch (1, 23) for batched (num_envs=1) obs
        assert tuple(action.shape) == (1, 23), f"unexpected action shape {tuple(action.shape)}"
        return action


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_b1k")
    ap.add_argument("--robot", default="b1k/R1Pro")
    ap.add_argument("--task", default="b1k/turning_on_radio")
    ap.add_argument("--repo-id", default="turning_on_radio", help="norm-stats asset id inside <ckpt>/assets")
    ap.add_argument("--control-mode", default="receding_horizon")
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    from openpi.configs.tasks import TASK_REGISTRY
    from openpi.policies import policy_config as _policy_config
    from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
    from openpi.training import config as _config

    bucket, name = args.task.split("/")
    prompt = args.prompt or TASK_REGISTRY[bucket][name]
    config = _config.get_config(args.config)
    config = dataclasses.replace(
        config, data=dataclasses.replace(config.data, repo_id=args.repo_id, robot_config_name=args.robot)
    )
    policy = _policy_config.create_trained_policy(config, args.ckpt, default_prompt=prompt)
    wrapper = B1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=prompt,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
    )
    logging.info(f"PROMPT in use: {prompt!r}; control {args.control_mode} K={args.action_horizon}")
    net = load_network_utils()
    server = net.WebsocketPolicyServer(
        policy=Adapter(wrapper, policy), host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    logging.info(f"Serving {args.ckpt} ({args.config}) on port {args.port}")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
