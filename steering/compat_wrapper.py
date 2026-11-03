"""Client-side env wrappers that make this fork's eval observations look like the 2025 (v3.7) ones Comet saw.

Use via the evaluator's env_wrapper target, e.g. ``steering.compat_wrapper.CompatFullResWrapper``.

Fixes (see steering/NOTES.md):
  #3 base_qvel: v3.7 proprio exposed the holonomic base's virtual-joint velocity in the WORLD frame (verified
     on the 2025 radio demos); this fork rotates it into the robot frame. We rotate it back.
  #4 resolution: CompatFullResWrapper renders RGB at the data-collection resolution (head 720x720, wrists
     480x480), like Comet's RGBWrapper; the server then resize_with_pads to 224. CompatWrapper renders 224x224
     directly (fork default / 2025 RGBLowResWrapper).
Also injects ``steering_seed`` into every obs so the policy server can seed its sampling RNG per episode.
"""

import torch as th

from omnigibson.eval.utils.eval_utils import EVAL_CAMERA_ROLES, HEAD_RESOLUTION, WRIST_RESOLUTION
from omnigibson.eval.wrappers import DefaultWrapper

PROPRIO_DIM = 61  # fork eval/r1pro.yaml proprio_obs layout; base_qvel is proprio[0:3]
BASE_QVEL = slice(0, 3)
SEED_KEY = "steering_seed"


class CompatWrapper(DefaultWrapper):
    """224x224 RGB (fork default) + 2025-compatible proprio + per-episode seed."""

    # When False, leave base_qvel in the fork's robot frame (for A/B tests).
    world_frame_base_qvel = True

    def __init__(self, env):
        super().__init__(env=env)
        self.rollout_seed = 0

    def _fix_obs(self, obs, env_idx):
        robot = self.env.scenes[env_idx].robots[0]
        proprio = obs[robot.name]["proprio"]
        assert proprio.shape[-1] == PROPRIO_DIM, f"expected {PROPRIO_DIM}-d proprio, got {tuple(proprio.shape)}"
        if self.world_frame_base_qvel:
            # Inverse of BaseRobot._get_base_qvel_for_proprioception: robot frame -> world frame, using the same
            # yaw source (6-DoF base virtual joints, index 5 = rz).
            yaw = robot.get_joint_positions()[robot.base_idx][5].to(proprio.device)
            c, s = th.cos(yaw), th.sin(yaw)
            vx, vy, wz = proprio[BASE_QVEL].clone()
            proprio = proprio.clone()
            proprio[BASE_QVEL] = th.stack([c * vx - s * vy, s * vx + c * vy, wz])
            obs[robot.name]["proprio"] = proprio
        obs[SEED_KEY] = th.tensor([int(self.rollout_seed)], dtype=th.int64)
        return obs

    def _fix_list(self, obs_list, env_indices=None):
        env_indices = range(len(obs_list)) if env_indices is None else [int(i) for i in env_indices]
        return [self._fix_obs(o, i) for o, i in zip(obs_list, env_indices)]

    def step(self, action, n_render_iterations=1):
        obs_list, rew, term, trunc, info = self.env.step(action, n_render_iterations=n_render_iterations)
        return self._fix_list(obs_list), rew, term, trunc, info

    def reset(self, **kwargs):
        obs_list, info = self.env.reset(**kwargs)
        return self._fix_list(obs_list, kwargs.get("env_indices")), info

    def get_obs(self, env_indices=None):
        obs_list, info = self.env.get_obs(env_indices=env_indices)
        return self._fix_list(obs_list, env_indices), info


class CompatFullResWrapper(CompatWrapper):
    """Same as CompatWrapper, but RGB rendered at head 720x720 / wrists 480x480 (Comet's recommended setup)."""

    @classmethod
    def camera_spec(cls) -> dict:
        resolution = {role: (HEAD_RESOLUTION if role == "head" else WRIST_RESOLUTION) for role in EVAL_CAMERA_ROLES}
        return {"modalities": ["rgb"], "resolution": resolution}
