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
TARGET_KEY = "steering_target"  # privileged: radio button position in the robot base frame (best-of-N critic)
RADIO = "radio_receiver.n.01_1"


class CompatWrapper(DefaultWrapper):
    """224x224 RGB (fork default) + 2025-compatible proprio + per-episode seed."""

    # When False, leave base_qvel in the fork's robot frame (for A/B tests).
    world_frame_base_qvel = True

    def __init__(self, env):
        super().__init__(env=env)
        self.rollout_seed = 0
        self.inject_target = False  # set by run_rollouts.py --inject-target

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
        if self.inject_target:
            import omnigibson.utils.transform_utils as T
            from omnigibson.object_states import ToggledOn

            entity = self.env.task.object_scopes[env_idx][RADIO]
            radio = getattr(entity, "wrapped_obj", entity)
            button = radio.states[ToggledOn].marker.get_position_orientation()[0]
            rel = T.relative_pose_transform(button, th.tensor([0.0, 0.0, 0.0, 1.0]), *robot.get_position_orientation())[
                0
            ]
            obs[TARGET_KEY] = rel.float().cpu()
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


class SteerVisionWrapper(CompatFullResWrapper):
    """Full-res RGB + linear depth; the radio's 2D box in every camera is the projection of its 3D bounding box from
    the simulator (a stand-in for an object detector; seg_semantic segfaults this Isaac build, NOTES 2026-11-19).
    Depth is used client-side only and stripped before the policy sees obs.

    overlay (run_rollouts.py --overlay):
      "tint": blend the radio's box toward red in all 3 cameras (visual prompting: point the policy at the target)
      "dim":  darken everything outside the radio's box (clutter removal)
    depth_target (--depth-target): radio position (robot base frame) from the head camera's depth inside the box
      (nearest-surface pixels) -> `steering_target` for the best-of-N critic, instead of the true button position.
    """

    TINT = (255.0, 0.0, 0.0)
    PAD_PX = 4

    @classmethod
    def camera_spec(cls) -> dict:
        import os

        spec = super().camera_spec()
        mods = os.environ.get("STEER_VISION_MODALITIES", "rgb,depth_linear").split(",")  # for debugging
        return {**spec, "modalities": mods}

    def __init__(self, env):
        super().__init__(env=env)
        self.overlay = None
        self.depth_target = False
        self.reset_stats()

    def reset_stats(self):
        self.est = None  # last good estimate
        self.est_err = []  # |estimate - true button| per estimate, m
        self.est_n = 0
        self.box_frac = []  # head-camera box area fraction per step

    def _radio(self, env_idx):
        entity = self.env.task.object_scopes[env_idx][RADIO]
        return getattr(entity, "wrapped_obj", entity)

    @staticmethod
    def _cam_frame(cam):
        import omnigibson.utils.transform_utils as T

        pos, quat = cam.get_position_orientation()
        return pos.cpu().float(), T.quat2mat(quat.cpu()).float()

    def _box(self, cam, h, w, corners):
        """Pixel box (u0, v0, u1, v1) of world-frame corners in camera `cam`, or None if not in view."""
        t, R = self._cam_frame(cam)
        pc = (corners - t) @ R  # world -> USD camera frame (+x right, +y up, looking down -z)
        z = -pc[:, 2]
        if bool((z <= 0.05).any()):
            return None
        f = w * cam.focal_length / cam.horizontal_aperture
        u = f * pc[:, 0] / z + w / 2
        v = -f * pc[:, 1] / z + h / 2
        u0, u1 = int(u.min()) - self.PAD_PX, int(u.max()) + self.PAD_PX + 1
        v0, v1 = int(v.min()) - self.PAD_PX, int(v.max()) + self.PAD_PX + 1
        u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, w), min(v1, h)
        return (u0, v0, u1, v1) if u1 > u0 and v1 > v0 else None

    def _fix_obs(self, obs, env_idx):
        robot = self.env.scenes[env_idx].robots[0]
        lo, hi = self._radio(env_idx).aabb
        lo, hi = lo.cpu().float(), hi.cpu().float()
        corners = th.stack(
            [th.stack([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        head = None
        for name, d in obs[robot.name].items():
            if not isinstance(d, dict) or "rgb" not in d:
                continue
            depth = d.pop("depth_linear", None)
            rgb = d["rgb"]
            h, w = rgb.shape[:2]
            box = self._box(robot.sensors[name], h, w, corners) if name in robot.sensors else None
            if "zed_link" in name:
                head = (name, box, depth)
                self.box_frac.append(0.0 if box is None else (box[2] - box[0]) * (box[3] - box[1]) / (h * w))
            if self.overlay is None or box is None and self.overlay == "tint":
                continue
            img = rgb[..., :3].float()
            mask = th.zeros((h, w), dtype=th.bool, device=img.device)
            if box is not None:
                mask[box[1] : box[3], box[0] : box[2]] = True
            if self.overlay == "tint":
                img = th.where(mask[..., None], 0.5 * img + 0.5 * th.tensor(self.TINT, device=img.device), img)
            else:
                img = th.where(mask[..., None], img, 0.35 * img)
            rgb = rgb.clone()
            rgb[..., :3] = img.round().clamp(0, 255).to(rgb.dtype)
            d["rgb"] = rgb
        obs = super()._fix_obs(obs, env_idx)  # base_qvel, seed, privileged target if inject_target
        if self.depth_target and head is not None:
            self._depth_target(obs, robot, env_idx, *head)
        return obs

    def _depth_target(self, obs, robot, env_idx, name, box, depth, min_px=30, slab_m=0.12):
        import omnigibson.utils.transform_utils as T
        from omnigibson.object_states import ToggledOn

        if depth is not None and box is not None:
            u0, v0, u1, v1 = box
            dz = depth[v0:v1, u0:u1].float().cpu()
            vv, uu = th.meshgrid(th.arange(v0, v1).float(), th.arange(u0, u1).float(), indexing="ij")
            ok = th.isfinite(dz) & (dz > 0.05) & (dz < 5.0)
            if int(ok.sum()) >= min_px:
                near = th.quantile(dz[ok], 0.1)
                sel = ok & (dz < near + slab_m)  # foreground surface in the box = the radio (not the table behind)
                cam = robot.sensors[name]
                h, w = depth.shape[:2]
                f = w * cam.focal_length / cam.horizontal_aperture
                z, u, v = dz[sel], uu[sel], vv[sel]
                p = th.stack([(u + 0.5 - w / 2) / f * z, -(v + 0.5 - h / 2) / f * z, -z], dim=-1)
                t, R = self._cam_frame(cam)
                world = p @ R.T + t
                bpos, bquat = robot.get_position_orientation()
                base = (world - bpos.cpu().float()) @ T.quat2mat(bquat.cpu()).float()  # R^T (p - t), row vectors
                self.est = base.median(dim=0).values
                self.est_n += 1
                button = self._radio(env_idx).states[ToggledOn].marker.get_position_orientation()[0]
                true = T.relative_pose_transform(button, th.tensor([0.0, 0.0, 0.0, 1.0]), bpos, bquat)[0].cpu()
                self.est_err.append(float(th.linalg.norm(self.est - true.float())))
        if self.est is not None:
            obs[TARGET_KEY] = self.est.float()

    def summary(self):
        import statistics as st

        e = self.est_err
        return {
            "depth_est_n": self.est_n,
            "depth_est_err_med_m": round(st.median(e), 3) if e else "",
            "depth_est_err_last_m": round(e[-1], 3) if e else "",
            "head_box_frac_med": round(st.median(self.box_frac), 5) if self.box_frac else "",
        }
