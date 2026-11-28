"""Record rollouts as fine-tuning demos in the steering/ft/radio_data.py format (same layout as the preprocessed
2025 radio demos in /home/behavior/ext/data/radio_ft), e.g. scripted PressAssist rollouts in perturbed scenes.

Per episode <out>/ep_<id>.npz with
    state (T, 23)  action order (state23_from_61 of the fork's 61-d proprio; base_qvel robot frame)
    action (T, 23) the absolute action actually applied at each step
    proprio61 (T, 61), success, instance, seed
and per camera <out>/ep_<id>_<cam>.jpg.npy / .off.npy: JPEG q95 bytes of every frame after openpi_client
resize_with_pad(.., 224, 224), concatenated, with frame offsets (exactly what preprocess.py writes).
Used via run_rollouts.py --record-demos DIR [--record-only-success].
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/behavior/ext/openpi-comet/packages/openpi-client/src")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft"))

CAM_OBS = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
}
PROPRIO = "robot_r1::proprio"
RES, JPEG_QUALITY = 224, 95


def _np(v):
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    v = np.asarray(v)
    return v[0] if v.ndim and v.shape[0] == 1 else v


class DemoRecorder:
    def __init__(self, out_dir, only_success=False, id_base=900000):
        from openpi_client.image_tools import resize_with_pad

        self.resize = resize_with_pad
        self.out, self.only_success, self.id_base = out_dir, only_success, id_base
        os.makedirs(out_dir, exist_ok=True)
        self.saved = 0
        self.reset()

    def reset(self):
        self.p61, self.act, self.jpg = [], [], {c: [] for c in CAM_OBS}

    def wrap(self, forward):
        def recorded_forward(obs, *args, **kwargs):
            action = forward(obs, *args, **kwargs)
            self.p61.append(_np(obs[PROPRIO]).astype(np.float32))
            self.act.append(_np(action).astype(np.float32).reshape(-1)[:23])
            for cam, key in CAM_OBS.items():
                img = _np(obs[key])[..., :3].astype(np.uint8)
                img = self.resize(img[None], RES, RES)[0]
                ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                self.jpg[cam].append(enc.reshape(-1))
            return action

        return recorded_forward

    def finish(self, instance, seed, success):
        """Write the episode (if kept); returns the episode id or ''."""
        from radio_data import state23_from_61

        n = len(self.act)
        if n == 0 or (self.only_success and not success):
            self.reset()
            return ""
        e = self.id_base + int(instance) * 1000 + int(seed)
        p = np.stack(self.p61)
        np.savez(f"{self.out}/ep_{e:08d}.npz", state=state23_from_61(p), action=np.stack(self.act), proprio61=p,
                 success=int(success), instance=int(instance), seed=int(seed))  # fmt: skip
        for cam, bufs in self.jpg.items():
            off = np.cumsum([0] + [len(b) for b in bufs]).astype(np.int64)
            np.save(f"{self.out}/ep_{e:08d}_{cam}.jpg.npy", np.concatenate(bufs))
            np.save(f"{self.out}/ep_{e:08d}_{cam}.off.npy", off)
        self.saved += 1
        self.reset()
        return e
