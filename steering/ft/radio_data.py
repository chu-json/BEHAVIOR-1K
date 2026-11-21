"""Dataset + transforms for fine-tuning pi0.5 on the preprocessed radio demos (steering/ft/preprocess.py output).

Sample (before openpi transforms):
    observation/egocentric_camera, observation/wrist_image_left, observation/wrist_image_right: (224, 224, 3) uint8
    observation/state: (23,) float32, action order (see preprocess.py)
    actions: (H, 23) float32 absolute, frames t..t+H-1 (indices past the episode end clamp to the last frame, like
             LeRobot's delta_timestamps padding that openpi uses)
    prompt: str from `prompt_fn(ctx)`

Augmentation hooks (all per-sample, all default to no-op so the baseline is Comet-like):
    prompt_fn(ctx) -> str              ctx = SampleContext (episode, frame, rng, state, seg lookup)
    image_fn(images, ctx) -> images    images = dict cam -> uint8 (224,224,3); ctx.seg(cam) gives radio/button masks
                                       and boxes for this frame (needs the "seg" preprocessing part)
See steering/ft/augment.py for examples (outline around the radio, prompt paraphrases).
"""

import dataclasses
import glob
import os
from typing import Callable

import cv2
import numpy as np

DATA = "/home/behavior/ext/data/radio_ft"
CAMS = ("head", "left_wrist", "right_wrist")
CAM_KEYS = {
    "head": "observation/egocentric_camera",
    "left_wrist": "observation/wrist_image_left",
    "right_wrist": "observation/wrist_image_right",
}
RADIO_PROMPT = "Turn on the radio receiver that's on the table in the living room."  # openpi-comet task_mapping.json

# Delta-action mask in the 23-d action order: base vel absolute, torso + left arm delta, left gripper absolute,
# right arm delta, right gripper absolute. State uses the same order, so openpi's DeltaActions applies directly.
DELTA_MASK = (-3, 4 + 7, -1, 7, -1)

# fork 61-d eval proprio (OmniGibson/omnigibson/eval/utils/eval_utils.py, R1Pro) -> 23-d state in action order
P61 = dict(base_qvel=np.s_[0:3], trunk=np.s_[53:57], arm_left=np.s_[3:10], grip_left=np.s_[24:26],
           arm_right=np.s_[28:35], grip_right=np.s_[49:51])  # fmt: skip


def state23_from_61(p):
    """23-d state from the fork's 61-d proprio (base_qvel there is already robot-frame)."""
    p = np.asarray(p, dtype=np.float64)
    return np.concatenate(
        [
            p[..., P61["base_qvel"]],
            p[..., P61["trunk"]],
            p[..., P61["arm_left"]],
            p[..., P61["grip_left"]].sum(-1, keepdims=True),
            p[..., P61["arm_right"]],
            p[..., P61["grip_right"]].sum(-1, keepdims=True),
        ],
        axis=-1,
    ).astype(np.float32)


def to_state23(s):
    s = np.asarray(s)
    if s.shape[-1] == 23:
        return s.astype(np.float32)
    if s.shape[-1] == 61:
        return state23_from_61(s)
    if s.shape[-1] == 256:
        from preprocess import state23_from_256

        return state23_from_256(s)
    raise ValueError(f"unexpected state size {s.shape}")


@dataclasses.dataclass(frozen=True)
class RadioInputs:
    """openpi data transform (train and serve): state -> 23-d action-order state, images -> model image dict.

    Accepts observation/state as 23-d (training), 61-d (fork eval proprio, what comet_server's B1KPolicyWrapper
    sends) or 256-d (2025 proprio). Camera naming matches openpi-comet B1kInputs (head -> base_0_rgb).
    """

    def __call__(self, data):
        out = {
            "state": to_state23(data["observation/state"]),
            "image": {
                "base_0_rgb": _hwc_uint8(data["observation/egocentric_camera"]),
                "left_wrist_0_rgb": _hwc_uint8(data["observation/wrist_image_left"]),
                "right_wrist_0_rgb": _hwc_uint8(data["observation/wrist_image_right"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class RadioOutputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"][:, :23])}


def _hwc_uint8(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)
    return img


@dataclasses.dataclass
class SampleContext:
    episode: int
    frame: int
    n_frames: int
    rng: np.random.Generator
    state: np.ndarray
    _ds: "RadioDataset"

    def seg(self, cam):
        """dict(radio_mask, button_mask: (224,224) bool, radio_box, button_box: (4,) x0,y0,x1,y1 or -1)."""
        return self._ds.seg(self.episode, cam, self.frame)


def default_prompt(ctx):
    return RADIO_PROMPT


class RadioDataset:
    """Map-style dataset over every frame of the preprocessed episodes. Light to pickle (lazy per-worker mmaps)."""

    def __init__(
        self,
        root: str = DATA,
        action_horizon: int = 32,
        episodes: list[int] | None = None,
        exclude: tuple[int, ...] = (),
        prompt_fn: Callable = default_prompt,
        image_fn: Callable | None = None,
        seed: int = 0,
    ):
        self.root = root
        self.H = action_horizon
        self.prompt_fn = prompt_fn
        self.image_fn = image_fn
        self.seed = seed
        eps = sorted(int(os.path.basename(f)[3:11]) for f in glob.glob(f"{root}/ep_????????.npz"))
        if episodes is not None:
            eps = [e for e in eps if e in set(episodes)]
        eps = [e for e in eps if e not in set(exclude)]
        assert eps, f"no preprocessed episodes in {root}"
        self.episodes = eps
        self.lengths = []
        for e in eps:
            with np.load(f"{root}/ep_{e:08d}.npz") as z:
                self.lengths.append(len(z["action"]))
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)])
        self._low = {}
        self._jpg = {}
        self._seg = {}
        self._calls = 0

    def __len__(self):
        return int(self.starts[-1])

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_low"], d["_jpg"], d["_seg"] = {}, {}, {}
        return d

    def locate(self, index):
        i = int(np.searchsorted(self.starts, index, side="right") - 1)
        return i, int(index - self.starts[i])

    def lowdim(self, e):
        if e not in self._low:
            with np.load(f"{self.root}/ep_{e:08d}.npz") as z:
                self._low[e] = {"state": z["state"], "action": z["action"]}
        return self._low[e]

    def image(self, e, cam, t):
        key = (e, cam)
        if key not in self._jpg:
            self._jpg[key] = (
                np.load(f"{self.root}/ep_{e:08d}_{cam}.jpg.npy", mmap_mode="r"),
                np.load(f"{self.root}/ep_{e:08d}_{cam}.off.npy"),
            )
            if len(self._jpg) > 3 * 64:  # bound open mmaps per worker
                self._jpg.pop(next(iter(self._jpg)))
        buf, off = self._jpg[key]
        img = cv2.imdecode(np.asarray(buf[off[t] : off[t + 1]]), cv2.IMREAD_COLOR)
        return img[..., ::-1].copy()

    def seg(self, e, cam, t):
        if e not in self._seg:
            if len(self._seg) > 16:
                self._seg.pop(next(iter(self._seg)))
            with np.load(f"{self.root}/ep_{e:08d}_seg.npz") as z:
                self._seg[e] = {k: z[k] for k in z.files}
        z = self._seg[e]
        return {
            "radio_mask": np.unpackbits(z[f"{cam}_radio_mask"][t], axis=-1).astype(bool),
            "button_mask": np.unpackbits(z[f"{cam}_button_mask"][t], axis=-1).astype(bool),
            "radio_box": z[f"{cam}_radio_box"][t],
            "button_box": z[f"{cam}_button_box"][t],
        }

    def __getitem__(self, index):
        index = int(index)
        i, t = self.locate(index)
        e, n = self.episodes[i], self.lengths[i]
        low = self.lowdim(e)
        idx = np.minimum(np.arange(t, t + self.H), n - 1)
        rng = np.random.default_rng([self.seed, index, self._calls])
        self._calls += 1
        ctx = SampleContext(e, t, n, rng, low["state"][t], self)
        images = {cam: self.image(e, cam, t) for cam in CAMS}
        if self.image_fn is not None:
            images = self.image_fn(images, ctx)
        out = {CAM_KEYS[c]: images[c] for c in CAMS}
        out["observation/state"] = low["state"][t].copy()
        out["actions"] = low["action"][idx].copy()
        out["prompt"] = self.prompt_fn(ctx)
        return out


def compute_norm_stats(root=DATA, action_horizon=32, stride=2, min_range=1e-3, episodes=None):
    """openpi NormStats for state (23-d) and DELTA actions (DELTA_MASK) over every `stride`-th frame's chunk.

    Matches what openpi's compute_norm_stats would see after the data transforms (RadioInputs + DeltaActions), but
    with exact quantiles. Dims whose q01..q99 range is below `min_range` are widened symmetrically so the quantile
    normalization (x - q01) / (q99 - q01 + 1e-6) cannot blow up.
    """
    from openpi import transforms as T

    mask = np.asarray(T.make_bool_mask(*DELTA_MASK))
    ds = RadioDataset(root, action_horizon, episodes=episodes)
    states, acts = [], []
    for e, n in zip(ds.episodes, ds.lengths):
        low = ds.lowdim(e)
        s, a = low["state"], low["action"]
        states.append(s)
        ts = np.arange(0, n, stride)
        idx = np.minimum(ts[:, None] + np.arange(action_horizon)[None], n - 1)
        chunk = a[idx] - np.where(mask, s[ts], 0)[:, None, :]
        acts.append(chunk.reshape(-1, 23))
    out = {}
    for k, x in (("state", np.concatenate(states)), ("actions", np.concatenate(acts))):
        x = x.astype(np.float64)
        q01, q99 = np.quantile(x, 0.01, axis=0), np.quantile(x, 0.99, axis=0)
        mid, half = (q01 + q99) / 2, np.maximum((q99 - q01) / 2, min_range / 2)
        out[k] = T.NormStats(mean=x.mean(0), std=x.std(0), q01=mid - half, q99=mid + half)
    return out
