"""Preprocess the 2025 BEHAVIOR radio demos (task-0000, 200 episodes) into a compact, fast-to-read training form.

Per episode `ep_<idx>` under --out (default /home/behavior/ext/data/radio_ft):
  ep_<idx>.npz          lowdim (part "lowdim"):
      state      (T, 23) float32  policy state in ACTION order: base vel (robot frame) 0:3, torso 3:7, left arm 7:14,
                                  left gripper width 14, right arm 15:22, right gripper width 22
      action     (T, 23) float32  absolute 23-d action as recorded (base vel 0:3 is robot-frame velocity command)
      proprio256 (T, 256) float32 raw 2025 proprio vector (for later features)
      timestamp  (T,), task_info (T, 44)
  ep_<idx>_<cam>.jpg.npy + ep_<idx>_<cam>.off.npy   (part "rgb", cam in head/left_wrist/right_wrist)
      JPEG (q95) bytes of every frame after openpi_client resize_with_pad(.., 224, 224) (PIL bilinear: exactly what
      Comet's B1KPolicyWrapper does to full-res sim images at serve time), concatenated; offsets (T+1,) int64.
  ep_<idx>_seg.npz      (part "seg") radio masks/boxes decoded from seg_instance_id videos:
      <cam>_radio_mask / <cam>_button_mask  packbits((T, 224, 224) bool)  [radio = every instance whose object name
                                            contains "radio" (body + toggle button); button = the togglebutton link]
      <cam>_radio_box / <cam>_button_box    (T, 4) int16 x0,y0,x1,y1 in 224 px (inclusive), -1 if not visible
      <cam>_radio_px  / <cam>_button_px     (T,) int32 visible pixel count at full resolution
      (masks get a 3x3 morphological opening at full resolution: video compression bleeds chroma at other objects'
      edges into isolated radio-colored pixels)

The seg videos store each instance id as a palette color (BEHAVIOR v3.7 obs_utils.generate_yuv_palette over the
camera's unique_ins_ids, written as rgb24). Decoding = nearest palette color; the palette is a regular 3-D grid, so
per-channel rounding gives the nearest grid point (exact fallback for points past the truncated end of the grid).

Run with openpi-comet's venv (PyAV, PIL, cv2, openpi_client):
    /home/behavior/ext/openpi-comet/.venv/bin/python steering/ft/preprocess.py --parts lowdim,rgb --workers 5
    /home/behavior/ext/openpi-comet/.venv/bin/python steering/ft/preprocess.py --parts seg --workers 5
Idempotent: skips outputs that already exist.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from openpi_client.image_tools import resize_with_pad

DEMOS = "/home/behavior/ext/data/2025-challenge-demos"
OUT = "/home/behavior/ext/data/radio_ft"
CAMS = ("head", "left_wrist", "right_wrist")
ROBOT_CAMERA_NAMES = {
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
    "head": "robot_r1::robot_r1:zed_link:Camera:0",
}
RES = 224
JPEG_QUALITY = 95
KERNEL = np.ones((3, 3), np.uint8)

# 2025 (v3.7.2) 256-d proprio indices
P256 = dict(
    base_qpos=np.s_[244:247],  # x, y, yaw of the holonomic base virtual joints
    base_qvel=np.s_[253:256],  # joint-frame (world) velocity
    trunk_qpos=np.s_[236:240],
    arm_left_qpos=np.s_[158:165],
    gripper_left_qpos=np.s_[193:195],
    arm_right_qpos=np.s_[197:204],
    gripper_right_qpos=np.s_[232:234],
)


def base_vel_to_robot_frame(yaw, v):
    """Same as fork eval_utils.holonomic_base_qvel_to_robot_frame (fork proprio reports robot-frame base_qvel)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], axis=-1)


def state23_from_256(p):
    """23-d policy state in action order from the 2025 256-d proprio (base vel rotated into the robot frame)."""
    bv = base_vel_to_robot_frame(p[..., P256["base_qpos"]][..., 2], p[..., P256["base_qvel"]])
    return np.concatenate(
        [
            bv,
            p[..., P256["trunk_qpos"]],
            p[..., P256["arm_left_qpos"]],
            p[..., P256["gripper_left_qpos"]].sum(-1, keepdims=True),
            p[..., P256["arm_right_qpos"]],
            p[..., P256["gripper_right_qpos"]].sum(-1, keepdims=True),
        ],
        axis=-1,
    ).astype(np.float32)


def generate_yuv_palette(num_ids):
    """Copied from BEHAVIOR v3.7.2 omnigibson/learning/utils/obs_utils.py (the encoder used for the 2025 videos)."""
    Y_vals = np.linspace(16, 235, int(np.ceil(num_ids ** (1 / 3))))
    U_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1 / 3))))
    V_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1 / 3))))
    palette = []
    for y in Y_vals:
        for u in U_vals:
            for v in V_vals:
                palette.append([y, u, v])
                if len(palette) >= num_ids:
                    return np.array(palette, dtype=np.uint8)
    return np.array(palette[:num_ids], dtype=np.uint8)


class PaletteDecoder:
    """rgb24 seg frame -> palette index (nearest palette color), vectorized via per-channel LUTs."""

    def __init__(self, num_ids):
        self.N = num_ids
        self.n = int(np.ceil(num_ids ** (1 / 3)))
        self.palette = generate_yuv_palette(num_ids).astype(np.float32)
        grids = [
            np.linspace(16, 235, self.n).astype(np.uint8),
            np.linspace(16, 240, self.n).astype(np.uint8),
            np.linspace(16, 240, self.n).astype(np.uint8),
        ]
        vals = np.arange(256)
        self.luts = [np.abs(vals[:, None] - g[None, :].astype(np.int32)).argmin(1).astype(np.int32) for g in grids]

    def __call__(self, rgb):
        n = self.n
        flat = self.luts[0][rgb[..., 0]] * n * n + self.luts[1][rgb[..., 1]] * n + self.luts[2][rgb[..., 2]]
        bad = flat >= self.N
        if bad.any():
            px = rgb[bad].astype(np.float32)
            d = ((px[:, None, :] - self.palette[None]) ** 2).sum(-1)
            flat[bad] = d.argmin(1)
        return flat


def open_video(path):
    c = av.open(path)
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    return c, s


def episode_ids(demos):
    return sorted(int(os.path.basename(f)[8:16]) for f in glob.glob(f"{demos}/data/task-0000/episode_*.parquet"))


def do_lowdim(ep, demos, out):
    dst = f"{out}/ep_{ep:08d}.npz"
    if os.path.exists(dst):
        return
    t = pq.read_table(f"{demos}/data/task-0000/episode_{ep:08d}.parquet")
    p = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
    a = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
    ti = np.stack(t["observation.task_info"].to_numpy(zero_copy_only=False)).astype(np.float32)
    ts = np.asarray(t["timestamp"].to_numpy(), dtype=np.float64)
    assert (np.asarray(t["episode_index"].to_numpy()) == ep).all()
    tmp = dst + ".tmp.npz"
    np.savez(tmp, state=state23_from_256(p), action=a, proprio256=p.astype(np.float32), timestamp=ts, task_info=ti)
    os.replace(tmp, dst)


def do_rgb(ep, demos, out, n_frames):
    for cam in CAMS:
        dst_j, dst_o = f"{out}/ep_{ep:08d}_{cam}.jpg.npy", f"{out}/ep_{ep:08d}_{cam}.off.npy"
        if os.path.exists(dst_j) and os.path.exists(dst_o):
            continue
        src = f"{demos}/videos/task-0000/observation.images.rgb.{cam}/episode_{ep:08d}.mp4"
        c, s = open_video(src)
        bufs = []
        for frame in c.decode(s):
            img = resize_with_pad(frame.to_ndarray(format="rgb24")[None], RES, RES)[0]
            ok, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            assert ok
            bufs.append(enc.reshape(-1))
        c.close()
        assert len(bufs) == n_frames, f"ep {ep} {cam}: {len(bufs)} video frames vs {n_frames} parquet rows"
        off = np.zeros(len(bufs) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(b) for b in bufs])
        np.save(dst_j + ".tmp.npy", np.concatenate(bufs))
        np.save(dst_o + ".tmp.npy", off)
        os.replace(dst_j + ".tmp.npy", dst_j)
        os.replace(dst_o + ".tmp.npy", dst_o)


def box_of(mask):
    ys, xs = np.nonzero(mask.any(1)), np.nonzero(mask.any(0))
    if len(ys[0]) == 0:
        return (-1, -1, -1, -1)
    return (xs[0][0], ys[0][0], xs[0][-1], ys[0][-1])


def do_seg(ep, demos, out, n_frames):
    dst = f"{out}/ep_{ep:08d}_seg.npz"
    if os.path.exists(dst):
        return
    meta = json.load(open(f"{demos}/meta/episodes/task-0000/episode_{ep:08d}.json"))
    mapping = {int(k): v for k, v in json.loads(meta["ins_id_mapping"]).items()}
    res = {}
    for cam in CAMS:
        ids = np.asarray(meta[f"{ROBOT_CAMERA_NAMES[cam]}::unique_ins_ids"], dtype=np.int64)
        name = [mapping.get(int(i), "") for i in ids]
        obj = [n.split("/")[3] if n.count("/") >= 3 else "" for n in name]
        is_radio = np.array(["radio" in o for o in obj])
        is_button = np.array(["togglebutton" in n and "radio" in o for n, o in zip(name, obj)])
        dec = PaletteDecoder(len(ids))
        src = f"{demos}/videos/task-0000/observation.images.seg_instance_id.{cam}/episode_{ep:08d}.mp4"
        c, s = open_video(src)
        masks = {"radio": [], "button": []}
        px = {"radio": [], "button": []}
        for frame in c.decode(s):
            idx = dec(frame.to_ndarray(format="rgb24"))
            for k, sel in (("radio", is_radio), ("button", is_button)):
                m = sel[idx] if sel.any() else np.zeros(idx.shape, bool)
                # 3x3 opening removes isolated false positives from chroma bleed at other objects' edges
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, KERNEL).astype(bool)
                px[k].append(int(m.sum()))
                # area fraction >= 0.25 keeps thin/small parts (the button) visible at 224
                small = cv2.resize(m.astype(np.float32), (RES, RES), interpolation=cv2.INTER_AREA) >= 0.25
                masks[k].append(small)
        c.close()
        assert len(masks["radio"]) == n_frames, f"ep {ep} seg {cam}: {len(masks['radio'])} vs {n_frames}"
        for k in ("radio", "button"):
            m = np.stack(masks[k])
            res[f"{cam}_{k}_mask"] = np.packbits(m, axis=-1)
            res[f"{cam}_{k}_box"] = np.array([box_of(f) for f in m], dtype=np.int16)
            res[f"{cam}_{k}_px"] = np.array(px[k], dtype=np.int32)
        res[f"{cam}_radio_ids"] = ids[is_radio]
    tmp = dst + ".tmp.npz"
    np.savez_compressed(tmp, **res)
    os.replace(tmp, dst)


def work(args):
    ep, demos, out, parts = args
    t0 = time.time()
    try:
        n = pq.read_metadata(f"{demos}/data/task-0000/episode_{ep:08d}.parquet").num_rows
        if "lowdim" in parts:
            do_lowdim(ep, demos, out)
        if "rgb" in parts:
            do_rgb(ep, demos, out, n)
        if "seg" in parts:
            do_seg(ep, demos, out, n)
        return ep, n, time.time() - t0, None
    except Exception as e:  # keep going; report at the end
        return ep, 0, time.time() - t0, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=DEMOS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--parts", default="lowdim,rgb", help="comma list of lowdim,rgb,seg")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", default=None, help="comma list of episode indices (default: all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    parts = set(args.parts.split(","))
    eps = [int(e) for e in args.episodes.split(",")] if args.episodes else episode_ids(args.demos)
    cv2.setNumThreads(1)
    t0 = time.time()
    errs = []
    with mp.Pool(args.workers) as pool:
        for i, (ep, n, dt, err) in enumerate(pool.imap_unordered(work, [(e, args.demos, args.out, parts) for e in eps])):
            if err:
                errs.append((ep, err))
            print(f"[{i + 1}/{len(eps)}] ep {ep} {n} frames {dt:.0f}s {err or ''} (elapsed {time.time() - t0:.0f}s)",
                  flush=True)
    print(f"DONE {len(eps) - len(errs)}/{len(eps)} ok; errors: {errs}", flush=True)


if __name__ == "__main__":
    main()
