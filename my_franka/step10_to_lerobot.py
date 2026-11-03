"""
Step 10: convert the wrist-camera demos (step 9 HDF5 files) into one LeRobot dataset, the format that
`lerobot-train` (and the BEHAVIOR challenge baselines) consume. Only successful episodes are converted.

    python my_franka/step10_to_lerobot.py                 # all successful episodes
    python my_franka/step10_to_lerobot.py --max 50        # small dataset for a pipeline smoke test

Output: my_franka/data/lerobot/<name>/ (parquet data with embedded PNG frames + normalization stats)
Use a subset at training time with --dataset.episodes='[0,1,...]' instead of converting again.
"""
import argparse
import glob
import os
import shutil

import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "vision", "demos_w*_c*.hdf5")
TASK = "pick up the apple"
FEATURES = {
    # Stored as per-frame PNG images rather than MP4/AV1 video: random-access decoding of video frames was the
    # training bottleneck (0.3 s/batch, ~4x slower than the model update); a 128x128 PNG decodes in ~0.2 ms.
    "observation.images.wrist": {"dtype": "image", "shape": [128, 128, 3], "names": ["height", "width", "channels"]},
    "observation.state": {
        "dtype": "float32", "shape": [7],
        "names": ["eef_x", "eef_y", "eef_z", "finger_l", "finger_r", "prev_gripper_cmd", "is_grasping"],
    },
    "action": {"dtype": "float32", "shape": [4], "names": ["dx", "dy", "dz", "gripper"]},
}


def open_by_another_process(path):
    """True if any running process has @path open (i.e. a collector is still writing it)."""
    target = os.path.realpath(path)
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.path.realpath(f"/proc/{pid}/fd/{fd}") == target:
                    return True
        except OSError:
            continue
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="franka_apple_wrist")
    parser.add_argument("--max", type=int, default=None, help="stop after this many episodes")
    parser.add_argument("--src", default=SRC, help="glob of HDF5 files to convert")
    args = parser.parse_args()

    root = os.path.join(HERE, "data", "lerobot", args.name)
    if os.path.exists(root):
        shutil.rmtree(root)  # always rebuild from the HDF5 source of truth
    ds = LeRobotDataset.create(repo_id=f"local/{args.name}", fps=30, features=FEATURES, root=root,
                               robot_type="franka", use_videos=False, image_writer_threads=8)
    n = n_frames = skipped = 0
    for path in sorted(glob.glob(args.src)):
        if open_by_another_process(path):
            print(f"skipping {path} (still being written)")
            continue
        try:
            f = h5py.File(path, "r")
        except OSError:
            print(f"skipping {path} (still being written?)")
            continue
        with f:
            for key in sorted(f.keys()):
                g = f[key]
                if not g.attrs.get("success", False):
                    skipped += 1
                    continue
                images, state, action = g["images"][:], g["state"][:], g["action"][:]
                for t in range(len(action)):
                    ds.add_frame({"observation.images.wrist": images[t], "observation.state": state[t],
                                  "action": action[t], "task": TASK})
                ds.save_episode()
                n, n_frames = n + 1, n_frames + len(action)
                if n % 50 == 0:
                    print(f"{n} episodes, {n_frames} frames", flush=True)
                if args.max and n >= args.max:
                    break
        if args.max and n >= args.max:
            break
    ds.finalize()
    print(f"wrote {n} successful episodes ({n_frames} frames, {skipped} failed episodes skipped) to {root}")


if __name__ == "__main__":
    main()
