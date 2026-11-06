"""Video gallery for every rollout in one or more run dirs (keyframe strip + video + closest-finger distance).

python steering/make_gallery.py --out <dir> --title "..." <run_dir> [<run_dir> ...]
"""

import argparse
import csv
import html
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_report import keyframe_strip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Rollout gallery")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = [
        "<!doctype html><meta charset=utf-8><title>Rollout gallery</title><style>body{font:15px system-ui;"
        "max-width:1400px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}.case{background:#fff;"
        "border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0}img{max-width:100%}"
        "video{width:520px}</style>",
        f"<h1>{html.escape(args.title)}</h1><p>Videos: left column = wrist cameras, right = head camera. "
        "Right-click a video to change playback speed.</p>",
    ]
    for run in args.runs:
        p.append(f"<h2>{html.escape(os.path.basename(run))}</h2>")
        for r in csv.DictReader(open(os.path.join(run, "rollouts.csv"))):
            stem = f"{os.path.basename(run)}_{r['instance_id']}_{r['seed']}"
            shutil.copy(r["video_path"], os.path.join(args.out, stem + ".mp4"))
            keyframe_strip(r["video_path"], os.path.join(args.out, stem + "_frames.jpg"))
            ok = r["success"] == "1"
            extra = ""
            if r.get("min_finger_button_dist"):
                extra = f", closest finger to button {100 * float(r['min_finger_button_dist']):.0f} cm"
            p.append(
                f"<div class=case><b>setup {r['instance_id']}, seed {r['seed']}: "
                f"{'success at step ' + r['steps'] if ok else 'timeout'}</b>{extra}<br>"
                f"<img src='{stem}_frames.jpg'><br><video controls preload=metadata src='{stem}.mp4'></video></div>"
            )
    with open(os.path.join(args.out, "all.html"), "w") as f:
        f.write("\n".join(p))
    print("wrote", os.path.join(args.out, "all.html"))


if __name__ == "__main__":
    main()
