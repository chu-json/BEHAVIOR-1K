"""Build a local HTML page to browse rollouts: stats, then one success and the failure cases grouped by type.

For each case: video, keyframe strip, and a per-step plot from the --trace CSV (closest finger-to-button distance,
radio height, grasp state, toggle). Writes <out>/index.html plus images; videos are copied next to it.

    python steering/make_report.py --run /home/behavior/ext/runs/val10_full_fixed --out /home/behavior/ext/report
"""

import argparse
import csv
import html
import os
import shutil

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TABLE_TOP_Z = 0.423
# Hand-written from watching the videos + traces (see steering/NOTES.md). instance -> (group, description)
CASES = {
    242: (
        "Success",
        "Typical success: drives to the table, grabs the radio (step ~970), lifts it and presses the "
        "button while holding it (step 1256).",
    ),
    295: (
        "Failure: never grasps",
        "The right hand hovers next to the radio from step ~860 until timeout and never "
        "closes on it. Closest finger stays ~11 cm from the button.",
    ),
    203: (
        "Failure: never grasps",
        "The right hand rests on/pushes the radio from step ~615 to timeout without "
        "ever closing a grasp, on the wrong side of the radio: closest finger stays ~15 cm from the button.",
    ),
    211: (
        "Failure: holds the radio, never presses",
        "Grabs the radio (step ~1180), carries it past the table edge "
        "and holds it below table height for ~2600 steps. Same failure in both batches.",
    ),
    181: (
        "Failure: holds the radio, never presses",
        "Near miss: right hand holds the radio, left hand keeps coming "
        "close to the button (6.9 cm at best, successes reach 5.7-5.9 cm) but never triggers it.",
    ),
    197: (
        "Failure: grasps, then gives up",
        "Grabs the radio, puts it back (step ~1500), one brief re-grasp, then "
        "retracts its arms and stands still looking at the radio until timeout.",
    ),
    139: ("Failure: grasps, then gives up", "Grabs the radio briefly (steps 1587-1952), releases it, and hovers."),
}
GROUP_ORDER = [
    "Success",
    "Failure: never grasps",
    "Failure: holds the radio, never presses",
    "Failure: grasps, then gives up",
]


def keyframe_strip(video, out_png, k=8):
    frames = [f.to_ndarray(format="rgb24") for f in av.open(video).decode(video=0)]
    idx = np.linspace(0, len(frames) - 1, k).astype(int)
    tiles = []
    for i in idx:
        t = cv2.resize(frames[i][:, 224:], (300, 300))  # head camera only (right 448x448 of the 448x672 frame)
        cv2.putText(t, f"step {i + 1}", (6, 22), 0, 0.7, (255, 255, 0), 2)
        tiles.append(t)
    cv2.imwrite(out_png, cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))


def trace_plot(trace_csv, out_png, success):
    t = list(csv.DictReader(open(trace_csv)))
    step = np.array([int(x["step"]) for x in t])
    d = np.array([min(float(x["finger_button_dist_left"]), float(x["finger_button_dist_right"])) for x in t])
    z = np.array([float(x["radio_z"]) for x in t])
    grasp = np.array([x["grasp_radio_left"] == "1" or x["grasp_radio_right"] == "1" for x in t])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True, constrained_layout=True)
    a1.plot(step, d * 100, color="#2a6fdb", lw=1.2)
    a1.axhline(5.9, color="#2a9d55", ls="--", lw=1)
    a1.text(step[-1], 6.4, "closest any success got (5.9 cm)", ha="right", fontsize=8, color="#2a9d55")
    a1.set_ylabel("finger to\nbutton (cm)")
    a1.set_ylim(0, min(100, d.max() * 100 * 1.05))
    a2.plot(step, z, color="#c0392b", lw=1.2, label="radio height")
    a2.axhline(TABLE_TOP_Z, color="gray", ls=":", lw=1)
    a2.text(step[0], TABLE_TOP_Z - 0.04, "table top", fontsize=8, color="gray")
    a2.fill_between(step, z.min() - 0.02, z.max() + 0.02, where=grasp, color="#f4a261", alpha=0.3, label="grasping")
    a2.set_ylabel("radio z (m)")
    a2.set_xlabel("step")
    a2.legend(loc="upper left", fontsize=8)
    a1.set_title("SUCCESS" if success else "timeout (no success)", fontsize=9, loc="left")
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch1", default=None, help="Optional earlier run dir (pre-fix) to show alongside.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(args.run, "rollouts.csv"))))
    by_inst = {int(r["instance_id"]): r for r in rows}
    b1 = {}
    if args.batch1:
        b1 = {int(r["instance_id"]): r for r in csv.DictReader(open(os.path.join(args.batch1, "rollouts.csv")))}

    n, s = len(rows), sum(r["success"] == "1" for r in rows)
    succ_steps = [int(r["steps"]) for r in rows if r["success"] == "1"]
    stat_rows = "".join(
        f"<tr><td>{r['instance_id']}</td><td>{r['seed']}</td>"
        f"<td class={'ok' if r['success'] == '1' else 'bad'}>{'success' if r['success'] == '1' else 'timeout'}</td>"
        f"<td>{r['steps']}</td><td>{float(r['min_finger_button_dist']) * 100:.1f}</td>"
        f"<td>{r['grasped_radio_steps']}</td><td>{r['stale_grasp_at_start'] or '-'}</td>"
        + (f"<td>{'success' if b1[int(r['instance_id'])]['success'] == '1' else 'fail'}</td>" if b1 else "")
        + "</tr>"
        for r in rows
    )
    parts = [
        "<!doctype html><meta charset=utf-8><title>Comet radio rollouts</title>",
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#fafafa;"
        "color:#222}h2{margin-top:40px;border-bottom:2px solid #ddd}td,th{padding:3px 10px;text-align:right}"
        "table{border-collapse:collapse}tr:nth-child(even){background:#eee}.ok{color:#1e8449;font-weight:600}"
        ".bad{color:#b03a2e}.case{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;"
        "margin:16px 0}img{max-width:100%}video{width:560px;margin-right:12px;vertical-align:top}</style>",
        "<h1>Comet (pi0.5) on turning_on_radio</h1>",
        f"<p><b>{s}/{n} successes</b> ({100 * s / n:.0f}%) on the 2025 validation instances, full-res, timeout "
        f"{rows[0]['max_steps']} steps, stale-grasp fix on. Successes took {min(succ_steps)}-{max(succ_steps)} steps. "
        "Every failure is a timeout. No rollout recovered after a failed attempt.</p>",
        "<table><tr><th>instance</th><th>seed</th><th>result</th><th>steps</th><th>closest finger to button "
        "(cm)</th><th>steps grasping radio</th><th>stale grasp released</th>"
        + ("<th>batch 1 (bug)</th>" if b1 else "")
        + f"</tr>{stat_rows}</table>",
        "<p>Videos: left column = wrist cameras, right = head camera. Tip: right-click a video to change speed.</p>",
    ]
    for group in GROUP_ORDER:
        insts = [i for i, (g, _) in CASES.items() if g == group and i in by_inst]
        parts.append(f"<h2>{html.escape(group)} ({len(insts)})</h2>")
        for inst in insts:
            r = by_inst[inst]
            stem = f"turning_on_radio_{inst}_{r['seed']}"
            vid_name = stem + ".mp4"
            shutil.copy(r["video_path"], os.path.join(args.out, vid_name))
            keyframe_strip(r["video_path"], os.path.join(args.out, stem + "_frames.jpg"))
            trace_plot(r["trace_path"], os.path.join(args.out, stem + "_trace.png"), r["success"] == "1")
            parts.append(
                f"<div class=case><h3>Instance {inst} (seed {r['seed']}): "
                f"{'success at step ' + r['steps'] if r['success'] == '1' else 'timeout'}</h3>"
                f"<p>{html.escape(CASES[inst][1])}</p>"
                f"<img src='{stem}_frames.jpg'><br><video controls preload=metadata src='{vid_name}'></video>"
                f"<img src='{stem}_trace.png' style='width:760px'></div>"
            )
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("\n".join(parts))
    print("wrote", os.path.join(args.out, "index.html"))


if __name__ == "__main__":
    main()
