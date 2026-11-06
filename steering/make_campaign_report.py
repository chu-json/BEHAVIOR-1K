"""Compare several rollout runs (conditions) in one HTML page.

    python steering/make_campaign_report.py --out /home/behavior/ext/report_campaign \
        --run "250 kg (s0-9)=/home/behavior/ext/runs/val10_full_fixed" \
        --run "47.6 kg (s0-9)=/home/behavior/ext/runs/massA_default_s0" ...

Per condition: success rate + Wilson 95% CI, success steps, auto-classified failure modes, and a press-rule check
(would the 2025 ToggledOn rule have fired on a failed rollout?). Then an instance x condition outcome grid, and
example cases (video, keyframes, per-step trace plot) for one success and each failure mode per condition.
"""

import argparse
import csv
import html
import math
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_report import TABLE_TOP_Z, keyframe_strip  # noqa: E402

CAN_TOGGLE_STEPS_2025 = 5
MODES = ["never grasped", "held radio until timeout", "grasped, then let go"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_trace(path):
    t = list(csv.DictReader(open(path)))
    num = lambda k: np.array([float(x[k]) if x.get(k, "") != "" else np.nan for x in t])  # noqa: E731
    return {
        "step": num("step"),
        "dist": np.minimum(num("finger_button_dist_left"), num("finger_button_dist_right")),
        "z": num("radio_z"),
        "grasp": (num("grasp_radio_left") > 0) | (num("grasp_radio_right") > 0),
        "contact": num("finger_contact_radio") if "finger_contact_radio" in t[0] else None,
        "overlap25": num("overlap_2025") if "overlap_2025" in t[0] else None,
        "timer": num("fork_toggle_timer_s") if "fork_toggle_timer_s" in t[0] else None,
    }


def max_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def classify(row, tr):
    if row["success"] == "1":
        return "success"
    if tr["grasp"].sum() < 15:
        return "never grasped"
    if tr["grasp"][-1]:
        return "held radio until timeout"
    return "grasped, then let go"


def trace_plot(tr, out_png, title):
    has_toggle = tr["overlap25"] is not None
    fig, axes = plt.subplots(
        3 if has_toggle else 2, 1, figsize=(9, 4.6 if has_toggle else 3.6), sharex=True, constrained_layout=True
    )
    a1, a2 = axes[0], axes[1]
    s = tr["step"]
    a1.plot(s, tr["dist"] * 100, color="#2a6fdb", lw=1.2)
    a1.axhline(5.9, color="#2a9d55", ls="--", lw=1)
    a1.set_ylabel("finger to\nbutton (cm)")
    a1.set_ylim(0, min(100, np.nanmax(tr["dist"]) * 100 * 1.05))
    a1.set_title(title, fontsize=9, loc="left")
    a2.plot(s, tr["z"], color="#c0392b", lw=1.2, label="radio height")
    a2.axhline(TABLE_TOP_Z, color="gray", ls=":", lw=1)
    lo, hi = np.nanmin(tr["z"]) - 0.02, np.nanmax(tr["z"]) + 0.02
    a2.fill_between(s, lo, hi, where=tr["grasp"], color="#f4a261", alpha=0.3, label="grasping")
    a2.set_ylabel("radio z (m)")
    a2.legend(loc="upper left", fontsize=8)
    if has_toggle:
        a3 = axes[2]
        a3.fill_between(
            s,
            1.1,
            2.1,
            where=tr["overlap25"] > 0,
            color="#16a085",
            alpha=0.6,
            step="mid",
            label="2025 rule: finger overlaps button",
        )
        a3.plot(s, 2.2 + 4 * np.nan_to_num(tr["timer"]), color="#d35400", lw=1, label="fork press timer (x4 s)")
        a3.set_yticks([])
        a3.legend(loc="upper left", fontsize=7, ncol=3)
    axes[-1].set_xlabel("step")
    fig.savefig(out_png, dpi=105)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"label=run_dir", repeatable')
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Comet on turning_on_radio: experiment campaign")
    ap.add_argument("--notes", default="", help="HTML paragraph inserted under the title.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conds = []
    for spec in args.run:
        label, d = spec.split("=", 1)
        rows = list(csv.DictReader(open(os.path.join(d, "rollouts.csv"))))
        for r in rows:
            r["_trace"] = load_trace(r["trace_path"]) if r.get("trace_path") else None
            r["_mode"] = classify(r, r["_trace"]) if r["_trace"] else ("success" if r["success"] == "1" else "?")
        conds.append((label, d, rows))

    css = (
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#fafafa;"
        "color:#222}h2{margin-top:40px;border-bottom:2px solid #ddd}td,th{padding:3px 10px;text-align:right}"
        "table{border-collapse:collapse;margin:8px 0}tr:nth-child(even){background:#eee}.ok{color:#1e8449;"
        "font-weight:600}.bad{color:#b03a2e}.case{background:#fff;border:1px solid #ddd;border-radius:8px;"
        "padding:12px;margin:16px 0}img{max-width:100%}video{width:560px;margin-right:12px;vertical-align:top}"
        "td.l,th.l{text-align:left}</style>"
    )
    p = [
        f"<!doctype html><meta charset=utf-8><title>{html.escape(args.title)}</title>{css}",
        f"<h1>{html.escape(args.title)}</h1>",
        args.notes,
    ]

    # Summary per condition.
    p.append(
        "<h2>Success rate per condition</h2><table><tr><th class=l>condition</th><th>rollouts</th>"
        "<th>successes</th><th>rate</th><th>95% CI</th><th>success steps (min / median / max)</th>"
        + "".join(f"<th>{m}</th>" for m in MODES)
        + "<th>failures where 2025 press rule would have fired</th></tr>"
    )
    for label, _, rows in conds:
        n, k = len(rows), sum(r["success"] == "1" for r in rows)
        lo, hi = wilson(k, n)
        ss = sorted(int(r["steps"]) for r in rows if r["success"] == "1")
        steps_txt = f"{ss[0]} / {int(np.median(ss))} / {ss[-1]}" if ss else "-"
        modes = {m: sum(r["_mode"] == m for r in rows) for m in MODES}
        fired = [
            r
            for r in rows
            if r["success"] != "1"
            and r["_trace"] is not None
            and r["_trace"]["overlap25"] is not None
            and max_run(r["_trace"]["overlap25"] > 0) >= CAN_TOGGLE_STEPS_2025
        ]
        has_shadow = any(r["_trace"] is not None and r["_trace"]["overlap25"] is not None for r in rows)
        p.append(
            f"<tr><td class=l>{html.escape(label)}</td><td>{n}</td><td>{k}</td><td><b>{100 * k / n:.0f}%</b></td>"
            f"<td>{100 * lo:.0f}-{100 * hi:.0f}%</td><td>{steps_txt}</td>"
            + "".join(f"<td>{modes[m]}</td>" for m in MODES)
            + f"<td>{(str(len(fired)) + ' (' + ', '.join(r['instance_id'] for r in fired) + ')') if has_shadow else 'not recorded'}</td></tr>"
        )
    p.append(
        "</table><p>Failure modes are classified from the trace: <i>never grasped</i> = fewer than 15 "
        "grasping steps; <i>held radio until timeout</i> = still grasping at the last step; otherwise "
        "<i>grasped, then let go</i>. The 2025 press-rule column counts failures where the 2025 sphere test "
        "overlapped the button for 5+ consecutive steps (an upper bound: 2025 also required finger contact).</p>"
    )

    # Instance x condition grid.
    insts = sorted({int(r["instance_id"]) for _, _, rows in conds for r in rows})
    p.append(
        "<h2>Outcome per instance</h2><table><tr><th>instance</th>"
        + "".join(f"<th>{html.escape(lbl)}</th>" for lbl, _, _ in conds)
        + "<th>total</th></tr>"
    )
    for i in insts:
        cells, tot_k, tot_n = [], 0, 0
        for _, _, rows in conds:
            rs = [r for r in rows if int(r["instance_id"]) == i]
            txt = " ".join(
                f"<span class={'ok' if r['success'] == '1' else 'bad'} title='seed {r['seed']}, {r['_mode']}'>"
                f"{'S' + r['steps'] if r['success'] == '1' else 'F'}</span>"
                for r in rs
            )
            tot_k += sum(r["success"] == "1" for r in rs)
            tot_n += len(rs)
            cells.append(f"<td>{txt or '-'}</td>")
        p.append(f"<tr><td>{i}</td>{''.join(cells)}<td>{tot_k}/{tot_n}</td></tr>")
    p.append("</table><p>S&lt;steps&gt; = success at that step, F = failure. Hover a cell for seed and mode.</p>")

    # Success-step histogram.
    fig, ax = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
    for label, _, rows in conds:
        ss = [int(r["steps"]) for r in rows if r["success"] == "1"]
        if ss:
            ax.hist(ss, bins=np.arange(800, 3100, 100), alpha=0.5, label=f"{label} (n={len(ss)})")
    ax.set_xlabel("steps to success")
    ax.set_ylabel("rollouts")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.out, "success_steps.png"), dpi=105)
    plt.close(fig)
    p.append("<h2>How long successes take</h2><img src='success_steps.png' style='width:760px'>")

    # Examples: per condition, one success + first example of each failure mode.
    p.append(
        "<h2>Example rollouts</h2><p>Videos: left = wrist cameras, right = head camera. Plots: closest "
        "finger to button; radio height with grasp periods shaded; and (new runs) the 2025 press-rule "
        "overlap and the fork's press timer (flips at 0.15 s).</p>"
    )
    for ci, (label, _, rows) in enumerate(conds):
        p.append(f"<h3>{html.escape(label)}</h3>")
        picks = []
        for mode in ["success"] + MODES:
            r = next((r for r in rows if r["_mode"] == mode and r.get("video_path")), None)
            if r is not None:
                picks.append((mode, r))
        for mode, r in picks:
            stem = f"c{ci}_{r['instance_id']}_{r['seed']}"
            shutil.copy(r["video_path"], os.path.join(args.out, stem + ".mp4"))
            keyframe_strip(r["video_path"], os.path.join(args.out, stem + "_frames.jpg"))
            tr = r["_trace"]
            title = f"{'SUCCESS at ' + r['steps'] if r['success'] == '1' else 'timeout'} - {mode}"
            trace_plot(tr, os.path.join(args.out, stem + "_trace.png"), title)
            closest = np.nanmin(tr["dist"]) * 100
            p.append(
                f"<div class=case><h4>{html.escape(mode)}: instance {r['instance_id']}, seed {r['seed']}, "
                f"{r['steps']} steps</h4><p>closest finger to button {closest:.1f} cm, grasping "
                f"{int(tr['grasp'].sum())} steps, radio max height {np.nanmax(tr['z']):.2f} m</p>"
                f"<img src='{stem}_frames.jpg'><br><video controls preload=metadata src='{stem}.mp4'></video>"
                f"<img src='{stem}_trace.png' style='width:760px'></div>"
            )
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("\n".join(p))
    print("wrote", os.path.join(args.out, "index.html"))


if __name__ == "__main__":
    main()
