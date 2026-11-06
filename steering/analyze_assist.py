"""Report for the align-assist causal test: assisted rollouts vs. the same (instance, seed) pairs without assist.

    python steering/analyze_assist.py --out /home/behavior/ext/report_assist \
        --assisted /home/behavior/ext/runs/assist_right /home/behavior/ext/runs/assist_left_control \
        --baseline /home/behavior/ext/runs/massA_default_s0 /home/behavior/ext/runs/massC_default_s10 \
                   /home/behavior/ext/runs/massE_default_s20_t4299
"""

import argparse
import csv
import html
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_campaign_report import (  # noqa: E402
    instance_geometry,
    load_trace,
    min_heading_err,
    trace_plot,
    wilson,
)
from make_report import keyframe_strip  # noqa: E402

BASELINE_TIMEOUT = 3000  # baseline runs with a longer timeout are scored as if cut at 3000 (none succeeded later)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--assisted", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = {}
    for d in args.baseline:
        for r in csv.DictReader(open(os.path.join(d, "rollouts.csv"))):
            ok = r["success"] == "1" and int(r["steps"]) <= BASELINE_TIMEOUT
            base[(r["instance_id"], r["seed"])] = (ok, r)
    rows = []
    for d in args.assisted:
        rows += list(csv.DictReader(open(os.path.join(d, "rollouts.csv"))))

    def group(r):
        return "right" if instance_geometry(int(r["instance_id"]))["bearing"] < 0 else "left"

    css = (
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#fafafa;"
        "color:#222}h2{margin-top:36px;border-bottom:2px solid #ddd}td,th{padding:3px 10px;text-align:right}"
        "table{border-collapse:collapse;margin:8px 0}tr:nth-child(even){background:#eee}.ok{color:#1e8449;"
        "font-weight:600}.bad{color:#b03a2e}.case{background:#fff;border:1px solid #ddd;border-radius:8px;"
        "padding:12px;margin:16px 0}img{max-width:100%}video{width:560px;margin-right:12px;vertical-align:top}"
        "td.l,th.l{text-align:left}</style>"
    )
    p = [
        f"<!doctype html><meta charset=utf-8><title>Align-assist causal test</title>{css}",
        "<h1>Causal test: is base alignment the bottleneck?</h1>",
        "<p>Intervention (steering/align_assist.py): once the base has moved and then stopped within 1.5 m of the "
        "radio, more than 15&deg; off facing it, and is not holding it, pause the policy, rotate the base in place "
        "to face the radio's true position, hold 10 steps, reset the policy's action queue, resume. At most once "
        "per episode. Same settings as the baseline (47.6 kg base, task rooms, full-res, stale-grasp fix, timeout "
        "3000) and the same (instance, seed) pairs. Note: the simulator is not bit-reproducible, so pairs share "
        "settings, not trajectories.</p>",
    ]

    p.append(
        "<h2>Success: assisted vs. baseline (same instance + seed)</h2><table><tr><th class=l>group</th>"
        "<th>pairs</th><th>baseline</th><th>assisted</th><th>assisted 95% CI</th><th>intervention fired</th>"
        "<th>assisted success when fired</th></tr>"
    )
    for g in ("right", "left"):
        rs = [r for r in rows if group(r) == g]
        if not rs:
            continue
        n = len(rs)
        kb = sum(base[(r["instance_id"], r["seed"])][0] for r in rs if (r["instance_id"], r["seed"]) in base)
        ka = sum(r["success"] == "1" for r in rs)
        fired = [r for r in rs if int(r["assist_interventions"] or 0) > 0]
        lo, hi = wilson(ka, n)
        p.append(
            f"<tr><td class=l>radio on robot's {g}</td><td>{n}</td><td>{kb}/{n}</td><td><b>{ka}/{n}</b></td>"
            f"<td>{100 * lo:.0f}-{100 * hi:.0f}%</td><td>{len(fired)}/{n}</td>"
            f"<td>{sum(r['success'] == '1' for r in fired)}/{len(fired)}</td></tr>"
        )
    p.append("</table>")

    p.append(
        "<h2>Every assisted rollout</h2><table><tr><th>instance</th><th>seed</th><th>side</th><th>baseline</th>"
        "<th>assisted</th><th>intervention (steps: heading error before -> after @ distance)</th>"
        "<th>closest finger to button after intervention</th><th>best heading alignment</th></tr>"
    )
    for r in rows:
        b = base.get((r["instance_id"], r["seed"]))
        tr = load_trace(r["trace_path"])
        geo = instance_geometry(int(r["instance_id"]))
        after = ""
        if tr.get("assist") is not None and np.nansum(tr["assist"]) > 0:
            end = int(np.max(np.nonzero(tr["assist"] > 0)))
            after = f"{np.nanmin(tr['dist'][end:]) * 100:.1f} cm"
        ok = r["success"] == "1"
        p.append(
            f"<tr><td>{r['instance_id']}</td><td>{r['seed']}</td><td>{group(r)}</td>"
            f"<td class={'ok' if b and b[0] else 'bad'}>{('S' if b[0] else 'F') if b else '?'}</td>"
            f"<td class={'ok' if ok else 'bad'}>{'S ' + r['steps'] if ok else 'F'}</td>"
            f"<td>{html.escape(r['assist_log'] or 'not fired')}</td><td>{after or '-'}</td>"
            f"<td>{min_heading_err(tr, geo['radio_xy']):.0f}&deg;</td></tr>"
        )
    p.append("</table><p>Successes in all earlier runs got a finger to within ~5-6 cm of the button.</p>")

    p.append("<h2>Examples</h2>")
    picks = []
    for want in (("right", True), ("right", False), ("left", True), ("left", False)):
        r = next(
            (
                r
                for r in rows
                if group(r) == want[0] and (r["success"] == "1") == want[1] and int(r["assist_interventions"] or 0) > 0
            ),
            None,
        )
        if r is not None:
            picks.append(r)
    for r in picks:
        stem = f"assist_{r['instance_id']}_{r['seed']}"
        shutil.copy(r["video_path"], os.path.join(args.out, stem + ".mp4"))
        keyframe_strip(r["video_path"], os.path.join(args.out, stem + "_frames.jpg"))
        ok = r["success"] == "1"
        trace_plot(
            load_trace(r["trace_path"]),
            os.path.join(args.out, stem + "_trace.png"),
            f"{'SUCCESS at ' + r['steps'] if ok else 'timeout'} - assisted",
        )
        p.append(
            f"<div class=case><h3>Instance {r['instance_id']} (radio on {group(r)}), seed {r['seed']}: "
            f"{'success at ' + r['steps'] if ok else 'timeout'}</h3><p>{html.escape(r['assist_log'])}</p>"
            f"<img src='{stem}_frames.jpg'><br><video controls preload=metadata src='{stem}.mp4'></video>"
            f"<img src='{stem}_trace.png' style='width:760px'></div>"
        )
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("\n".join(p))
    print("wrote", os.path.join(args.out, "index.html"))


if __name__ == "__main__":
    main()
