"""Aggregate the prompt-steerability sweep into one CSV + an HTML summary. Safe to rerun at any time (partial data).

    python steering/sweep_report.py --root /home/behavior/ext/runs/prompt_sweep

<root>/groups.json lists the groups ({"name", "checkpoint", "prompt_id", "prompt", "seed", "pass"}); each group's
run_rollouts.py output dir is <root>/<name>. Per rollout we derive from the --trace CSV:
  min_dist_left_cm / min_dist_right_cm  closest finger-link-to-button distance per hand over the episode
  closest_hand                           hand with the smaller of the two minima
  pressing_hand                          successes only: hand closer to the button over the last 10 steps
  max_lift_cm / lifted                   max radio height above its resting height; lifted if > 3 cm
  heading_err_grasp_deg                  base heading error to the radio at first grasp (or at closest base
                                         approach if never grasped); min_heading_err_deg over the episode
"""

import argparse
import csv
import html
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_campaign_report import instance_geometry, wilson  # noqa: E402

REST_Z = 0.534  # radio resting height on the table
LIFT_M = 0.03
FIELDS = [
    "pass",
    "checkpoint",
    "prompt_id",
    "prompt",
    "instance_id",
    "seed",
    "success",
    "steps_to_success",
    "steps",
    "min_dist_left_cm",
    "min_dist_right_cm",
    "closest_hand",
    "pressing_hand",
    "max_lift_cm",
    "lifted",
    "grasped",
    "heading_err_grasp_deg",
    "min_heading_err_deg",
    "wall_time_s",
    "video_path",
    "trace_path",
]
PROMPT_ORDER = ["P0", "P1", "P2", "P3", "P4"]


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def derive(r):
    t = list(csv.DictReader(open(r["trace_path"])))
    dl = [float(x["finger_button_dist_left"]) for x in t]
    dr = [float(x["finger_button_dist_right"]) for x in t]
    z = [float(x["radio_z"]) for x in t]
    rx, ry = instance_geometry(int(r["instance_id"]))["radio_xy"]
    herr = [
        abs(math.degrees(wrap(math.atan2(ry - float(x["base_y"]), rx - float(x["base_x"])) - float(x["base_yaw"]))))
        for x in t
    ]
    k = next((j for j, x in enumerate(t) if x["grasp_radio_left"] == "1" or x["grasp_radio_right"] == "1"), None)
    grasped = int(k is not None)
    if k is None:
        k = min(range(len(t)), key=lambda j: math.hypot(rx - float(t[j]["base_x"]), ry - float(t[j]["base_y"])))
    ok = r["success"] == "1"
    pressing = ""
    if ok:
        pressing = "left" if sum(dl[-10:]) < sum(dr[-10:]) else "right"
    lift = max(0.0, max(z) - REST_Z)
    return {
        "success": int(ok),
        "steps_to_success": r["steps"] if ok else "",
        "steps": r["steps"],
        "min_dist_left_cm": round(100 * min(dl), 1),
        "min_dist_right_cm": round(100 * min(dr), 1),
        "closest_hand": "left" if min(dl) < min(dr) else "right",
        "pressing_hand": pressing,
        "max_lift_cm": round(100 * lift, 1),
        "lifted": int(lift > LIFT_M),
        "grasped": grasped,
        "heading_err_grasp_deg": round(herr[k], 1),
        "min_heading_err_deg": round(min(herr), 1),
        "wall_time_s": r["wall_time_s"],
        "video_path": r["video_path"],
        "trace_path": r["trace_path"],
    }


def pct(k, n):
    return f"{100 * k / n:.0f}%" if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    groups = json.load(open(os.path.join(args.root, "groups.json")))

    rows = []
    for g in groups:
        f = os.path.join(args.root, g["name"], "rollouts.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            if not r.get("trace_path"):
                continue
            row = {k: g[k] for k in ("pass", "checkpoint", "prompt_id", "prompt", "seed")}
            row["instance_id"] = r["instance_id"]
            row.update(derive(r))
            rows.append(row)
    with open(os.path.join(args.root, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    prompts = {g["prompt_id"]: g["prompt"] for g in groups}
    ckpts = list(dict.fromkeys(g["checkpoint"] for g in groups))
    css = (
        "<style>body{font:15px system-ui;max-width:1300px;margin:24px auto;padding:0 16px;background:#fafafa;"
        "color:#222}h2{margin-top:32px;border-bottom:2px solid #ddd}td,th{padding:3px 9px;text-align:right}"
        "table{border-collapse:collapse;margin:8px 0}tr:nth-child(even){background:#eee}td.l,th.l{text-align:left}"
        ".up{color:#1e8449;font-weight:600}.down{color:#b03a2e;font-weight:600}</style>"
    )
    p = [
        f"<!doctype html><meta charset=utf-8><title>Prompt steerability sweep</title>{css}",
        "<h1>Prompt steerability: turning_on_radio</h1>",
        "<p>Whole house, 47.6 kg base, full-res, stale-grasp fix, timeout 3000. Setups 242, 203, 109, 181, 187. "
        "<b>Hand</b> = which hand's fingers were closest to the button (for successes: over the last 10 steps, i.e. "
        "the pressing hand). <b>Lifted</b> = radio raised &gt; 3 cm above its resting height at any point.</p>",
    ]
    p.append(
        "<h2>Prompts</h2><table><tr><th class=l>id</th><th class=l>prompt</th></tr>"
        + "".join(
            f"<tr><td class=l>{k}</td><td class=l>{html.escape(prompts[k])}</td></tr>"
            for k in PROMPT_ORDER
            if k in prompts
        )
        + "</table>"
    )

    for pas in dict.fromkeys(r["pass"] for r in rows):
        p.append(f"<h2>Results ({html.escape(pas)})</h2>")
        p.append(
            "<table><tr><th class=l>checkpoint</th><th class=l>prompt</th><th>rollouts</th><th>success</th>"
            "<th>95% CI</th><th>presses by left / right</th><th>closest hand left (all rollouts)</th>"
            "<th>lifted</th><th>median max lift (cm)</th><th>median heading err at grasp</th>"
            "<th>shift vs P0 in requested direction?</th></tr>"
        )
        for ck in ckpts:
            base = [r for r in rows if r["pass"] == pas and r["checkpoint"] == ck and r["prompt_id"] == "P0"]
            for pid in PROMPT_ORDER:
                rs = [r for r in rows if r["pass"] == pas and r["checkpoint"] == ck and r["prompt_id"] == pid]
                if not rs:
                    continue
                n, k = len(rs), sum(r["success"] for r in rs)
                lo, hi = wilson(k, n)
                pl = sum(r["pressing_hand"] == "left" for r in rs)
                pr = sum(r["pressing_hand"] == "right" for r in rs)
                cl = sum(r["closest_hand"] == "left" for r in rs)
                lf = sum(r["lifted"] for r in rs)
                med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")  # noqa: E731
                verdict = ""
                if pid != "P0" and base:
                    bl = sum(r["closest_hand"] == "left" for r in base) / len(base)
                    if pid == "P2":
                        d = cl / n - bl
                        verdict = f"left-closest {100 * bl:.0f}% -> {100 * cl / n:.0f}%"
                    elif pid == "P3":
                        d = (1 - cl / n) - (1 - bl)
                        verdict = f"right-closest {100 * (1 - bl):.0f}% -> {100 * (1 - cl / n):.0f}%"
                    elif pid == "P4":
                        # Only rollouts that grasped the radio can show compliance; a robot that never touches
                        # the radio "doesn't lift it" trivially.
                        bg = [r for r in base if r["grasped"]]
                        g = [r for r in rs if r["grasped"]]
                        blg = sum(r["lifted"] for r in bg) / len(bg) if bg else float("nan")
                        lg = sum(r["lifted"] for r in g) / len(g) if g else float("nan")
                        d = blg - lg
                        verdict = (
                            f"lifted among rollouts that grasped: {100 * blg:.0f}% ({len(bg)}) -> "
                            f"{100 * lg:.0f}% ({len(g)})"
                        )
                    else:
                        d = None
                        verdict = f"success {sum(r['success'] for r in base)}/{len(base)} -> {k}/{n} (paraphrase)"
                    if d is not None:
                        cls = "up" if d > 0 else ("down" if d < 0 else "")
                        word = "yes" if d > 0 else ("no (opposite)" if d < 0 else "no change")
                        verdict = f"<span class={cls}>{word}</span>: {verdict}"
                p.append(
                    f"<tr><td class=l>{html.escape(ck)}</td><td class=l>{pid}</td><td>{n}</td><td><b>{k}/{n}</b></td>"
                    f"<td>{100 * lo:.0f}-{100 * hi:.0f}%</td><td>{pl} / {pr}</td><td>{pct(cl, n)}</td>"
                    f"<td>{pct(lf, n)}</td><td>{med([r['max_lift_cm'] for r in rs]):.1f}</td>"
                    f"<td>{med([r['heading_err_grasp_deg'] for r in rs]):.0f}&deg;</td><td class=l>{verdict}</td></tr>"
                )
        p.append("</table>")
        # Pressing hand per setup x prompt (L/R for successes, F for failures).
        for ck in ckpts:
            p.append(
                f"<p><b>{html.escape(ck)}</b>: pressing hand per setup (L/R = success pressed with left/right, "
                "F = failed; lowercase = hand closest to the button in a failure)</p><table><tr><th>setup</th>"
                + "".join(f"<th>{pid}</th>" for pid in PROMPT_ORDER)
                + "</tr>"
            )
            for inst in dict.fromkeys(r["instance_id"] for r in rows):
                cells = []
                for pid in PROMPT_ORDER:
                    rr = [
                        r
                        for r in rows
                        if r["pass"] == pas
                        and r["checkpoint"] == ck
                        and r["prompt_id"] == pid
                        and r["instance_id"] == inst
                    ]
                    if not rr:
                        cells.append("<td>-</td>")
                        continue
                    r = rr[0]
                    txt = r["pressing_hand"][0].upper() if r["success"] else "F" + r["closest_hand"][0]
                    cells.append(f"<td>{txt}</td>")
                p.append(f"<tr><td>{inst}</td>{''.join(cells)}</tr>")
            p.append("</table>")

    shown = [c for c in FIELDS if c != "trace_path"]
    p.append("<h2>Every rollout</h2><table><tr>" + "".join(f"<th>{c}</th>" for c in shown) + "</tr>")
    for r in rows:
        cells = []
        for c in shown:
            v = r[c]
            if c == "video_path" and v:
                v = f"<a href='file://{html.escape(v)}'>video</a>"
            else:
                v = html.escape(str(v))
            cells.append(f"<td>{v}</td>")
        p.append("<tr>" + "".join(cells) + "</tr>")
    p.append("</table>")
    with open(os.path.join(args.root, "index.html"), "w") as f:
        f.write("\n".join(p))
    print(f"{len(rows)} rollouts -> {os.path.join(args.root, 'results.csv')}, index.html")


if __name__ == "__main__":
    main()
