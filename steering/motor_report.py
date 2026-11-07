"""Closed-loop motor-command test: does a simple instruction change what the robot does, in the requested direction?

    python steering/motor_report.py --root /home/behavior/ext/runs/day/motor

Reads <root>/groups.json and each group's traces (run_rollouts.py --trace). Per rollout:
  left_raise / right_raise  max end-effector height above its starting height (robot base frame), m
  left_path / right_path    end-effector path length in the base frame, m
  yaw_change                signed cumulative base rotation, deg (+ = left / counter-clockwise)
  forward                   base displacement along the starting heading, m
  total_motion              left_path + right_path + base path length, m
Each prompt is compared with P0 on the same setup (paired), and judged on its expected readout:
  M_left_arm_up  -> left_raise - right_raise up      M_right_arm_up -> right_raise - left_raise up
  M_turn_left    -> yaw_change up                    M_turn_right   -> yaw_change down
  M_forward      -> forward up                       M_still        -> total_motion down
"""

import argparse
import csv
import html
import json
import math
import os

EXPECT = {
    "M_left_arm_up": ("raise_bias_left", +1),
    "M_right_arm_up": ("raise_bias_left", -1),
    "M_turn_left": ("yaw_change", +1),
    "M_turn_right": ("yaw_change", -1),
    "M_forward": ("forward", +1),
    "M_still": ("total_motion", -1),
}
ORDER = ["P0", "M_left_arm_up", "M_right_arm_up", "M_turn_left", "M_turn_right", "M_forward", "M_still", "X_nonsense"]
SHOW = [
    "left_raise",
    "right_raise",
    "raise_bias_left",
    "left_path",
    "right_path",
    "yaw_change",
    "forward",
    "total_motion",
]


def metrics(trace_path):
    t = list(csv.DictReader(open(trace_path)))
    f = lambda x, k: float(x[k])  # noqa: E731
    out = {}
    for side in ("left", "right"):
        z0 = f(t[0], f"eef_{side}_z")
        out[f"{side}_raise"] = max(f(x, f"eef_{side}_z") for x in t) - z0
        out[f"{side}_path"] = sum(
            math.dist([f(a, f"eef_{side}_{c}") for c in "xyz"], [f(b, f"eef_{side}_{c}") for c in "xyz"])
            for a, b in zip(t, t[1:])
        )
    out["raise_bias_left"] = out["left_raise"] - out["right_raise"]
    yaw = [f(x, "base_yaw") for x in t]
    out["yaw_change"] = math.degrees(sum(math.atan2(math.sin(b - a), math.cos(b - a)) for a, b in zip(yaw, yaw[1:])))
    x0, y0, h0 = f(t[0], "base_x"), f(t[0], "base_y"), yaw[0]
    out["forward"] = (f(t[-1], "base_x") - x0) * math.cos(h0) + (f(t[-1], "base_y") - y0) * math.sin(h0)
    base_path = sum(
        math.dist((f(a, "base_x"), f(a, "base_y")), (f(b, "base_x"), f(b, "base_y"))) for a, b in zip(t, t[1:])
    )
    out["total_motion"] = out["left_path"] + out["right_path"] + base_path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    groups = json.load(open(os.path.join(args.root, "groups.json")))
    data = {}  # (model, prompt) -> {setup: metrics}
    for g in groups:
        f = os.path.join(args.root, g["name"], "rollouts.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            m = metrics(r["trace_path"])
            m["success"] = int(r["success"])
            m["video"] = r["video_path"]
            data.setdefault((g["checkpoint"], g["prompt_id"]), {})[f"{r['instance_id']}_s{r['seed']}"] = m
    models = list(dict.fromkeys(g["checkpoint"] for g in groups))
    prompts = {g["prompt_id"]: g["prompt"] for g in groups}
    css = (
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#fafafa;"
        "color:#222}h2{margin-top:32px;border-bottom:2px solid #ddd}td,th{padding:3px 8px;text-align:right}"
        "table{border-collapse:collapse;margin:8px 0}tr:nth-child(even){background:#eee}td.l,th.l{text-align:left}"
        ".yes{color:#1e8449;font-weight:600}.no{color:#b03a2e}</style>"
    )
    p = [
        f"<!doctype html><meta charset=utf-8><title>Motor-command steerability</title>{css}",
        "<h1>Closed-loop motor-command prompts</h1>",
        "<p>1000-step episodes, setups 242/203/187 (labels setup_seed), whole house, 47.6 kg base. Values are means over setups. "
        "<b>Followed?</b> compares each prompt with P0 on the same setup: the change in the expected readout must "
        "have the requested sign on every setup+seed pair (k/n shown). Units: m, deg.</p>",
    ]
    lines = []
    for model in models:
        p.append(
            f"<h2>{html.escape(model)}</h2><table><tr><th class=l>prompt</th><th class=l>text</th>"
            + "".join(f"<th>{c}</th>" for c in SHOW)
            + "<th>success</th><th>expected</th>"
            "<th>delta vs P0 per setup</th><th>followed?</th></tr>"
        )
        base = data.get((model, "P0"), {})
        for pid in ORDER:
            cell = data.get((model, pid))
            if not cell:
                continue
            means = {c: sum(v[c] for v in cell.values()) / len(cell) for c in SHOW}
            exp, verdict, deltas = "-", "", ""
            if pid in EXPECT and base:
                key, sgn = EXPECT[pid]
                ds = [cell[s][key] - base[s][key] for s in cell if s in base]
                k = sum((d > 0) if sgn > 0 else (d < 0) for d in ds)
                exp = f"{key} {'up' if sgn > 0 else 'down'}"
                deltas = ", ".join(f"{d:+.2f}" for d in ds)
                ok = k == len(ds) and ds
                verdict = f"<span class={'yes' if ok else 'no'}>{'yes' if ok else 'no'} ({k}/{len(ds)})</span>"
                lines.append(f"{model:12s} {pid:15s} {exp:22s} deltas [{deltas}] -> {k}/{len(ds)}")
            vids = " ".join(f"<a href='file://{html.escape(v['video'])}'>{s}</a>" for s, v in cell.items())
            p.append(
                f"<tr><td class=l>{pid}</td><td class=l>{html.escape(prompts[pid])}<br><small>videos: {vids}</small></td>"
                + "".join(f"<td>{means[c]:+.2f}</td>" for c in SHOW)
                + f"<td>{sum(v['success'] for v in cell.values())}/{len(cell)}</td><td class=l>{exp}</td>"
                f"<td>{deltas}</td><td>{verdict}</td></tr>"
            )
        p.append("</table>")
    with open(os.path.join(args.root, "motor.html"), "w") as f:
        f.write("\n".join(p))
    print("\n".join(lines) or "no motor data yet")


if __name__ == "__main__":
    main()
