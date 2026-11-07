"""Closed-loop opposite-pair / control contrasts for the motor-command runs (/home/behavior/ext/runs/day/motor)."""

import csv
import json
import os
import sys

sys.path.insert(0, "/home/behavior/BEHAVIOR-1K/steering")
from motor_report import metrics  # noqa: E402

root = "/home/behavior/ext/runs/day/motor"
groups = json.load(open(f"{root}/groups.json"))
D = {}
for g in groups:
    f = f"{root}/{g['name']}/rollouts.csv"
    if not os.path.exists(f):
        continue
    for r in csv.DictReader(open(f)):
        D.setdefault((g["checkpoint"], g["prompt_id"]), {})[f"{r['instance_id']}_s{r['seed']}"] = metrics(
            r["trace_path"]
        )


def contrast(model, a, b, key, sign=+1):
    A, B = D.get((model, a), {}), D.get((model, b), {})
    ks = sorted(set(A) & set(B))
    v = [sign * (A[k][key] - B[k][key]) for k in ks]
    return v


for model in ("C_pi05_base", "A_radio_sft", "B_pt50"):
    print("==", model)
    for a, b, key, label in [
        ("M_turn_left", "M_turn_right", "yaw_change", "turn left - turn right (yaw, deg)"),
        ("M_left_arm_up", "M_right_arm_up", "raise_bias_left", "raise left - raise right (L-R raise, m)"),
        ("X_nonsense", "M_still", "total_motion", "nonsense - stay still (total motion, m)"),
        ("P0", "M_still", "total_motion", "P0 - stay still (total motion, m)"),
        ("M_forward", "M_still", "forward", "forward - stay still (forward disp, m)"),
    ]:
        v = contrast(model, a, b, key)
        if v:
            print(
                f"  {label:42s} expected >0: {sum(x > 0 for x in v)}/{len(v)}  median {sorted(v)[len(v) // 2]:+.2f}  values {[round(x, 2) for x in v]}"
            )
