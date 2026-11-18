"""Summarize the base-pi0.5 fix ablation (steering/fix_campaign.py) from traces.

    python steering/fix_report.py --root /home/behavior/ext/runs/fix

Per condition (5 setups): successes, grasps, closest finger to the button (median / best), steps with a finger
within 15 cm of the button (after any scripted approach), torso drift (max |torso - start| over the torso joints),
and arm jitter (median per-step hand displacement in the base frame, mm/step; Comet runs at ~1-3).
"""

import argparse
import csv
import glob
import math
import os
import statistics as st


def per_rollout(r):
    t = list(csv.DictReader(open(r["trace_path"])))
    f = float
    act = [x.get("assist_active") == "1" for x in t]
    pol = [x for x, a in zip(t, act) if not a] or t  # policy-controlled steps
    d = [min(f(x["finger_button_dist_left"]), f(x["finger_button_dist_right"])) for x in pol]
    out = {
        "success": r["success"] == "1",
        "grasp": any(x["grasp_radio_left"] == "1" or x["grasp_radio_right"] == "1" for x in t),
        "closest_cm": 100 * min(d),
        "near15": sum(v < 0.15 for v in d),
    }
    if "torso_q0" in t[0] and t[0]["torso_q0"] != "":
        q0 = [f(t[0][f"torso_q{k}"]) for k in range(4)]
        out["torso_drift"] = max(abs(f(x[f"torso_q{k}"]) - q0[k]) for x in t for k in range(4))
    jit = []
    for side in ("left", "right"):
        p = [[f(x[f"eef_{side}_{c}"]) for c in "xyz"] for x in pol]
        jit.append(st.median(math.dist(a, b) for a, b in zip(p, p[1:])) * 1000)
    out["jitter_mm"] = sum(jit) / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    lines = [
        f"{'condition':32s} succ grasp closest(cm) med/best  near15(med steps)  torso drift(rad, med)  jitter(mm/step)"
    ]
    for d in sorted(glob.glob(f"{args.root}/*/rollouts.csv")):
        name = os.path.basename(os.path.dirname(d))
        rs = [per_rollout(r) for r in csv.DictReader(open(d))]
        n = len(rs)
        td = [x["torso_drift"] for x in rs if "torso_drift" in x]
        lines.append(
            f"{name:32s} {sum(x['success'] for x in rs)}/{n}  {sum(x['grasp'] for x in rs)}/{n}   "
            f"{st.median(x['closest_cm'] for x in rs):5.1f} / {min(x['closest_cm'] for x in rs):5.1f}     "
            f"{st.median(x['near15'] for x in rs):7.0f}           {st.median(td) if td else float('nan'):.3f}"
            f"               {st.median(x['jitter_mm'] for x in rs):.1f}"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
