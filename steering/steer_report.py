"""Summarize the overnight steering campaign (steering/steer_campaign.py) against the Comet baseline on the same
(instance, seed) pairs (massD + massF runs: whole house, default mass, 3000 steps).

    python steering/steer_report.py [--root /home/behavior/ext/runs/steer] [--json out.json]

Per condition: successes / done, paired flips vs baseline (fail->success, success->fail) with an exact two-sided sign
test, median success step, and condition-specific columns (recoveries fired, depth-estimate error, critic spread).
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st

EXT = "/home/behavior/ext"
BASELINE = [
    f"{EXT}/runs/massD_default_fullscene_s0/rollouts.csv",
    f"{EXT}/runs/massF_default_fullscene_s10/rollouts.csv",
]


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def key(r):
    return (int(r["instance_id"]), int(r["seed"]))


def sign_test(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def critic_spread(name):
    log = f"{EXT}/logs/steer_server_{name}.log"
    if not os.path.exists(log):
        return None
    sp = []
    for line in open(log):
        m = re.search(r"best-of-\d+: hand-to-button ([\d.]+) m \(worst ([\d.]+)\)", line)
        if m:
            sp.append(float(m[2]) - float(m[1]))
    return round(100 * st.median(sp), 1) if sp else None


def summarize(root):
    base = {key(r): r["success"] == "1" for p in BASELINE for r in rows(p)}
    out = []
    for d in sorted(glob.glob(f"{root}/*/rollouts.csv"), key=os.path.getmtime):
        name = os.path.basename(os.path.dirname(d))
        rs = rows(d)
        if not rs:
            continue
        succ = {key(r): r["success"] == "1" for r in rs}
        paired = [k for k in succ if k in base]
        fs = sum(succ[k] and not base[k] for k in paired)
        sf = sum(base[k] and not succ[k] for k in paired)
        steps = [int(r["success_step"]) for r in rs if r["success"] == "1" and r["success_step"]]
        row = {
            "condition": name,
            "success": sum(succ.values()),
            "done": len(rs),
            "baseline_same_pairs": sum(base[k] for k in paired),
            "fail_to_success": fs,
            "success_to_fail": sf,
            "sign_p": round(sign_test(fs, sf), 3),
            "median_success_step": int(st.median(steps)) if steps else None,
            "per_pair": {f"{k[0]}:{k[1]}": int(succ[k]) for k in sorted(succ)},
        }
        if "recoveries" in rs[0]:
            row["recoveries_fired"] = sum(int(r["recoveries"] or 0) > 0 for r in rs)
            row["recovered_then_succeeded"] = sum(int(r["recoveries"] or 0) > 0 and r["success"] == "1" for r in rs)
        if rs[0].get("depth_est_err_med_m") not in (None, ""):
            e = [float(r["depth_est_err_last_m"]) for r in rs if r.get("depth_est_err_last_m")]
            row["depth_err_last_med_cm"] = round(100 * st.median(e), 1) if e else None
        sp = critic_spread(name)
        if sp is not None:
            row["critic_spread_med_cm"] = sp
        out.append(row)
    return base, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{EXT}/runs/steer")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    base, out = summarize(args.root)
    print(f"baseline (massD+massF, same pairs): {sum(base.values())}/{len(base)}")
    print(f"{'condition':22s} succ    base(same)  F->S S->F  p      med step  extra")
    for r in out:
        extra = {
            k: r[k]
            for k in r
            if k in ("recoveries_fired", "recovered_then_succeeded", "depth_err_last_med_cm", "critic_spread_med_cm")
        }
        print(
            f"{r['condition']:22s} {r['success']:2d}/{r['done']:<2d}   {r['baseline_same_pairs']:2d}/{r['done']:<2d}"
            f"       {r['fail_to_success']:2d}   {r['success_to_fail']:2d}  {r['sign_p']:.3f}  "
            f"{r['median_success_step'] or '-':>6}    {extra or ''}"
        )
    if args.json:
        json.dump(
            {"baseline": {f"{k[0]}:{k[1]}": int(v) for k, v in base.items()}, "conditions": out},
            open(args.json, "w"),
            indent=1,
        )


if __name__ == "__main__":
    main()
