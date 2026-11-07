"""Base pi0.5 exact-sweep analysis: per-prompt hand, lift, and per-arm activity (uses motor_report.metrics)."""

import csv, sys, statistics as st

sys.path.insert(0, "/home/behavior/BEHAVIOR-1K/steering")
from motor_report import metrics

rows = list(csv.DictReader(open("/home/behavior/ext/runs/day/base_sweep/results.csv")))
print("rollouts", len(rows), "successes", sum(r["success"] == "1" for r in rows))
print("prompt  closest-left  minL(cm) minR(cm)  lifted  grasped | L raise  R raise  L path  R path  (medians over 10)")
M = {}
for r in rows:
    M[r["trace_path"]] = metrics(r["trace_path"])
for p in ["P0", "P1", "P2", "P3", "P4"]:
    rs = [r for r in rows if r["prompt_id"] == p]
    m = [M[r["trace_path"]] for r in rs]
    med = lambda k: st.median(x[k] for x in m)
    print(
        f"{p}      {sum(r['closest_hand'] == 'left' for r in rs)}/{len(rs)}        {st.median(float(r['min_dist_left_cm']) for r in rs):5.1f}  {st.median(float(r['min_dist_right_cm']) for r in rs):5.1f}    {sum(r['lifted'] == '1' for r in rs)}/{len(rs)}   {sum(r['grasped'] == '1' for r in rs)}/{len(rs)}   | {med('left_raise'):.2f}    {med('right_raise'):.2f}    {med('left_path'):.1f}   {med('right_path'):.1f}"
    )
key = lambda r: (r["instance_id"], r["seed"])
base = {key(r): M[r["trace_path"]] for r in rows if r["prompt_id"] == "P0"}
for p in ["P2", "P3", "P1", "P4"]:
    ds = []
    for r in rows:
        if r["prompt_id"] != p:
            continue
        m = M[r["trace_path"]]
        b = base[key(r)]
        ds.append((m["left_path"] - m["right_path"]) - (b["left_path"] - b["right_path"]))
    print(
        p,
        "paired change in (left path - right path) vs P0: median",
        round(st.median(ds), 2),
        "positive",
        sum(d > 0 for d in ds),
        "/",
        len(ds),
    )
