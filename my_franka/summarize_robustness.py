"""
Merge all robustness runs in data/robustness/ into one table.

Reads results_first50.json (layouts 0-49, all 7 conditions) plus results_A.json / results_B.json (layouts 50-199),
concatenates the per-layout outcomes of each condition, and prints:
  - success rate with a 95% Wilson confidence interval
  - for each condition, how many layouts it LOST / GAINED relative to each of the two baseline runs
    (baseline_repeat is the same condition as baseline, so the flips between those two are the noise floor:
    the diffusion policy samples its actions randomly, so even identical runs disagree on some layouts)

    python my_franka/summarize_robustness.py
"""
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROB = os.path.join(HERE, "data", "robustness")
ORDER = ["baseline", "baseline_repeat", "light_0.5x", "light_2x", "wall_texture", "red_apple", "table_texture"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def load():
    merged = {}  # condition -> {layout number: success}
    for path in [os.path.join(ROB, "results_first50.json")] + sorted(glob.glob(os.path.join(ROB, "results_[A-Z].json"))):
        if not os.path.exists(path):
            continue
        for cond, r in json.load(open(path))["results"].items():
            start = r.get("start", 0)
            for i, ok in enumerate(r["outcomes"]):
                merged.setdefault(cond, {})[start + i] = bool(ok)
    return merged


def main():
    data = load()
    conds = [c for c in ORDER if c in data] + [c for c in data if c not in ORDER]
    print(f"{'condition':<17}{'layouts':>8}{'success':>9}   {'95% CI':<14}  {'vs baseline':<26}{'vs baseline_repeat':<26}")
    for c in conds:
        d = data[c]
        n, k = len(d), sum(d.values())
        lo, hi = wilson(k, n)
        cells = []
        for ref in ("baseline", "baseline_repeat"):
            if ref in data and c != ref:
                common = sorted(set(d) & set(data[ref]))
                lost = sum(data[ref][i] and not d[i] for i in common)
                gained = sum(d[i] and not data[ref][i] for i in common)
                cells.append(f"lost {lost:>2}, gained {gained:>2} / {len(common)}")
            else:
                cells.append("-")
        print(f"{c:<17}{n:>8}{k / n * 100:>8.1f}%   {lo * 100:4.1f}-{hi * 100:4.1f}%      {cells[0]:<26}{cells[1]:<26}")
    if "baseline" in data and "baseline_repeat" in data:
        common = sorted(set(data["baseline"]) & set(data["baseline_repeat"]))
        flips = sum(data["baseline"][i] != data["baseline_repeat"][i] for i in common)
        print(f"\nnoise floor: the two identical baseline runs disagree on {flips} of {len(common)} layouts "
              f"({100 * flips / len(common):.0f}%). A condition is only clearly different if its lost/gained counts "
              f"are well beyond that, or its confidence interval excludes the baselines' rates.")
        both = {i: data["baseline"][i] and data["baseline_repeat"][i] for i in common}
        pooled = (sum(data["baseline"].values()) + sum(data["baseline_repeat"].values())) / (
            len(data["baseline"]) + len(data["baseline_repeat"]))
        print(f"pooled baseline success rate (both runs): {100 * pooled:.1f}%")


if __name__ == "__main__":
    main()
