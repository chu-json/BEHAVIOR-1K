"""Turn prompt_probe.py outputs into language-sensitivity numbers + an HTML page.

    python steering/probe_report.py --inputs probe_C_pi05_base.npz probe_A_radio_sft.npz ... --out <dir>

Per model and prompt (vs P0, same observation and same diffusion noise):
  effect ratio[group] = mean|a(prompt) - a(P0)| / mean|a(P0, noise a) - a(P0, noise b)|   (>1: prompt beats noise)
  directional deltas (paired over observations, averaged over noise seeds), with the expected sign:
    arm bias     = predicted left-arm travel - right-arm travel (rad over the 32-step chunk)
                   expected up for P2 / M_left_arm_up, down for P3 / M_right_arm_up
    turn rate    = mean predicted base wz; expected up for M_turn_left, down for M_turn_right
    forward vel  = mean predicted base vx; expected up for M_forward
    total motion = arm travel (both) + |base velocity|; expected down for M_still
  p = two-sided sign test over observations.
"""

import argparse
import html
import math
import os

import numpy as np

GROUPS = {
    "base": [0, 1, 2],
    "torso": [3, 4, 5, 6],
    "left_arm": list(range(7, 14)),
    "left_grip": [14],
    "right_arm": list(range(15, 22)),
    "right_grip": [22],
}
Q_LEFT, Q_RIGHT = slice(3, 10), slice(28, 35)  # arm joint positions in the 61-d proprio
EXPECT = {
    "P2": ("arm_bias", +1),
    "P3": ("arm_bias", -1),
    "M_left_arm_up": ("arm_bias", +1),
    "M_right_arm_up": ("arm_bias", -1),
    "M_turn_left": ("turn_rate", +1),
    "M_turn_right": ("turn_rate", -1),
    "M_forward": ("forward_vel", +1),
    "M_still": ("total_motion", -1),
}


def sign_test_p(x):
    x = x[np.abs(x) > 1e-9]
    n, k = len(x), int((x > 0).sum())
    if n == 0:
        return 1.0
    k = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n * 2
    return min(1.0, p)


CONTROLS = ("X_nonsense", "X_empty")
PAIRS = [
    ("P2", "P3", "arm_bias", "left-hand vs right-hand"),
    ("M_left_arm_up", "M_right_arm_up", "arm_bias", "raise left vs raise right"),
    ("M_turn_left", "M_turn_right", "turn_rate", "turn left vs turn right"),
]


def specific(res, pid, key, sgn):
    """Control-adjusted test: the prompt's shift minus the mean shift of the nonsense/empty prompts, sign-tested per
    observation in the expected direction. Passes if the mean has the expected sign and p < 0.05."""
    ctrl = [res[c]["dirs"][key]["vec"] for c in CONTROLS if c in res]
    if not ctrl:
        return "no controls", False
    v = res[pid]["dirs"][key]["vec"] - np.mean(ctrl, axis=0)
    pval = sign_test_p(v)
    return f"vs controls: {v.mean():+.3f}, p={pval:.2g}", bool(np.sign(v.mean()) == sgn and pval < 0.05)


def pair_contrast(res, a, b, key):
    """Opposite-instruction contrast readout(a) - readout(b); cancels any generic 'extra words' effect."""
    if a not in res or b not in res:
        return None
    v = res[a]["dirs"][key]["vec"] - res[b]["dirs"][key]["vec"]
    return float(v.mean()), float((v > 0).mean()), sign_test_p(v)


def readouts(chunks, proprio):
    """chunks (N, P, S, 32, 23) -> dict of (N, P, S) arrays."""

    def travel(sl, q):
        a = chunks[..., sl]
        first = np.abs(a[..., 0, :] - q[:, None, None, :]).sum(-1)
        return first + np.abs(np.diff(a, axis=-2)).sum((-1, -2))

    tl = travel(slice(7, 14), proprio[:, Q_LEFT])
    tr = travel(slice(15, 22), proprio[:, Q_RIGHT])
    base = chunks[..., 0:3]
    return {
        "arm_bias": tl - tr,
        "turn_rate": base[..., 2].mean(-1),
        "forward_vel": base[..., 0].mean(-1),
        "total_motion": tl + tr + np.abs(base).sum(-1).mean(-1),
        "left_travel": tl,
        "right_travel": tr,
    }


def analyze(path):
    d = np.load(path, allow_pickle=True)
    chunks, proprio = d["chunks"], d["proprio"]
    prompts = list(d["prompts"])
    p0 = prompts.index("P0")
    n = chunks.shape[0]
    noise = {
        g: float(np.abs(chunks[:, p0, 0][..., idx] - chunks[:, p0, 1][..., idx]).mean()) for g, idx in GROUPS.items()
    }
    ro = readouts(chunks, proprio)
    res = {}
    for j, pid in enumerate(prompts):
        eff = {g: float(np.abs(chunks[:, j][..., idx] - chunks[:, p0][..., idx]).mean()) for g, idx in GROUPS.items()}
        ratio = {g: eff[g] / noise[g] if noise[g] > 0 else float("nan") for g in GROUPS}
        dirs = {}
        for key in ("arm_bias", "turn_rate", "forward_vel", "total_motion"):
            delta = (ro[key][:, j] - ro[key][:, p0]).mean(-1)  # paired per observation, averaged over seeds
            scale = np.abs(ro[key][:, p0, 0] - ro[key][:, p0, 1]).mean()  # noise-induced spread of the readout
            dirs[key] = {
                "mean_delta": float(delta.mean()),
                "frac_pos": float((delta > 0).mean()),
                "p": sign_test_p(delta),
                "noise": float(scale),
                "vec": delta,
            }
        res[pid] = {"ratio": ratio, "dirs": dirs, "text": str(d["prompt_text"][j])}
    return str(d["model"]), n, noise, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    models = [analyze(p) for p in args.inputs]
    css = (
        "<style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#fafafa;"
        "color:#222}h2{margin-top:32px;border-bottom:2px solid #ddd}td,th{padding:3px 8px;text-align:right}"
        "table{border-collapse:collapse;margin:8px 0}tr:nth-child(even){background:#eee}td.l,th.l{text-align:left}"
        ".yes{color:#1e8449;font-weight:600}.no{color:#b03a2e}.hi{background:#d5f5e3}</style>"
    )
    p = [
        f"<!doctype html><meta charset=utf-8><title>Counterfactual prompt probe</title>{css}",
        "<h1>Counterfactual prompt probe: does the prompt change predicted actions?</h1>",
        "<p>Same recorded observation, same diffusion noise, different prompt. <b>Effect ratio</b> = mean action "
        "change caused by the prompt / mean change caused by resampling noise under the original prompt "
        "(1 = indistinguishable from noise). Arms = mean of left_arm and right_arm ratios. Directional tests are "
        "paired over observations (sign test).</p>",
    ]
    for model, n, noise, res in models:
        p.append(f"<h2>{html.escape(model)} ({n} observations)</h2>")
        p.append(
            "<table><tr><th class=l>prompt</th><th class=l>text</th>"
            + "".join(f"<th>{g}</th>" for g in GROUPS)
            + "<th>expected effect</th><th>mean delta (noise spread)</th><th>% obs in expected dir</th>"
            "<th>p</th><th>followed?</th></tr>"
        )
        for pid, r in res.items():
            if pid == "P0":
                continue
            cells = "".join(f"<td class={'hi' if r['ratio'][g] > 1.5 else ''}>{r['ratio'][g]:.2f}</td>" for g in GROUPS)
            exp = EXPECT.get(pid)
            if exp:
                key, sgn = exp
                dd = r["dirs"][key]
                frac = dd["frac_pos"] if sgn > 0 else 1 - dd["frac_pos"]
                spec_txt, ok = specific(res, pid, key, sgn)
                verdict = (
                    f"<span class={'yes' if ok else 'no'}>{'yes' if ok else 'no'}</span> <small>{spec_txt}</small>"
                )
                extra = (
                    f"<td class=l>{key} {'up' if sgn > 0 else 'down'}</td><td>{dd['mean_delta']:+.3f} "
                    f"({dd['noise']:.3f})</td><td>{100 * frac:.0f}%</td><td>{dd['p']:.3g}</td><td>{verdict}</td>"
                )
            else:
                extra = "<td class=l>-</td><td></td><td></td><td></td><td></td>"
            p.append(f"<tr><td class=l>{pid}</td><td class=l>{html.escape(r['text'])}</td>{cells}{extra}</tr>")
        p.append(
            "</table><p><b>Opposite-instruction contrasts</b> (first minus second; expected &gt; 0). These cancel any "
            "generic effect of adding words.</p><table><tr><th class=l>contrast</th><th>readout</th><th>mean</th>"
            "<th>% obs &gt; 0</th><th>p</th><th>followed?</th></tr>"
        )
        for a, b, key, label in PAIRS:
            c = pair_contrast(res, a, b, key)
            if c is None:
                continue
            ok = c[0] > 0 and c[2] < 0.05
            p.append(
                f"<tr><td class=l>{label}</td><td>{key}</td><td>{c[0]:+.3f}</td><td>{100 * c[1]:.0f}%</td>"
                f"<td>{c[2]:.2g}</td><td><span class={'yes' if ok else 'no'}>{'yes' if ok else 'no'}</span></td></tr>"
            )
        p.append(
            "</table><p>Noise floor (mean |action difference| between two noise samples, P0): "
            + ", ".join(f"{g} {v:.3f}" for g, v in noise.items())
            + "</p>"
        )
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write("\n".join(p))
    # Plain-text summary for logs / chat.
    for model, n, noise, res in models:
        print(f"== {model} ({n} obs)")
        for pid, r in res.items():
            if pid == "P0":
                continue
            arms = (r["ratio"]["left_arm"] + r["ratio"]["right_arm"]) / 2
            line = f"  {pid:15s} arms x{arms:.2f} base x{r['ratio']['base']:.2f}"
            if pid in EXPECT:
                key, sgn = EXPECT[pid]
                dd = r["dirs"][key]
                line += f" | {key} {'+' if sgn > 0 else '-'} expected: delta {dd['mean_delta']:+.3f} p={dd['p']:.3g}"
                spec_txt, ok = specific(res, pid, key, sgn)
                line += f" | {spec_txt} -> {'FOLLOWED' if ok else 'not followed'}"
            print(line)
        for a, b, key, label in PAIRS:
            c = pair_contrast(res, a, b, key)
            if c is not None:
                print(f"  pair {label:28s} {key}: {c[0]:+.3f} ({100 * c[1]:.0f}% obs > 0) p={c[2]:.2g}")


if __name__ == "__main__":
    main()
