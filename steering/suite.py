"""Steerability suite (steering/RESEARCH.md taxonomy) for one model: runs each cell on the same (instance, seed) pairs
with the campaign harness and writes runs/suite/<model>/<cell>/rollouts.csv (+ traces). Resumable.

    python3 steering/suite.py --model released [--cells base,S1_clutter6,...] [--n 10] [--steps 3000]
    python3 steering/suite.py --model ftcont:/path/to/ckpt
    python3 steering/suite.py --model comet          # Comet radio SFT (openpi-comet)
    python3 steering/suite.py --report               # summary table over everything under runs/suite

Cells:
  base           task prompt, unperturbed scene
  L3_para        paraphrase                             (success should be unchanged)
  L2_left/right  "... using your left/right hand"       (which hand ends closest to the button)
  L2_nolift      "... without picking the radio up"     (radio lifted or not)
  L_garbage      unrelated instruction                  (language-blindness check, LIBERO-PRO style)
  S1_clutter6    6 distractors around the radio
  S2_occluder    panel partly hiding the radio from the start view
  S4_yaw90       radio rotated 90 deg (button faces sideways)
"""

import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from steer_campaign import EXT, PAIRS, REPO, run  # noqa: E402

BASE_PROMPT = "Turn on the radio receiver that's on the table in the living room."
STEM = BASE_PROMPT.rstrip(".")
CELLS = {
    "base": (None, []),
    "L3_para": ("Switch on the radio receiver sitting on the living room table.", []),
    "L2_left": (STEM + " using your left hand.", []),
    "L2_right": (STEM + " using your right hand.", []),
    "L2_nolift": (STEM + " without picking the radio up.", []),
    "L_garbage": ("Fold the towel and put it in the laundry basket.", []),
    "S1_clutter6": (None, ["--clutter", "6"]),
    "S2_occluder": (None, ["--occluder"]),
    "S4_yaw90": (None, ["--radio-yaw-deg", "90"]),
}
DEFAULT_ORDER = ["base", "S1_clutter6", "S2_occluder", "S4_yaw90", "L3_para", "L_garbage", "L2_left", "L2_right",
                 "L2_nolift"]  # fmt: skip
ROOT = f"{EXT}/runs/suite"
RELEASED = f"{EXT}/mentor/released_pi05_radio/pi05_turn_on_the_radio"


def server_for(model, prompt):
    extra = ["--prompt", prompt] if prompt else []
    if model == "comet":
        cmd = [".venv/bin/python", f"{REPO}/steering/comet_server.py", "--ckpt", f"{EXT}/ckpt/pi05-turning_on_radio-sft",
               "--config", "pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft", "--port", "8000"]  # fmt: skip
        return {"cmd": cmd + extra, "cwd": f"{EXT}/openpi-comet"}
    ckpt = RELEASED if model == "released" else model.split(":", 1)[1]
    cmd = [".venv/bin/python", f"{REPO}/steering/mentor_server.py", "--ckpt", ckpt, "--port", "8000"]
    return {"cmd": cmd + extra, "cwd": f"{EXT}/mentor/openpi"}


def model_tag(model):
    if ":" not in model:
        return model
    p = model.split(":", 1)[1].rstrip("/")
    return f"{os.path.basename(os.path.dirname(p))}_{os.path.basename(p)}"


def trace_stats(trace):
    """Which hand got closest to the button, and whether the radio was lifted (> 10 cm above its start)."""
    rs = list(csv.DictReader(open(trace)))
    best = {"left": 9.0, "right": 9.0}
    for r in rs:
        for s in best:
            v = r.get(f"finger_button_dist_{s}", "")
            if v not in ("", "nan"):
                best[s] = min(best[s], float(v))
    z = [float(r["radio_z"]) for r in rs if r.get("radio_z") not in (None, "", "nan")]
    lifted = bool(z) and max(z) - z[0] > 0.10
    return min(best, key=best.get), round(min(best.values()), 3), lifted


def report():
    print(f"{'model':28s} {'cell':12s} succ   closest-hand L/R  lifted  med-min-dist")
    for d in sorted(glob.glob(f"{ROOT}/*/*/rollouts.csv")):
        model, cell = d.split("/")[-3], d.split("/")[-2]
        rows = list(csv.DictReader(open(d)))
        hands, lifted, dmins = [], 0, []
        for r in rows:
            t = r.get("trace_path")
            if t and os.path.exists(t):
                h, dm, lf = trace_stats(t)
                hands.append(h)
                dmins.append(dm)
                lifted += lf
        succ = sum(r["success"] == "1" for r in rows)
        dmins.sort()
        med = dmins[len(dmins) // 2] if dmins else ""
        print(f"{model:28s} {cell:12s} {succ:2d}/{len(rows):<2d}  {hands.count('left'):2d}/{hands.count('right'):<2d}"
              f"             {lifted:2d}      {med}")  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="released", help="comet | released | ftcont:<ckpt dir>")
    ap.add_argument("--cells", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--n", type=int, default=10, help="first n of the 20 campaign (instance, seed) pairs")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        return report()
    root = f"{ROOT}/{model_tag(args.model)}"
    for cell in args.cells.split(","):
        prompt, rflags = CELLS[cell]
        run(cell, [], rflags, PAIRS[: args.n], args.steps, root, None, server_for(args.model, prompt))


if __name__ == "__main__":
    main()
