"""Overnight steering campaign on Comet (radio SFT), no fine-tuning (see steering/NOTES.md, 2026-11-19).

Conditions (same 20 (instance, seed) pairs each: the 10 2025-validation instances x seeds 0-19, as massD/massF):
  S1 best-of-8 chunk selection, privileged critic (FK hand path vs true button position)
  R1 failure monitor (stall / radio pushed) -> arms home, back off, re-approach, retry (oracle-triggered)
  V1 visual prompt: radio pixels tinted red in all 3 cameras (simulator segmentation)
  S2 best-of-8 with a sensor-only critic target (radio position from head depth + segmentation)
  V2 clutter removal: everything but the radio and the robot darkened
  N1 flow-matching initial noise x0.5
  B0 fresh baseline (reference: massD+massF = 14/20 with the same pairs)
Resumable: each condition appends to its rollouts.csv and the runner skips completed (instance, seed) pairs.
The run order lives in runs/steer/queue.json (created from CONDITIONS; editable while running).
Run detached:

    setsid nohup systemd-inhibit --what=sleep:idle --who=steering --why="steer campaign" --mode=block \
        python3 steering/steer_campaign.py > /home/behavior/ext/logs/steer_campaign.log 2>&1 &

`--smoke` runs three short checks instead (output under runs/steer_smoke).
"""

import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day_campaign import CONDA, EXT, LOGS, REPO, healthy, log, stop_server  # noqa: E402

CKPT = f"{EXT}/ckpt/pi05-turning_on_radio-sft"
CONFIG = "pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft"
VAL = [242, 295, 211, 203, 109, 181, 197, 187, 214, 139]
PAIRS = [(VAL[s % 10], s) for s in range(20)]
STEPS = 3000

CONDITIONS = [
    # name, server flags, run_rollouts flags
    ("S1_best8_oracle", ["--best-of-n", "8"], ["--inject-target"]),
    ("R1_recovery", [], ["--recovery"]),
    ("V1_tint", [], ["--overlay", "tint"]),
    ("S2_best8_depth", ["--best-of-n", "8"], ["--depth-target"]),
    ("V2_dim", [], ["--overlay", "dim"]),
    ("N1_noise05", ["--noise-scale", "0.5"], []),
    ("B0_baseline", [], []),
]
SMOKE = [  # name, server flags, rollout flags, pairs, steps, extra env
    ("box_tint_depth_best4", ["--best-of-n", "4"], ["--overlay", "tint", "--depth-target"], [(211, 2)], 400, {}),
    ("box_dim", [], ["--overlay", "dim"], [(242, 0)], 60, {}),
    ("recovery_forced2", [], ["--recovery"], [(211, 2)], 2200, {"STEER_RECOVERY_STALL": "150"}),
]


COMET_SERVER = {
    "cmd": [
        ".venv/bin/python",
        f"{REPO}/steering/comet_server.py",
        "--ckpt",
        CKPT,
        "--config",
        CONFIG,
        "--port",
        "8000",
    ],
    "cwd": f"{EXT}/openpi-comet",
}


def start(flags, name, server=None):
    server = server or COMET_SERVER
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.32")
    proc = subprocess.Popen(
        server["cmd"] + flags,
        cwd=server["cwd"],
        env=env,
        stdout=open(f"{LOGS}/steer_server_{name}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(240):
        if healthy():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"server exited, see {LOGS}/steer_server_{name}.log")
        time.sleep(2)
    proc.kill()
    raise RuntimeError("server not healthy")


def done_pairs(out):
    p = f"{out}/rollouts.csv"
    if not os.path.exists(p):
        return set()
    return {(int(r["instance_id"]), int(r["seed"])) for r in csv.DictReader(open(p))}


def run(name, sflags, rflags, pairs, steps, root, extra_env=None, server=None):
    out = f"{root}/{name}"
    todo = [pr for pr in pairs if pr not in done_pairs(out)]
    if not todo:
        log(f"skip {name}: complete")
        return
    log(f"=== {name}: {len(todo)} rollouts, server {sflags}, rollouts {rflags}")
    t0 = time.time()
    try:
        server = start(sflags, name, server)
    except RuntimeError as e:
        log(f"ERROR {name}: {e}")
        return
    try:
        cmd = [CONDA, "run", "--no-capture-output", "-n", "behavior", "python", "steering/run_rollouts.py"]
        cmd += ["--instances", *[str(i) for i, _ in todo], "--seeds", *[str(s) for _, s in todo]]
        cmd += ["--max-steps", str(steps), "--resolution", "full", "--trace", "--base-mass", "default"]
        cmd += ["--full-scene", "--output-dir", out, *rflags]
        env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", **(extra_env or {}))
        with open(f"{LOGS}/steer_{name}.log", "a") as f:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    finally:
        stop_server(server)
    n = len(done_pairs(out))
    log(f"{name} exit {rc} in {(time.time() - t0) / 60:.1f} min ({n}/{len(pairs)} done)")


def main():
    smoke = "--smoke" in sys.argv
    root = f"{EXT}/runs/{'steer_smoke' if smoke else 'steer'}"
    os.makedirs(root, exist_ok=True)
    watchdog = subprocess.Popen(["bash", f"{REPO}/steering/gpu_watchdog.sh", "1000", f"{LOGS}/gpu_watchdog.log"])
    try:
        if smoke:
            for name, sflags, rflags, pairs, steps, extra in SMOKE:
                run(name, sflags, rflags, pairs, steps, root, extra)
        else:
            # The queue file is re-read before every condition, so entries can be added/reordered while this runs.
            # Entry: {"name", "sflags", "rflags", optional "server": {"cmd": [...], "cwd": ...}, "env": {...}}
            qpath = f"{root}/queue.json"
            if not os.path.exists(qpath):
                json.dump(
                    [{"name": n, "sflags": sf, "rflags": rf} for n, sf, rf in CONDITIONS], open(qpath, "w"), indent=1
                )
            tries = {}
            while True:
                queue = json.load(open(qpath))
                todo = [q for q in queue if len(done_pairs(f"{root}/{q['name']}")) < len(PAIRS)]
                todo = [q for q in todo if tries.get(q["name"], 0) < 2]  # at most 2 attempts (crash recovery)
                if not todo:
                    break
                q = todo[0]
                tries[q["name"]] = tries.get(q["name"], 0) + 1
                run(q["name"], q["sflags"], q["rflags"], PAIRS, STEPS, root, q.get("env"), q.get("server"))
    finally:
        watchdog.terminate()
    log("steer campaign done" + (" (smoke)" if smoke else ""))


if __name__ == "__main__":
    main()
