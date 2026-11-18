"""Can base pi0.5 do better without training? Ablation of three fixes (see steering/NOTES.md, 2026-11-17):
  delta   joint actions as offsets from the current pose (base pi0.5's pre-training convention; delta norm stats)
  torso   hold the torso at its start pose (base pi0.5 folds it until the head camera sees the floor)
  approach scripted drive to 0.8 m from the radio, facing it, before handing over (shared autonomy)
Resumable (skips conditions with results). Run detached:

    setsid nohup systemd-inhibit --what=sleep:idle --who=steering --why="fix campaign" --mode=block \
        python3 steering/fix_campaign.py > /home/behavior/ext/logs/fix_campaign.log 2>&1 &
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from day_campaign import CONDA, EXT, LOGS, REPO, healthy, log, stop_server  # noqa: E402
from probe_prompts import PROMPTS  # noqa: E402

ROOT = f"{EXT}/runs/fix"
SETUPS = [242, 203, 109, 181, 187]
ABS = f"{EXT}/ckpt/pi05_base_b1k"
DELTA = f"{EXT}/ckpt/pi05_base_b1k_delta"
CONDITIONS = [
    # name, ckpt, server flags, run_rollouts flags
    ("C0_abs", ABS, [], []),
    ("C1_delta", DELTA, ["--delta-joints"], []),
    ("C2_delta_torso", DELTA, ["--delta-joints", "--freeze-torso"], []),
    ("C3_delta_torso_approach", DELTA, ["--delta-joints", "--freeze-torso"], ["--approach-assist", "0.8"]),
    ("C4_abs_torso_approach", ABS, ["--freeze-torso"], ["--approach-assist", "0.8"]),
]


# Best-of-N steering with a privileged FK critic (V-GPS / RoboMonkey style selection; see NOTES.md).
CONDITIONS_BEST = [
    (
        "C5_delta_torso_approach_best8",
        DELTA,
        ["--delta-joints", "--freeze-torso", "--best-of-n", "8"],
        ["--approach-assist", "0.8", "--inject-target"],
    ),
    (
        "C6_abs_torso_approach_best8",
        ABS,
        ["--freeze-torso", "--best-of-n", "8"],
        ["--approach-assist", "0.8", "--inject-target"],
    ),
]


def start(ckpt, flags, name, prompt):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.32")
    cmd = [
        ".venv/bin/python",
        f"{REPO}/steering/comet_server.py",
        "--ckpt",
        ckpt,
        "--config",
        "pi05_b1k-base",
        "--prompt",
        prompt,
        "--port",
        "8000",
        *flags,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=f"{EXT}/openpi-comet",
        env=env,
        stdout=open(f"{LOGS}/fix_server_{name}.log", "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(240):
        if healthy():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"server exited, see {LOGS}/fix_server_{name}.log")
        time.sleep(2)
    proc.kill()
    raise RuntimeError("server not healthy")


def run(name, ckpt, sflags, rflags, prompt_id="P0", steps=1500, seeds=(0,)):
    out = f"{ROOT}/{name}_{prompt_id}"
    if os.path.exists(f"{out}/rollouts.csv"):
        log(f"skip {name}_{prompt_id}: has results")
        return
    log(f"=== {name}_{prompt_id}: server {sflags}, rollouts {rflags}")
    t0 = time.time()
    try:
        server = start(ckpt, sflags, f"{name}_{prompt_id}", PROMPTS[prompt_id])
    except RuntimeError as e:
        log(f"ERROR {name}: {e}")
        return
    try:
        inst = [str(s) for s in SETUPS for _ in seeds]
        sd = [str(x) for _ in SETUPS for x in seeds]
        cmd = [
            CONDA,
            "run",
            "--no-capture-output",
            "-n",
            "behavior",
            "python",
            "steering/run_rollouts.py",
            "--instances",
            *inst,
            "--seeds",
            *sd,
            "--max-steps",
            str(steps),
            "--resolution",
            "full",
            "--trace",
            "--base-mass",
            "default",
            "--full-scene",
            "--output-dir",
            out,
            *rflags,
        ]
        env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES")
        with open(f"{LOGS}/fix_{name}_{prompt_id}.log", "w") as f:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    finally:
        stop_server(server)
    log(f"{name}_{prompt_id} exit {rc} in {(time.time() - t0) / 60:.1f} min")


def main():
    os.makedirs(ROOT, exist_ok=True)
    watchdog = subprocess.Popen(["bash", f"{REPO}/steering/gpu_watchdog.sh", "1000", f"{LOGS}/gpu_watchdog.log"])
    try:
        for name, ckpt, sflags, rflags in CONDITIONS + (CONDITIONS_BEST if "--best" in sys.argv else []):
            run(name, ckpt, sflags, rflags)
    finally:
        watchdog.terminate()
    log("fix campaign done")


if __name__ == "__main__":
    main()
