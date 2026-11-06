"""Overnight prompt-steerability sweep on turning_on_radio (run detached; see steering/NOTES.md).

For each group (checkpoint x prompt x seed) in the requested order: start the Comet server with that checkpoint
and prompt, run the 5 setups with run_rollouts.py, stop the server, rebuild results.csv / index.html. A bonus pass
(seed 1) starts a group only if it can finish before --deadline.

    setsid nohup systemd-inhibit --what=sleep:idle --who=steering --why="prompt sweep" --mode=block \
        python3 steering/prompt_sweep.py --hours 8 > /home/behavior/ext/logs/prompt_sweep.log 2>&1 &
"""

import argparse
import json
import os
import subprocess
import time
import urllib.request

REPO = "/home/behavior/BEHAVIOR-1K"
EXT = "/home/behavior/ext"
LOGS = f"{EXT}/logs"
ROOT = f"{EXT}/runs/prompt_sweep"
CONDA = os.path.expanduser("~/miniconda3/condabin/conda")
SETUPS = [242, 203, 109, 181, 187]

BASE_PROMPT = "Turn on the radio receiver that's on the table in the living room."
STEM = BASE_PROMPT.rstrip(".")
PROMPTS = {
    "P0": BASE_PROMPT,
    "P1": "Switch on the radio receiver sitting on the living room table.",
    "P2": STEM + " using your left hand.",
    "P3": STEM + " using your right hand.",
    "P4": STEM + " without picking the radio up.",
}
CKPTS = {
    "A_radio_sft": {
        "ckpt": f"{EXT}/ckpt/pi05-turning_on_radio-sft",
        "config": "pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft",
    },
    "B_pt50": {
        "ckpt": f"{EXT}/ckpt/openpi_comet/pi05-b1kpt50-cs32",
        "config": "pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k",
    },
}
ORDER = ["P0", "P2", "P3", "P4", "P1"]
# Conservative per-group time (s) used for the deadline check: 3.5 min startup + 5 rollouts x 6 min (all fail).
GROUP_BUDGET_S = 3.5 * 60 + 5 * 6 * 60


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def healthy():
    try:
        return urllib.request.urlopen("http://localhost:8000/healthz", timeout=2).status == 200
    except Exception:
        return False


def start_server(ck, prompt, name):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.32")
    cmd = [
        ".venv/bin/python",
        f"{REPO}/steering/comet_server.py",
        "--ckpt",
        CKPTS[ck]["ckpt"],
        "--config",
        CKPTS[ck]["config"],
        "--prompt",
        prompt,
        "--port",
        "8000",
    ]
    out = open(f"{LOGS}/sweep_server_{name}.log", "w")
    proc = subprocess.Popen(cmd, cwd=f"{EXT}/openpi-comet", env=env, stdout=out, stderr=subprocess.STDOUT)
    for _ in range(240):
        if healthy():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup, see {LOGS}/sweep_server_{name}.log")
        time.sleep(2)
    proc.kill()
    raise RuntimeError("server did not become healthy")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    for _ in range(30):
        if not healthy():
            return
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--max-steps", type=int, default=3000)
    args = ap.parse_args()
    deadline = time.time() + args.hours * 3600
    os.makedirs(ROOT, exist_ok=True)

    groups = []
    for pas, seed in (("main (seed 0)", 0), ("bonus (seed 1)", 1)):
        for pid in ORDER:
            for ck in CKPTS:
                groups.append(
                    {
                        "name": f"{ck}_{pid}_s{seed}",
                        "checkpoint": ck,
                        "prompt_id": pid,
                        "prompt": PROMPTS[pid],
                        "seed": seed,
                        "pass": pas,
                    }
                )
    json.dump(groups, open(f"{ROOT}/groups.json", "w"), indent=1)

    watchdog = subprocess.Popen(["bash", f"{REPO}/steering/gpu_watchdog.sh", "1000", f"{LOGS}/gpu_watchdog.log"])
    try:
        for g in groups:
            if g["seed"] != 0 and time.time() + GROUP_BUDGET_S > deadline:
                log(f"skip {g['name']}: would not finish before the deadline")
                continue
            if os.path.exists(f"{ROOT}/{g['name']}/rollouts.csv"):
                log(f"skip {g['name']}: already has results")
                continue
            log(f"=== {g['name']}: {g['prompt']!r}")
            t0 = time.time()
            try:
                server = start_server(g["checkpoint"], g["prompt"], g["name"])
            except RuntimeError as e:
                log(f"ERROR {g['name']}: {e}")
                continue
            try:
                env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES")
                cmd = [
                    CONDA,
                    "run",
                    "--no-capture-output",
                    "-n",
                    "behavior",
                    "python",
                    "steering/run_rollouts.py",
                    "--instances",
                    *map(str, SETUPS),
                    "--seeds",
                    *([str(g["seed"])] * len(SETUPS)),
                    "--max-steps",
                    str(args.max_steps),
                    "--resolution",
                    "full",
                    "--trace",
                    "--base-mass",
                    "default",
                    "--full-scene",
                    "--output-dir",
                    f"{ROOT}/{g['name']}",
                ]
                with open(f"{LOGS}/sweep_{g['name']}.log", "w") as out:
                    rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
            finally:
                stop_server(server)
            log(f"{g['name']} exit {rc} in {(time.time() - t0) / 60:.1f} min")
            subprocess.call(
                [CONDA, "run", "-n", "behavior", "python", "-I", "steering/sweep_report.py", "--root", ROOT],
                cwd=REPO,
                stdout=open(f"{LOGS}/sweep_report.log", "a"),
                stderr=subprocess.STDOUT,
            )
    finally:
        watchdog.terminate()
    log("sweep done")


if __name__ == "__main__":
    main()
