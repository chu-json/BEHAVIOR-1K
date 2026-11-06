"""Unattended multi-experiment campaign: base pi0.5 vs. Comet steerability (see steering/NOTES.md).

Jobs run in order; each rollout job = (model, prompt, setups, seeds, max_steps), served by a fresh comet_server.py,
optionally dumping observations for the offline prompt probe. Probe jobs run steering/prompt_probe.py (GPU, no sim).
A job is skipped if its output already exists, so the script can be re-launched to resume.

    setsid nohup systemd-inhibit --what=sleep:idle --who=steering --why="day campaign" --mode=block \
        python3 steering/day_campaign.py > /home/behavior/ext/logs/day_campaign.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_prompts import MODELS, PROMPTS  # noqa: E402

REPO = "/home/behavior/BEHAVIOR-1K"
EXT = "/home/behavior/ext"
LOGS = f"{EXT}/logs"
DAY = f"{EXT}/runs/day"
CONDA = os.path.expanduser("~/miniconda3/condabin/conda")
SETUPS5 = [242, 203, 109, 181, 187]
SETUPS3 = [242, 203, 187]
SWEEP = ["P0", "P2", "P3", "P4", "P1"]
MOTOR = ["P0", "M_left_arm_up", "M_right_arm_up", "M_turn_left", "M_turn_right", "M_forward", "M_still", "X_nonsense"]


def rollout_job(model, pid, setups, seeds, max_steps, root, dump=False):
    name = f"{model}_{pid}_s{'-'.join(map(str, sorted(set(seeds))))}"
    return {
        "kind": "rollout",
        "name": name,
        "root": root,
        "model": model,
        "prompt_id": pid,
        "setups": setups,
        "seeds": seeds,
        "max_steps": max_steps,
        "dump": dump,
    }


def jobs():
    J = []
    base_sweep = f"{DAY}/base_sweep"
    # 1. Base pi0.5, original prompt, 5 setups x 2 seeds (the "basic rollouts"), with observation dumps.
    J.append(rollout_job("C_pi05_base", "P0", SETUPS5 * 2, [0] * 5 + [1] * 5, 3000, base_sweep, dump=True))
    # 2. Comet A on the same setups, observations only (states where the task is going well).
    J.append(rollout_job("A_radio_sft", "P0", SETUPS5, [2] * 5, 3000, f"{DAY}/obs_record", dump=True))
    # 3. Counterfactual prompt probe on both observation sets, for all three models.
    obs_sets = {
        "cometstates": f"{DAY}/obs_record/A_radio_sft_P0_s2/obs",  # states along successful Comet rollouts
        "basestates": f"{base_sweep}/C_pi05_base_P0_s0-1/obs",  # states the base model wanders into
    }
    for oset, odir in obs_sets.items():
        for m in ("C_pi05_base", "A_radio_sft", "B_pt50"):
            J.append(
                {"kind": "probe", "name": f"probe_{m}_{oset}", "model": m, "obs": [odir], "label": f"{m} on {oset}"}
            )
    # 4. Base pi0.5: rest of the exact overnight sweep.
    for pid in SWEEP[1:]:
        J.append(rollout_job("C_pi05_base", pid, SETUPS5 * 2, [0] * 5 + [1] * 5, 3000, base_sweep))
    # 5. Closed-loop motor-command prompts, base and Comet A (shorter episodes: behavior shows early).
    for m in ("C_pi05_base", "A_radio_sft"):
        for pid in MOTOR:
            J.append(rollout_job(m, pid, SETUPS3, [0] * 3, 1000, f"{DAY}/motor"))
    return J


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def healthy():
    try:
        return urllib.request.urlopen("http://localhost:8000/healthz", timeout=2).status == 200
    except Exception:
        return False


def start_server(model, prompt, name, dump_dir=None):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.32")
    cmd = [
        ".venv/bin/python",
        f"{REPO}/steering/comet_server.py",
        "--ckpt",
        MODELS[model]["ckpt"],
        "--config",
        MODELS[model]["config"],
        "--prompt",
        prompt,
        "--port",
        "8000",
    ]
    if dump_dir:
        cmd += ["--dump-obs", dump_dir]
    out = open(f"{LOGS}/day_server_{name}.log", "w")
    proc = subprocess.Popen(cmd, cwd=f"{EXT}/openpi-comet", env=env, stdout=out, stderr=subprocess.STDOUT)
    for _ in range(240):
        if healthy():
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup, see {LOGS}/day_server_{name}.log")
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


def write_groups(root, J):
    groups = [
        {
            "name": j["name"],
            "checkpoint": j["model"],
            "prompt_id": j["prompt_id"],
            "prompt": PROMPTS[j["prompt_id"]],
            "seed": "-".join(map(str, sorted(set(j["seeds"])))),
            "pass": os.path.basename(root),
        }
        for j in J
        if j["kind"] == "rollout" and j["root"] == root
    ]
    os.makedirs(root, exist_ok=True)
    json.dump(groups, open(f"{root}/groups.json", "w"), indent=1)


def run_rollout(j):
    out = f"{j['root']}/{j['name']}"
    if os.path.exists(f"{out}/rollouts.csv"):
        log(f"skip {j['name']}: has results")
        return
    log(
        f"=== {j['name']}: {PROMPTS[j['prompt_id']]!r}, setups {j['setups']}, seeds {j['seeds']}, {j['max_steps']} steps"
    )
    t0 = time.time()
    try:
        server = start_server(j["model"], PROMPTS[j["prompt_id"]], j["name"], f"{out}/obs" if j["dump"] else None)
    except RuntimeError as e:
        log(f"ERROR {j['name']}: {e}")
        return
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
            *map(str, j["setups"]),
            "--seeds",
            *map(str, j["seeds"]),
            "--max-steps",
            str(j["max_steps"]),
            "--resolution",
            "full",
            "--trace",
            "--base-mass",
            "default",
            "--full-scene",
            "--output-dir",
            out,
        ]
        with open(f"{LOGS}/day_{j['name']}.log", "w") as f:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    finally:
        stop_server(server)
    log(f"{j['name']} exit {rc} in {(time.time() - t0) / 60:.1f} min")


def run_probe(j):
    out = f"{DAY}/probe/{j['name']}.npz"
    if os.path.exists(out):
        log(f"skip {j['name']}: has results")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    log(f"=== {j['name']} on {j['obs']}")
    t0 = time.time()
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.45")
    cmd = [
        ".venv/bin/python",
        f"{REPO}/steering/prompt_probe.py",
        "--model",
        j["model"],
        "--obs-dir",
        *j["obs"],
        "--max-obs",
        "150",
        "--out",
        out,
        "--label",
        j["label"],
    ]
    with open(f"{LOGS}/day_{j['name']}.log", "w") as f:
        rc = subprocess.call(cmd, cwd=f"{EXT}/openpi-comet", env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"{j['name']} exit {rc} in {(time.time() - t0) / 60:.1f} min")


def reports():
    for root in (f"{DAY}/base_sweep", f"{DAY}/motor"):
        if os.path.exists(f"{root}/groups.json"):
            subprocess.call(
                [CONDA, "run", "-n", "behavior", "python", "-I", "steering/sweep_report.py", "--root", root],
                cwd=REPO,
                stdout=open(f"{LOGS}/day_reports.log", "a"),
                stderr=subprocess.STDOUT,
            )
    probes = [
        f"{DAY}/probe/probe_{m}_{o}.npz"
        for o in ("cometstates", "basestates")
        for m in ("C_pi05_base", "A_radio_sft", "B_pt50")
    ]
    probes = [p for p in probes if os.path.exists(p)]
    if probes:
        subprocess.call(
            [
                CONDA,
                "run",
                "-n",
                "behavior",
                "python",
                "-I",
                "steering/probe_report.py",
                "--inputs",
                *probes,
                "--out",
                f"{DAY}/probe",
            ],
            cwd=REPO,
            stdout=open(f"{LOGS}/day_reports.log", "a"),
            stderr=subprocess.STDOUT,
        )
    if os.path.exists(f"{REPO}/steering/motor_report.py"):
        subprocess.call(
            [CONDA, "run", "-n", "behavior", "python", "-I", "steering/motor_report.py", "--root", f"{DAY}/motor"],
            cwd=REPO,
            stdout=open(f"{LOGS}/day_reports.log", "a"),
            stderr=subprocess.STDOUT,
        )


def main():
    J = jobs()
    for root in {j["root"] for j in J if j["kind"] == "rollout"}:
        write_groups(root, J)
    watchdog = subprocess.Popen(["bash", f"{REPO}/steering/gpu_watchdog.sh", "1000", f"{LOGS}/gpu_watchdog.log"])
    try:
        for j in J:
            (run_rollout if j["kind"] == "rollout" else run_probe)(j)
            reports()
    finally:
        watchdog.terminate()
    log("day campaign done")


if __name__ == "__main__":
    main()
