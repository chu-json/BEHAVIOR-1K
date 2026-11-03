"""
Step 5: generate demonstrations with the scripted Expert, in parallel sim processes.

Each episode: random robot pose on the table + random apple pose -> Expert picks the apple -> we record
(obs, expert action) at every step. Only episodes that pass check_success() are marked usable.

Noise injection (DART): in a fraction of episodes we execute expert_action + noise but RECORD the clean
expert action. The robot drifts a little off the ideal path and the data shows how the expert corrects it,
which is exactly what a cloned policy needs when it makes its own small mistakes.

Launch N workers:      python my_franka/step5_collect.py --workers 3 --episodes-per-worker 400
One worker (debug):    OMNIGIBSON_HEADLESS=1 python my_franka/step5_collect.py --worker-id 0 --episodes 5
Output: my_franka/data/demos_w{id}.hdf5, one group per episode.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

NOISE_FRACTION = 0.5  # fraction of episodes with action noise
NOISE_STD = 0.002  # metres of noise per step on the executed position delta


def run_worker(worker_id, episodes, base_seed):
    import h5py
    import numpy as np
    import torch as th

    import omnigibson as og
    from pick_common import (
        MAX_EPISODE_STEPS, Expert, apple_z, apply_layout, build_cfg, check_success, get_obs, get_state,
        sample_layout, to_env_action,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"demos_w{worker_id}.hdf5")
    env = og.Environment(configs=build_cfg())

    with h5py.File(out_path, "a") as f:
        start_ep = len(f.keys())  # resume: keep appending if the file already has episodes
        n_ok = sum(bool(f[k].attrs["success"]) for k in f.keys())
        t0 = time.time()
        for ep in range(start_ep, start_ep + episodes):
            seed = base_seed + worker_id * 1_000_000 + ep
            rng = np.random.default_rng(seed)
            layout = sample_layout(rng)
            noisy = rng.random() < NOISE_FRACTION

            apply_layout(env, layout)
            rest_z = apple_z(env)
            expert, obs_buf, act_buf, prev_grip = Expert(), [], [], 1.0
            for _ in range(MAX_EPISODE_STEPS):
                s = get_state(env, prev_grip)
                a = expert.act(s)
                obs_buf.append(get_obs(s).numpy())
                act_buf.append(a.numpy())
                executed = a.clone()
                if noisy and th.norm(a[:3]) > 0:  # perturb only while the expert is moving the arm
                    executed[:3] += th.from_numpy(rng.normal(0, NOISE_STD, 3)).float()
                env.step(to_env_action(env, s, executed))
                prev_grip = float(executed[3])
                if expert.done:
                    break
            success, lifted = check_success(env, rest_z)
            n_ok += success

            g = f.create_group(f"ep_{ep:06d}")
            g.create_dataset("obs", data=np.array(obs_buf, dtype=np.float32))
            g.create_dataset("act", data=np.array(act_buf, dtype=np.float32))
            g.attrs.update(
                success=success, lifted=lifted, seed=seed, noisy=noisy, steps=len(obs_buf),
                final_phase=expert.phase, robot_pos=layout["robot_pos"], robot_yaw=layout["robot_yaw"],
                apple_pos=layout["apple_pos"],
            )
            f.flush()
            rate = (ep - start_ep + 1) / (time.time() - t0)
            print(f"[w{worker_id}] ep {ep} success={success} lifted={lifted * 100:.1f}cm steps={len(obs_buf)} "
                  f"phase={expert.phase} | ok {n_ok}/{ep + 1} | {rate * 60:.1f} ep/min", flush=True)
    og.shutdown()


def launch(workers, episodes_per_worker, base_seed):
    """Start @workers headless sim processes, each writing its own HDF5 file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    procs = []
    for w in range(workers):
        log = open(os.path.join(DATA_DIR, f"collect_w{w}.log"), "a")
        env = dict(os.environ, OMNIGIBSON_HEADLESS="1")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker-id", str(w),
               "--episodes", str(episodes_per_worker), "--seed", str(base_seed)]
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE))
        print(f"started worker {w} (pid {procs[-1].pid}), log: data/collect_w{w}.log", flush=True)
    for p in procs:
        p.wait()
    print("all workers finished:", [p.returncode for p in procs])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None, help="launch this many parallel workers")
    parser.add_argument("--episodes-per-worker", type=int, default=400)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.workers:
        launch(args.workers, args.episodes_per_worker, args.seed)
    else:
        run_worker(args.worker_id, args.episodes, args.seed)
