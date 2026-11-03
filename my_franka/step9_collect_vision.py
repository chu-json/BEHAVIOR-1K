"""
Step 9: collect WRIST-CAMERA demonstrations for a vision policy, with several scenes per sim process.

Same expert, layouts and DART noise as step 5, but each step also records the 128x128 wrist RGB image and a
proprio vector WITHOUT the apple position (the policy has to find the apple in the image):

    state (7) = eef xyz (base frame), 2 finger joints, previous gripper command, is_grasping
    action (4) = dx, dy, dz scaled to [-1, 1] (x MAX_STEP = metres), gripper (+1 open / -1 close)

Parallelism: one process runs NUM_ENVS copies of the world in one Isaac Sim instance (shares the ~4 GB app
overhead), and the launcher runs several such processes. Each process exits after --episodes-per-launch
episodes and is relaunched: long-lived sim processes slow down and leak memory.

Launch:  python my_franka/step9_collect_vision.py --workers 2 --target 300
Output:  my_franka/data/vision/demos_w{worker}_c{chunk}.hdf5 (one group per episode, successes AND failures)
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "vision")
IMG_RES = 128
NUM_ENVS = 4
SETTLE_STEPS = 40
NOISE_FRACTION, NOISE_STD = 0.5, 0.002
BASE_SEED = 100_000_000  # disjoint from state-based demos (0..), eval (50M), check (60M), RL (70M)


def build_vision_cfg(num_envs):
    from pick_common import build_cfg

    cfg = build_cfg()
    cfg["env"]["num_envs"] = num_envs
    robot = cfg["robots"][0]
    robot["obs_modalities"] = ["rgb"]
    robot["sensor_config"] = {"VisionSensor": {"sensor_kwargs": {"image_height": IMG_RES, "image_width": IMG_RES}}}
    return cfg


def wrist_rgb(env, i):
    cam = list(env.scenes[i].robots[0].sensors.values())[0]
    return cam.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype("uint8")


def proprio(s):
    import torch as th

    return th.cat([s["eef"], s["fingers"], th.tensor([s["prev_grip"], s["grasping"]])]).float().numpy()


def run_worker(worker_id, chunk, episodes, num_envs):
    import h5py
    import numpy as np
    import torch as th

    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    from pick_common import (
        MAX_EPISODE_STEPS, MAX_STEP, OPEN, Expert, apple_z, check_success, get_state, place_layout,
        robot_and_apple, sample_layout, to_env_action,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"demos_w{worker_id}_c{chunk}.hdf5")
    part_path = out_path + ".part"  # renamed when complete, so readers never see a file that's still being written
    env = og.Environment(configs=build_vision_cfg(num_envs))
    f = h5py.File(part_path, "w")

    next_ep = 0  # episodes started by this launch

    class Slot:  # per-scene episode bookkeeping
        pass

    def start_episode(i):
        nonlocal next_ep
        sl = slots[i]
        sl.ep = next_ep
        next_ep += 1
        sl.seed = BASE_SEED + worker_id * 1_000_000 + chunk * 10_000 + sl.ep
        rng = np.random.default_rng(sl.seed)
        sl.layout = sample_layout(rng)
        sl.noisy = rng.random() < NOISE_FRACTION
        sl.rng = rng
        place_layout(env, sl.layout, i)
        robot, _ = robot_and_apple(env, i)
        pos, quat = robot.get_relative_eef_pose()
        sl.hold = th.cat([pos, T.quat2axisangle(quat), th.tensor([OPEN])])
        sl.settle, sl.t, sl.prev_grip = SETTLE_STEPS, 0, 1.0
        sl.expert, sl.imgs, sl.states, sl.acts = Expert(), [], [], []

    def finish_episode(i):
        sl = slots[i]
        success, lifted = check_success(env, sl.rest_z, i)
        g = f.create_group(f"ep_{sl.ep:05d}")
        g.create_dataset("images", data=np.stack(sl.imgs), compression="lzf", chunks=(1, IMG_RES, IMG_RES, 3))
        g.create_dataset("state", data=np.stack(sl.states))
        g.create_dataset("action", data=np.stack(sl.acts))
        g.attrs.update(success=bool(success), lifted=lifted, seed=sl.seed, noisy=bool(sl.noisy), steps=len(sl.acts),
                       robot_pos=sl.layout["robot_pos"], robot_yaw=sl.layout["robot_yaw"],
                       apple_pos=sl.layout["apple_pos"])
        f.flush()
        stats["done"] += 1
        stats["ok"] += bool(success)
        rate = stats["done"] / (time.time() - t0) * 60
        print(f"[w{worker_id}c{chunk}] ep {sl.ep} scene {i} success={bool(success)} steps={len(sl.acts)} "
              f"| ok {stats['ok']}/{stats['done']} | {rate:.1f} ep/min", flush=True)

    slots = [Slot() for _ in range(num_envs)]
    stats = {"done": 0, "ok": 0}
    for i in range(num_envs):
        start_episode(i)
    active = [True] * num_envs
    t0 = time.time()
    while any(active):
        actions = []
        for i in range(num_envs):
            sl = slots[i]
            if not active[i] or sl.settle > 0:
                actions.append(sl.hold)
                continue
            s = get_state(env, sl.prev_grip, i)
            a = sl.expert.act(s)
            sl.imgs.append(wrist_rgb(env, i))  # image rendered after the previous step = current state
            sl.states.append(proprio(s))
            sl.acts.append(np.concatenate([a[:3].numpy() / MAX_STEP, a[3:].numpy()]).astype(np.float32))
            executed = a.clone()
            if sl.noisy and th.norm(a[:3]) > 0:
                executed[:3] += th.from_numpy(sl.rng.normal(0, NOISE_STD, 3)).float()
            actions.append(to_env_action(env, s, executed))
            sl.prev_grip = float(executed[3])
            sl.t += 1
        env.step(th.stack(actions))
        for i in range(num_envs):
            sl = slots[i]
            if not active[i]:
                continue
            if sl.settle > 0:
                sl.settle -= 1
                if sl.settle == 0:
                    sl.rest_z = apple_z(env, i)
                continue
            if sl.expert.done or sl.t >= MAX_EPISODE_STEPS:
                finish_episode(i)
                if next_ep < episodes:
                    start_episode(i)
                else:
                    active[i] = False
    f.close()
    os.rename(part_path, out_path)
    og.shutdown()


def count_successes():
    import h5py

    ok = 0
    for path in glob.glob(os.path.join(OUT_DIR, "demos_w*_c*.hdf5")):
        try:
            with h5py.File(path, "r") as f:
                ok += sum(bool(f[k].attrs.get("success", False)) for k in f.keys())
        except OSError:
            pass  # file of a launch that is still running / was killed
    return ok


def launch(workers, target, episodes_per_launch, num_envs):
    """Keep `workers` sim processes running, relaunching each with a new chunk id, until `target` successes."""
    os.makedirs(OUT_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(OUT_DIR, "demos_w*_c*.hdf5*"))
    next_chunk = {w: 1 + max([int(p.split("_c")[-1].split(".")[0]) for p in existing if f"_w{w}_" in p] + [-1])
                  for w in range(workers)}
    procs = {}
    while True:
        ok = count_successes()
        if ok >= target and not procs:
            break
        for w in range(workers):
            p = procs.get(w)
            if p is not None and p.poll() is None:
                continue
            if p is not None:
                print(f"worker {w} chunk {next_chunk[w] - 1} exited with code {p.returncode}", flush=True)
                del procs[w]
            if ok < target:
                c = next_chunk[w]
                next_chunk[w] += 1
                log = open(os.path.join(OUT_DIR, f"collect_w{w}.log"), "a")
                cmd = [sys.executable, os.path.abspath(__file__), "--worker-id", str(w), "--chunk", str(c),
                       "--episodes", str(episodes_per_launch), "--num-envs", str(num_envs)]
                procs[w] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                                            env=dict(os.environ, OMNIGIBSON_HEADLESS="1"))
                print(f"launched worker {w} chunk {c} (pid {procs[w].pid}); successes so far {ok}/{target}", flush=True)
        time.sleep(30)
    print(f"DONE: {count_successes()} successful episodes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None, help="launcher mode: number of sim processes")
    parser.add_argument("--target", type=int, default=300, help="launcher mode: stop at this many successes")
    parser.add_argument("--episodes-per-launch", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()
    if args.workers:
        launch(args.workers, args.target, args.episodes_per_launch, args.num_envs)
    else:
        run_worker(args.worker_id, args.chunk, args.episodes, args.num_envs)
