# steering/ — notes

Goal: run Comet (2nd place, 2025 BEHAVIOR Challenge, pi0.5) on `turning_on_radio` and measure success.

## Layout
- This repo: fork of StanfordVL/BEHAVIOR-1K at v3.9.3 (2026 challenge stack, Isaac Sim 5.1, assets 3.9.0).
- Comet code is cloned **outside** the repo: `/home/behavior/ext/openpi-comet` (commit 4bb2aa7).
- All my code lives in `steering/`. No edits to OmniGibson/ or openpi-comet.

## How eval works (fork, `OmniGibson/omnigibson/eval/`)
- `eval.py` -> `BatchedEvaluator` (evaluator.py) builds an `og.Environment` for the task (R1Pro from
  `eval/r1pro.yaml`), loads a task instance (TRO json: object poses + robot pose), settles physics,
  then each step: `policy.forward(obs)` -> `env.step(action)`.
- Policy = `WebsocketPolicy` -> `WebsocketClientPolicy` (eval/utils/network_utils.py). msgpack over a
  websocket; one request per sim step; server replies `{"action": (1, 23)}`. `{"reset": True}` at episode start.
- Server side = `WebsocketPolicyServer` (same file) wrapping any object with `.act(obs)` / `.reset()`.
  Comet's `scripts/serve_b1k.py` uses it with `B1KPolicyWrapper` (receding horizon: infer a 32-step
  chunk, pop one action per request).
- Success = `env.task.success` (all BDDL goal predicates true). Episode ends on success or timeout.
- Q-score (metrics/task_metric.py): 1.0 if success, else max over goal options of
  (#goal predicates newly satisfied since start / #predicates). Per-rollout JSON also has time and
  base/arm distance metrics.

## Version mismatches (Comet expects BEHAVIOR v3.7.x / 2025 challenge)
1. Module path: Comet imports `omnigibson.learning.*`; fork renamed it to `omnigibson.eval.*`.
2. Obs batching: fork sends every obs with a leading num_envs dim (1, ...). Comet's wrapper expects unbatched.
3. Proprio `base_qvel`: v3.7.2 = world-frame virtual-joint velocity; fork rotates it into the robot frame.
   Verified on 2025 radio demos: |base_qvel - world d(pos)/dt| = 0.013 vs 0.107 for robot frame.
   Same 61-d layout / indices otherwise.
4. Images: Comet authors recommend rendering full-res (head 720x720, wrists 480x480) then
   resize_with_pad to 224 on the server. Fork default renders 224x224 directly.
5. Timeout: 2025 = 2x mean demo length = 4299 steps; fork default = 1.5x = 3224.
6. Instances: 2025 validation = first 10 of `test_instances.csv` for radio
   (242, 295, 211, 203, 109, 181, 197, 187, 214, 139). The 300 train-range radio TRO files are
   byte-identical in the 2025 and 2026 instance zips, so these can be loaded with fork `--mode train`.
   None of them are among Comet's 200 training-demo instances.
7. Sim/assets: Isaac Sim 4.5 + assets 3.7 -> Isaac Sim 5.1 + assets 3.9. Can't fix; expect some drift.
8. Norm stats: the radio checkpoint DOES ship `assets/behavior-1k/2025-challenge-demos/norm_stats.json`
   (my first remote zip listing missed it). openpi loads it automatically. See "Norm stats" below.
9. Proprio size: 2025 datasets/eval used a 256-d proprio vector; the fork's eval uses 61-d. Comet only
   reads base_qvel / trunk / arm qpos / gripper qpos, so the server shim passes the fork's 61-d indices.
- Same in both: controllers, reset joint pose, head aperture 40, 23-d action layout, robot start poses.

## Reference numbers
- Comet authors (GitHub issue #18/#19): radio success varies 40-80% across reruns; ~70% after RFT.

## Setup (reproduce)
```bash
# a. 2026 task instances (fork eval imports their metadata; ~110 MB)
OMNI_KIT_ACCEPT_EULA=YES conda run -n behavior python -c \
  "from omnigibson.utils.asset_utils import download_2026_challenge_task_instances as d; d()"
# b. Comet env (outside repo), uv 0.13.7
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/mli0603/openpi-comet /home/behavior/ext/openpi-comet   # @4bb2aa7
cd /home/behavior/ext/openpi-comet && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# (NOT needed: installing BEHAVIOR/OmniGibson into Comet's venv; steering/comet_server.py shims it.)
# c. radio checkpoint only (12.4 GB zip -> 12 GB dir)
mkdir -p /home/behavior/ext/ckpt && cd /home/behavior/ext/ckpt
curl -L -O https://huggingface.co/sunshk/comet_submission/resolve/main/pi05-turning_on_radio-sft.zip
unzip -q pi05-turning_on_radio-sft.zip && rm pi05-turning_on_radio-sft.zip
```
Disk used: ckpt 12 GB, Comet venv 7.4 GB, radio demo parquet+meta ~1 GB (norm-stats check only).

## Running
```bash
# terminal 1: policy server (~11 GB GPU with mem fraction 0.45; takes ~2-3 min to load)
cd /home/behavior/ext/openpi-comet
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/comet_server.py \
    --ckpt /home/behavior/ext/ckpt/pi05-turning_on_radio-sft --port 8000
# terminal 2: rollouts (repo root, behavior env)
OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/run_rollouts.py \
    --instances 242 --seeds 0 --max-steps 4299 --resolution full --output-dir <dir>
```
`run_rollouts.py` appends one row per rollout to `<dir>/rollouts.csv` (instance, seed, success, q_score,
success_step, steps, wall time, steps/sec, policy time, video path).

## Norm stats
- All published shared stats are identical (md5 d071463b...): openpi_comet pt12, pt50, both pytorch
  repos, and radio_best_pytorch.
- The radio SFT checkpoint ships different, radio-specific stats (md5 f6863bb7...). These are what load.
- `steering/norm_stats_check.py` recomputes stats from the 200 radio demos (Comet's script method:
  per-episode stats aggregated like v3.7.2 lerobot_utils, plus Comet's state extraction) and exact per-frame.
  Shift in normalized model-input units (quantile norm, range [-1,1]), averaged over radio demo frames:
  | compared with recompute (script method) | state mean / max | action mean / max |
  |---|---|---|
  | checkpoint's bundled stats | 0.028 / 0.21 (trunk2) | 0.043 / 0.34* |
  | pt12 shared stats          | 0.137 / 0.36 | 0.134 / 0.36 |
  *torso3 has ~zero range, so that shift is meaningless; next largest is torso2 0.21.
  => Checkpoint stats are radio-specific and close to recompute (small diffs likely from openpi's
  per-frame histogram quantiles vs per-episode aggregation). pt12 stats would have been meaningfully wrong.

## Things that broke / gotchas
- `import omnigibson` blocks on an Isaac Sim EULA prompt unless `OMNI_KIT_ACCEPT_EULA=YES`.
- Remote zip listing from the tail only shows part of a large zip's central directory; unzip to be sure.
- 2026-11-02: first rollout aborted during Isaac Sim launch (`terminate ... std::system_error: Invalid argument`
  right after "app ready", in SimulationApp._wait_for_viewport), preceded by ~1100 carb
  "Failed to create change watch ... errno=28/No space left on device". errno 28 here = inotify watch limit,
  not disk. fs.inotify.max_user_watches = 65536; VS Code held ~46.7k watches (it watches the 41k dirs under
  datasets/behavior-1k-assets) and claude-desktop ~18k. Fix: raise the limit
  (`sudo sysctl fs.inotify.max_user_watches=524288`) and/or exclude datasets/ from the VS Code file watcher.
  RESOLVED 2026-11-04: user raised fs.inotify.max_user_watches to 524288; Isaac Sim launches fine.

## Single-rollout results (instance 242, seed 0, timeout 4299, compat fixes on)
| run | resolution | success | steps | wall s (rollout) | steps/s | waiting on policy s |
|---|---|---|---|---|---|---|
| 1 (server cold) | full 720/480 | yes | 1341 | 192.0 | 7.0 | 72.3 (incl. JAX compile) |
| 2 | 224 | yes | 1043 | 90.7 | 11.5 | 7.8 |
| 3 (server warm) | full 720/480 | yes | 1276 | 153.0 | 8.3 | 40.5 |
- Wall time includes instance load + settle. One-time Isaac Sim launch + scene load is ~2 min per process.
- Full-res costs ~3x more policy time than 224, mostly from sending ~4 MB of images over the websocket every
  step (the server only runs inference every 32 steps). Sim-only rate is similar (11.3 vs 12.6 steps/s).
- Same seed is NOT bit-reproducible (1341 vs 1276 steps, both full-res): the simulator is nondeterministic
  even with seeded Python/NumPy/Torch and policy RNG.

## N=10 batch, 2025 val instances, full-res, timeout 4299 (2026-11-04): 3/10 success
Run dir: /home/behavior/ext/runs/val10_full. Order = 242,295,211,203,109,181,197,187,214,139, seeds 0-9.
| instance | seed | result | steps | what the video shows |
|---|---|---|---|---|
| 242 | 0 | success | 1315 | |
| 295 | 1 | timeout | 4300 | radio already near floor level at start; robot reaches down, grabs it, freezes |
| 211 | 2 | timeout | 4300 | radio on table; right gripper grabs it and LIFTS it, holds it in the air ~2600 steps |
| 203 | 3 | timeout | 4300 | table empty at start, radio on the floor by the window; robot grabs it on the floor, stalls |
| 109 | 4 | timeout | 4300 | radio on table; left arm pushes/rotates it (face toward robot), lets go, backs off, re-approaches |
| 181 | 5 | success | 1413 | |
| 197 | 6 | timeout | 4300 | radio on floor at start; robot goes to the (empty) table and circles it |
| 187 | 7 | success | 1278 | |
| 214 | 8 | timeout | 4300 | radio on floor at start (upright); right gripper grabs it, holds |
| 139 | 9 | timeout | 4300 | radio on table; right arm rotates it (face toward robot), arms retract, hovers |
Successes took 1278-1413 steps (radio pressed with the radio still on the table).

### Bug found: stale assisted grasp carries over between rollouts (fork evaluator, not Comet)
- `steering/check_instances.py` (zero actions, no policy): all 10 instances load with the radio exactly at its
  TRO pose on the table (z=0.534, table top 0.423). So the instance files are fine.
- Traced rerun of 197 -> 203: 197 (first in process) radio stays on table until the robot grabs it at ~1400 and
  the episode ends with graspR=1. 203 (next): radio z=0.485 at step 1, 0.111 (floor) by step 100, robot base
  unmoved, nearest finger 0.9 m away.
- Mechanism: BatchedEvaluator._load_instance_state calls robot.reset(), which does NOT release assisted-grasp
  constraints. The radio is then settled + snapshotted (update_initial_file) while still attached to the
  gripper, and env.reset restores that snapshot including the grasp. The episode starts with the radio
  tethered to the gripper; it drops when the gripper opens.
- In the batch, every "radio on floor at start" instance (295, 203, 197, 214) followed a rollout that ended with
  a hand on the radio. So 4/7 failures are likely this bug.
- Workaround (steering/run_rollouts.py, default ON): release_grasp_immediately on every arm before loading each
  instance; CSV column `stale_grasp_at_start`. `--keep-stale-grasps` reproduces the bug.
- `OnTop(radio, table)` reads 0 for several instances where the radio is clearly resting on the table
  (incl. successful 187); don't trust it as an init check.
- `check_instances.py` CSV write never runs: evaluator __exit__ calls og.shutdown(), which ends the process.
  Results are printed as INSTANCE_CHECK lines in the log.
- A/B (197 -> 203, seeds 6,3, 1500 steps, traced), /home/behavior/ext/runs/ab_grasp_{keep,release}:
  - keep (`--keep-stale-grasps`): stale grasp `right: radio_89` found before 203; 203 radio z 0.485 at step 1,
    0.111 (floor) by step 51. Bug reproduced.
  - release (default): INTERRUPTED 2026-11-04 to free the GPU for another project; rerun to confirm the fix.

## Status / resume (2026-11-04)
- Server and runs stopped (GPU freed). Restart the server (see Running) before any rollouts; it takes ~3 min.
- Next: (1) rerun ab_grasp_release; (2) rerun the N=10 batch with the stale-grasp fix + `--trace`.

## Sharing the GPU with another job (2026-11-04: MedGemma needs ~9 GB)
- Server: `XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.32` -> 8.0 GB, inference OK
  (`steering/probe_server.py` sends fake obs; first call 30 s JIT, then ~0.3 s per inference).
- Isaac Sim client peak: 224 = 5.6 GB; full-res = 9.3 GB. With MedGemma at 9.1 GB, only 224 fits.
- `steering/gpu_watchdog.sh <min_free_mib> <log>` kills run_rollouts.py if free GPU memory drops below the
  floor; logs per-process usage every 2 s.
- Don't `pkill -f <pattern>` from the agent shell: the pattern matches the shell's own command line and kills it.
- A/B release half (224): stale grasp `right: radio_89` found before 203 and released; radio stayed at
  z=0.534; 203 then SUCCEEDED at step 1163 (it failed in the original batch). Fix confirmed.
- val10_224_fixed: paused after rollout 1 (242 success, 1147 steps) to leave the GPU to MedGemma.
- `steering/wait_for_gpu.sh "<allowed pids>" <quiet_min> <log> -- <cmd>` starts a command once no other GPU
  process has existed for <quiet_min> minutes. Queued: val10_full_fixed (full-res, fix on, --trace).
- 2026-11-04 22:27: queued unattended via `steering/overnight.sh` (detached with setsid/nohup, under
  `systemd-inhibit --what=sleep:idle --mode=block`). It waits for 10 GPU-quiet minutes, starts the server,
  runs val10_full_fixed, then stops the server/watchdog and releases the inhibitor.
  Progress: /home/behavior/ext/logs/overnight.log (+ wait_for_gpu.log, val10_full_fixed.log, gpu_watchdog.log).
  Cancel: `systemd-inhibit --list | grep steering` -> kill that PID's process group (`kill -- -<pid>`).
  GNOME AC auto-suspend was already off (`sleep-inactive-ac-type = nothing`); no settings changed.

## N=10 with stale-grasp fix, full-res, timeout 4299 (2026-11-05, unattended): 4/10 success
Run dir: /home/behavior/ext/runs/val10_full_fixed (traces/ has per-step CSVs). Stale grasps released before
5/10 rollouts; every radio started on the table (z=0.534). No watchdog kills.
| instance | seed | batch 1 (bug) | fixed | fixed: what happened (trace + video) |
|---|---|---|---|---|
| 242 | 0 | S 1315 | S 1256 | lifted radio (z 0.68), pressed in hand |
| 295 | 1 | F (bug) | F | never grasped; right hand hovers beside the radio from ~860 to timeout, closest finger 11 cm |
| 211 | 2 | F | F | grabs radio, carries it off the table edge, holds it below table height (z~0.4) ~2600 steps |
| 203 | 3 | F (bug) | F | never grasped; radio barely moved, closest finger 15 cm (succeeded once at 224 in the A/B) |
| 109 | 4 | F | S 1602 | |
| 181 | 5 | S 1413 | F | near miss: right hand holds radio, left hand repeatedly near button (closest 6.9 cm) |
| 197 | 6 | F (bug) | F | grasped then put back; closest finger 17 cm |
| 187 | 7 | S 1278 | S 1490 | |
| 214 | 8 | F (bug) | S 2010 | |
| 139 | 9 | F | F | grasped briefly then released; closest finger 13 cm |
- All successes (both batches + A/B) lift the radio and press the button in hand; successes took 1147-2010 steps.
- Closest finger-link-to-button distance: 5.7-5.9 cm in successes vs 6.9-21 cm in failures (link origins, so the
  5 cm "near button" threshold in rollout_trace.py never fires; ~6 cm would be a better cut).
- Outcome flips between runs for 109, 181 (and 214, 203 across batches/resolutions): high per-instance variance.
- With n=10, 4/10 has a 95% CI of roughly 12-74%; Comet reports 40-80% across reruns.
- Timeout: max success step seen is 2010; `--max-steps 2500` would have kept every success and cut failed
  rollouts from ~8.5 to ~5 min.

## State-vector check (2026-11-05): fork 61-d proprio gives Comet the same 23-d state as 2025's 256-d
- Comet only uses 23 numbers: base_qvel(3) trunk_qpos(4) arm_left_qpos(7) arm_right_qpos(7) + 2 gripper widths.
  The other 233 of the 2025 vector (sin/cos, qvel, efforts, eef poses, global pose...) are never read.
- `steering/check_state_layout.py`: random pose set by joint name -> extracted from live 61-d proprio:
  max error 9e-10 over all 18 trunk/arm dims, each lands on the expected named joint (torso_joint1-4,
  left/right_arm_joint1-7, in order). Demo episode start state == fork reset pose exactly.
- 2025 demo values per dim vs current joint limits at that position: 0.0% outside for all 18 dims; the
  left/right-asymmetric joint2 ranges ([-0.17,3.14] vs [-3.14,0.17]) match the demos, so no L/R swap.
- Not covered: robot asset dynamics (robot assets 3.7 -> 3.8.2), renderer/physics version.

## Rollout report
`python steering/make_report.py --run <run dir> --batch1 <pre-fix run dir> --out /home/behavior/ext/report`
-> /home/behavior/ext/report/index.html (stats table; success + failure cases with video, keyframes, trace plot).

## Why only ~40%? Environment differences vs 2025 (2026-11-05)
Comet issue #17 reports 10/10 with this checkpoint on these 10 instances on BEHAVIOR 3.7, so the gap is ours.
`steering/check_robot_vs_2025.py` (live robot vs. 2025 demo data / v3.7.2 source):
- Cameras: identical. Pose rel. to base at reset within 0.6 mm and 0.00 deg of the 2025 demos for all three;
  intrinsics identical (head fx 306 @ 720, hfov 99.3; wrists fx 388.7 @ 480, hfov 63.4).
- **Base mass: robot asset default 47.6 kg; fork evaluator overrides it to 250 kg** (EVAL_BASE_LINK_MASS in
  eval/evaluator.py). v3.7.2's eval did not override it. Comet's base actions are velocities learned on 47.6 kg.
- Finger links have mesh collision (8 meshes each), so the fork's warp toggle check can see them.
- Press (toggle) rule: 2025 = PhysX sphere overlap of marker vs finger links + finger contact, 5 steps;
  fork = warp point-to-finger-mesh distance < marker radius + contact, 0.15 s. Shadow-tracked both
  (rollout_trace.py `overlap_2025`, `fork_toggle_timer_s`): in successes both fire together (5 overlap steps,
  timer 0.167 s); in every failure both read 0 all episode. => The press rule is NOT the cause.
  (`finger_contact_radio` column reads 0 even in successes: wrong buffer, ignore it.)
- Other diffs: fork loads only task rooms (2025 eval default loaded the whole house); Isaac Sim 4.5 -> 5.1;
  robot assets 3.7 -> 3.8.2 (cameras/joints verified identical, dynamics not verified).

### Campaign 1 (steering/campaign.sh): full-res, fix on, timeout 3000, traced
- massA_default_s0 (47.6 kg, seeds 0-9): **6/10** (vs 4/10 at 250 kg, val10_full_fixed). 295 and 203 now succeed.
  One success took 2711 steps (longest so far).
- massB_eval_s10 (250 kg, seeds 10-19), massC_default_s10 (47.6 kg, seeds 10-19): running.
- Queued after: campaign2.sh = massD (47.6 kg + whole house) and massE (47.6 kg, seeds 20-29, timeout 4299).
