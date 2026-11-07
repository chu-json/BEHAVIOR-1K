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
- Campaign 1 results: massB (250 kg, s10-19) **7/10**; massC (47.6 kg, s10-19) **4/10**.
  Pooled: 250 kg 11/20, 47.6 kg 10/20 => **base mass does not matter**; the 4/10 vs 6/10 was noise.
  Base motion is also nearly identical (peak speed 0.234 vs 0.238 m/s, distance 1.07 vs 1.17 m).
- Overall with the stale-grasp fix: **21/40 = 52.5%** (Wilson 95% CI ~37-68%), within Comet's 40-80%.
- Instance difficulty dominates: 242 4/4; 109, 181, 187, 203 3/4; 139, 214 2/4; 295 1/4; **197, 211 0/4**.
  Radio to robot's left at start: 16/24; to its right: 4/16. The 200 training demos are balanced (96 left /
  104 right), so it's not missing coverage. The robot does turn the correct way; the always-fail instances
  never align squarely with the radio (best heading error 197: 13-31 deg, 211: 28-34 deg; successes ~0-11).
- Successes now range 907-2711 steps; massE uses the full 4299 timeout to check for late successes.
- massD (47.6 kg + whole house, s0-9): **8/10** (fails only 211, 197) vs massA (same seeds, task rooms only) 6/10.
  Not significant alone; queued massF (whole house, s10-19) to pair with massC (4/10). campaign3.sh.
- massE (47.6 kg, rooms, s20-29, timeout 4299): 3/10; no failure recovered after step 3000.
- massF (47.6 kg, whole house, s10-19): 6/10 (211 succeeded for the first time; 7 tries before: 0).

### Final tally (70 rollouts with the fix), report: /home/behavior/ext/report_campaign/index.html
| condition | result |
|---|---|
| 250 kg, task rooms (s0-9, s10-19) | 11/20 |
| 47.6 kg, task rooms (s0-9, s10-19, s20-29) | 13/30 |
| 47.6 kg, whole house (s0-9, s10-19) | 14/20 |
| all | 38/70 = 54% |
- Base mass: no effect (Fisher p=0.57). Timeout: no effect beyond 3000.
- Whole house vs task rooms: 70% vs 48%, Fisher p=0.12; paired (same instance+seed, 47.6 kg): both succeed 9,
  house-only 5, rooms-only 1, neither 5 (McNemar p=0.22). Suggestive, not significant. Next: more whole-house
  batches; if it holds, use `--full-scene` as the default for baselines.
- Failure modes (auto-classified, all conditions): held radio until timeout 14, grasped then let go 13,
  never grasped 5. The 2025 press rule never fired on a failed rollout.

## Causal test: rotate the base to face the radio (steering/align_assist.py), 2026-11-05
Trigger: base moved, then stopped (<3 cm / 3 deg over 30 steps) within 1.5 m of the radio, >15 deg off-heading,
not grasping. Action: pause policy, rotate in place to face the radio's true position (<3 deg), hold 10 steps,
reset the policy's action queue, resume. Once per episode. 47.6 kg, task rooms, full-res, fix on, timeout 3000,
same (instance, seed) pairs as massA/C/E. Runs: /home/behavior/ext/runs/assist_{right,left_control};
report: /home/behavior/ext/report_assist/index.html.
| group | baseline (same pairs) | assisted | fired |
|---|---|---|---|
| radio on robot's right (211, 197, 214, 139 x 3 seeds) | 1/12 | **5/12** | 11/12 |
| radio on robot's left control (242, 295, 203, 109, 181, 187) | 6/6 | 4/6 | 3/6 (3/3 succeeded when fired) |
- Right side: 4 pairs flipped F->S, 0 S->F (exact sign test p=0.125). Per instance: 214 3/3 (baseline 0/3),
  139 2/3 (1/3), 211 0/3 (0/3), 197 0/3 (0/3).
- After the assist the base mostly stays aligned (211 at -5..-11 deg, 197 s16 at -3 deg; 197 s6 turned itself back
  to -27 deg), yet in 211/197 the closest finger never gets within 13-21 cm of the button (successes 5.0-5.7 cm).
- Left control: both failures were rollouts where the assist never fired (ordinary variance), so no sign of harm.
=> Alignment is a real bottleneck for some right-side instances (214), but not sufficient for the hardest two
   (211, 197): their failure is downstream, getting a finger onto the button.

### Button geometry follow-up (inconclusive; corrections)
- Toggle marker (radio model wxnicr, 3.9 assets): on the radio's front (+x) face, 4.2 cm off-center sideways,
  1.2 cm below mid-height; marker center 0.17 cm behind the face, detection sphere (r=2.24 cm) protrudes 2.1 cm
  (`steering/button_reachability.py`, PhysX raycasts). It IS reachable from the front face.
  CORRECTION: an earlier note that the marker was inside the radio body ignored the model's base_link_offset.
- Front-face direction vs robot approach: 197 160 deg (0/7), 211 104 deg (1/7), but 187 138 deg succeeds 5/7 and
  295 5 deg only 2/7. Not a clean explanation.
- 2025 demos: the radio is always on the table (z=0.534) when it turns on; operators rotate it ~31 deg (median).
  In our sim every Comet success has the radio lifted (z 0.68-0.81). Open question why (physics differences in
  pushing/friction/assisted grasp?). No evidence that the button moved between asset versions (not checked
  against the 3.7 asset).

## Prompt steerability sweep (overnight 2026-11-05/06) — steering/prompt_sweep.py
Config: whole house, 47.6 kg base, full-res, stale-grasp fix, timeout 3000. Setups 242 203 109 181 187, seed 0.
Checkpoints: A = radio SFT (its own norm stats), B = pt50 (`sunshk/openpi_comet/pi05-b1kpt50-cs32`, shared norm
stats d071463b, config pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k). Prompt via `comet_server.py --prompt`.
Prompts (suffix inserted before the final period): P0 original; P1 "Switch on the radio receiver sitting on the
living room table."; P2/P3 "... in the living room using your left/right hand."; P4 "... without picking the
radio up." Order: P0, P2, P3, P4, P1, each A then B; fresh server + sim per group.
Time estimate (from whole-house runs on these setups): success ~2.8 min, failure ~6 min, startup ~3.5 min/group
=> 50 rollouts worst case ~5.6 h, likely ~4-4.5 h. Bonus pass with seed 1 (same order) only starts a group if it
can finish within the 8 h deadline; reported separately.
Outputs: /home/behavior/ext/runs/prompt_sweep/{results.csv,index.html} rebuilt after every group
(`steering/sweep_report.py`). Logs: /home/behavior/ext/logs/prompt_sweep.log, sweep_*.log.
- pt50 download: unauthenticated HF limit is 1000 API requests / 5 min; snapshot_download silently stopped at
  3852/5948 files. Use a paced per-file download (see /home/behavior/ext/paced_dl.py) and verify the file count.
### Results (100 rollouts: main seed-0 pass + full bonus seed-1 pass), 2026-11-06 07:40
Report: /home/behavior/ext/runs/prompt_sweep/index.html; per-rollout CSV: .../prompt_sweep/results.csv.
Pooled over both seeds (10 rollouts per cell):
| ckpt | P0 | P1 para | P2 left | P3 right | P4 no-lift |
|---|---|---|---|---|---|
| A radio SFT success | 9/10 | 8/10 | 9/10 | 9/10 | 10/10 |
| B pt50 success | 4/10 | 5/10 | 4/10 | 2/10 | 3/10 |
- Hand: the pressing hand is fixed by the setup, not the prompt. Across all 63 successes (both checkpoints, all
  prompts, both seeds): 242 left 15/15, 203 right 14/14, 109 right 11/11, 187 left 10/10, 181 left 12/13 (one
  right). P2 "left hand" / P3 "right hand" never moved a setup to the requested hand.
- Lifting: every rollout that grasped the radio lifted it (>3 cm) in every condition, incl. P4 "without picking
  the radio up" (A 10/10, B 8/8 grasped). Raw "lifted %" dips (e.g. B 80%) come only from rollouts that never
  grasped.
- Paraphrase P1: no meaningful change for either checkpoint.
- pt50 is much weaker than the radio SFT on this task (18/50 vs 45/50 pooled over prompts); its lowest cells
  (P3 2/10, P4 3/10) are within noise of P0 (4/10) at n=10.
=> Comet's behavior on this task is not steerable through the language prompt (hand choice, lifting); the
   prompt mostly acts as a task identifier.

## Base pi0.5 (no BEHAVIOR fine-tuning) + steerability day campaign (started 2026-11-06 14:39)
Setup: `gs://openpi-assets/checkpoints/pi05_base` (12 GB, cached at ~/.cache/openpi) -> /home/behavior/ext/ckpt/
pi05_base_b1k = params symlink + Comet's shared B1K norm stats (d071463b; the base ckpt only ships stats for
its own embodiments: droid, franka, arx(_mobile), trossen(_mobile), ur5e(_dual), fibocom_mobile). Served with
config `pi05_b1k-base` (same B1K input/output transforms + 23-d action layout as Comet). Registry:
steering/probe_prompts.py (models A_radio_sft, B_pt50, C_pi05_base + prompt set).
- First look (600 steps, setup 242): drives ~0.9 m, turns ~3.4 rad, bends the torso until the head camera sees
  the floor/its own base, arms flail (EEF paths ~10-16 m), grippers half close. Not frozen: there is behavior.
New tools:
- `comet_server.py --dump-obs DIR`: saves the preprocessed obs (3x224x224 images + 61-d proprio) at every
  inference step.
- `prompt_probe.py` / `probe_report.py`: counterfactual prompt probe. Same obs + same diffusion noise, different
  prompt; effect ratio = prompt-induced action change / noise-induced change, per joint group; directional
  sign tests (arm bias, turn rate, forward velocity, total motion) paired over observations.
- Trace now logs EEF positions (base frame) and gripper openings; `motor_report.py` scores closed-loop motor
  prompts (hand raise, yaw change, forward displacement, total motion) vs P0 per setup.
Campaign (`steering/day_campaign.py`, detached, sleep inhibitor, resumable):
 1. base P0, setups 242/203/109/181/187 x seeds 0,1, 3000 steps, obs dumped   (basic rollouts)
 2. Comet A P0, same setups, seed 2, obs dumped                                (states for the probe)
 3. probe: {base, A, pt50} x {comet states, base states} x 150 obs x 13 prompts x 2 noise seeds
 4. base P1-P4 (exact overnight-sweep prompts), same setups/seeds
 5. motor prompts (P0, raise left/right arm, turn left/right, move forward, stay still, nonsense) on base and
    Comet A, setups 242/203/187, seed 0, 1000 steps
Outputs: /home/behavior/ext/runs/day/{base_sweep,obs_record,probe,motor}; log /home/behavior/ext/logs/day_campaign.log.
### Day campaign results so far (2026-11-06 17:25)
- Base pi0.5, P0, 10 rollouts: 0/10; never grasps; closest finger 13-117 cm; radio bumped up 14-17 cm in 3/10
  without a grasp (flailing arms). Gallery: /home/behavior/ext/runs/day/base_watch/all.html.
- Comet A, P0, seed 2 (obs recording): 4/5.
- Counterfactual prompt probe (150 obs per model x state set; report /home/behavior/ext/runs/day/probe/index.html).
  Opposite-instruction pair contrasts (cancel generic "extra words" effects):
  | contrast | base (comet st / base st) | A (comet / base) | pt50 (comet / base) |
  |---|---|---|---|
  | left-hand vs right-hand arm bias | null / null | null / null | null / null |
  | raise-left vs raise-right arm bias | -0.239 p=3e-9 / -0.164 p=3e-12 (REVERSED) | null / null | -0.029 p=0.02 / null |
  | turn-left vs turn-right wz | +0.001 p=1e-9 / null | +0.003 p=0.03 / +0.003 p=0.04 | +0.006 p=0.004 / null |
  "Stay still" vs controls: base strongly less motion (p=6e-21 / 8e-16); A weak (base states only); pt50 no.
  Effect ratios (prompt change / noise change): base 0.1-0.7, A 0.5-1.7, pt50 0.4-1.3.
  Single-prompt "followed" verdicts can be generic text effects (e.g. base P2 and P3 shift arm bias identically).
  Hypothesis for the reversed raise-left/right in base pi0.5: left/right action-slot convention of its pretraining
  embodiments differs from the B1K 23-d layout (left arm 7:14, right 15:22). Closed-loop motor test checks this.
- Base pi0.5 exact sweep (P0-P4 x 5 setups x seeds 0,1 = 50 rollouts, 3000 steps): **0/50**, never grasped.
  Left hand closest to the button in 7-9/10 under every prompt; radio bumped up (no grasp) 3/10 P0, 3 P1, 4 P2,
  4 P3, 5 P4 (most under "without picking the radio up"). Paired arm activity: P2 and P3 BOTH shift
  (left path - right path) toward the left arm (median +5.9 / +9.1 m) => generic text effect, not instruction
  following. EEF paths ~70 m per 3000 steps (flailing). Analysis script: scratch base_sweep_analysis.py (uses
  motor_report.metrics on base_sweep traces).
