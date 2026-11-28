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
- Closed-loop motor commands (120 rollouts, 1000 steps; base + Comet A: 3 setups x seeds 0,1; pt50: seed 0).
  `steering/motor_contrasts.py` (paired per setup+seed, expected > 0):
  | test | base pi0.5 | Comet A | pt50 |
  |---|---|---|---|
  | stay still vs nonsense (less motion) | 5/6, median 10 m | 4/6, ~0.3 m | 0/3 |
  | turn left vs turn right (yaw) | 5/6, median +41 deg (noisy) | 3/6 | 3/3, +271/+4/+354 deg |
  | raise left vs raise right (hand height) | 1/6 (reversed, as in probe) | 4/6, ~0.02 m | 2/3, ~0.03 m |
  | forward vs stay still (forward disp.) | 2/6 | 2/6 | 2/3 |
  Successes within 1000 steps: base 0/48, A 2/48, pt50 0/24.
- Summary page: /home/behavior/ext/runs/day/index.html (links gallery, base sweep, probe, motor reports).
=> Comet A is not language-steerable (prompt = task identifier). Base pi0.5 follows "stay still" (probe + closed
   loop), weakly "turn left/right", and puts left/right-arm words on the wrong arm (consistent with an ALOHA-style
   action-slot convention). pt50: possible turn signal, needs more seeds.
- Watch page (2026-11-12): https://claude.ai/artifact/QovHugAvtWgyjHQ1bwp3d6 (private). Built by
  `steering/build_watch_site.py --out /home/behavior/ext/site`: one example rollout per prompt per model (setup 242,
  seed 0; sweep prompts P0-P4 at 3000 steps, motor prompts at 1000 steps), videos re-encoded at x264 CRF 32
  (39 videos, 63 MB; published in two batches to stay under the 64 MB per-publish limit), plus side-by-side tables.
- Comet checkpoints: the submission repo has 26 (12 task-specific SFT/RFT + 14 snapshots of pt7/pt10/pt12/pt50);
  each of the 50 tasks used the best one on its validation set. Radio = pi05-turning_on_radio-sft.

## Why base pi0.5 fails, and can we steer it without training? (2026-11-17) — steering/fix_campaign.py
Diagnosis (base pi0.5 on the 23-d B1K action space, B1K norm stats, absolute joint targets):
- Not trained on R1Pro/BEHAVIOR at all; openpi README says base checkpoints are for fine-tuning.
- Action-convention mismatch: openpi pre-training uses joint DELTAS (ALOHA/DROID "use_delta_joint_actions"),
  Comet's B1K config uses absolute targets. With absolute wiring the first arm target is 0.11/0.59 rad (L/R) away
  from the current pose (Comet 0.003/0.004), step-to-step 0.014 rad (Comet <= 0.002); torso drifts 0.006 rad/step
  and folds until the head camera sees the floor; gripper commands normalize to ~0 (half-closed).
Fixes, all in my code: `comet_server.py --delta-joints` (+ delta norm stats from the 200 radio demos,
`steering/delta_norm_stats.py` -> /home/behavior/ext/ckpt/pi05_base_b1k_delta), `--freeze-torso`, `--freeze-base`,
`--best-of-n N` (V-GPS/RoboMonkey-style selection: sample N chunks, keep the one whose URDF-FK hand path gets
closest to the button; `steering/urdf_fk.py`, validated to 0.00 mm vs proprio eef; button position injected by
`run_rollouts.py --inject-target`), and `run_rollouts.py --approach-assist 0.8` (scripted drive to 0.8 m facing the
radio, then hand over). Setups 242 203 109 181 187, seed 0, 1500 steps, whole house, default mass, prompt P0.
`python steering/fix_report.py --root /home/behavior/ext/runs/fix` (closest = finger-to-button during policy steps):
| condition | succ | grasp | closest cm med/best | torso drift rad | jitter mm/step |
|---|---|---|---|---|---|
| C0 abs (as before) | 0/5 | 0/5 | 30.4 / 13.4 | 1.86 | 14.6 |
| C1 delta | 0/5 | 0/5 | 42.4 / 31.6 | 3.94 | 2.4 |
| C2 delta+torso | 0/5 | 0/5 | 50.1 / 38.4 | 0.31 | 1.5 |
| C3 delta+torso+approach | 0/5 | 0/5 | 27.0 / 14.2 | 0.60 | 1.6 |
| C4 abs+torso+approach | 0/5 | 0/5 | 20.0 / 15.3 | 0.45 | 11.7 |
| C5 C3 + best-of-8 | 0/5 | 0/5 | 36.0 / 13.8 | 0.07 | 1.5 |
| C6 C4 + best-of-8 | 0/5 | 0/5 | 16.0 / 8.1 | 0.14 | 13.4 |
| C7 delta+torso+base frozen+approach | 0/5 | 0/5 | 69.5 / 60.5 | 0.00 | 1.6 |
| C8 abs+torso+base frozen+approach | 0/5 | 0/5 | 27.6 / 25.3 | 0.04 | 11.8 |
- Delta wiring fixes the jitter (14.6 -> ~1.5 mm/step, Comet-like) but the arms then barely move toward anything.
- Approach works (ends 0.85 m, <3 deg). Comet presses the button from 0.67-0.90 m base-to-radio (radio pos
  recovered from the approach end pose; Comet A prompt-sweep P0 traces), so the standoff is right. After handover
  base pi0.5 drives 1-3.6 m away (C3/C4); freezing the base (C7/C8) keeps hands >= 25 cm away: no reaching.
- Best-of-8: delta candidates are nearly identical (critic spread median 6 cm, best predicted 1.29 m median);
  absolute candidates differ more (spread 35 cm, best predicted 0.93 m, min 3 cm) -> C6 is the closest any base
  run got (8 cm), but never contact; radio pushed 0.5-1.8 m by the body.
=> Inference-time steering can't make base pi0.5 do the task: selection/guidance need a policy that sometimes
   proposes good actions. Next: LoRA fine-tune of pi05_base on the 200 radio demos (openpi-comet already has a full
   fine-tune config `pi05_b1k-base`, 30k steps x batch 256 — too big for one 3090; LoRA needs > 22.5 GB).

## RLC setup (2026-11-19)
RLC = 1st place 2025 (IliaLarchenko/behavior-1k-solution @ca556f7, openpi fork wensi-ai/openpi `behavior` @01177e0;
paper arXiv 2512.06951). pi0.5 variant with 50 learned task embeddings (no text), stage predictor ("system 2"),
correlated flow noise, delta joint actions with per-timestep norm. All code outside the repo; no edits to RLC/openpi.
```bash
cd /home/behavior/ext/rlc && GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/IliaLarchenko/behavior-1k-solution solution
cd solution && GIT_LFS_SKIP_SMUDGE=1 git submodule update --init openpi      # BEHAVIOR-1K submodule NOT needed
GIT_LFS_SKIP_SMUDGE=1 UV_PYTHON=3.11 uv sync                                # py3.11, jax 0.5.3, torch 2.7.1
# RLC ships no uv.lock and the fresh resolve breaks openpi (no chex, pyarrow needs numpy 2): pin to openpi's lock
uv export --project openpi --frozen --no-hashes --no-emit-project --no-emit-workspace --no-dev | grep -v '^av==' > /tmp/req.txt
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -r /tmp/req.txt pytest   # pytest: imported by openpi
# checkpoint for radio only (task 0 -> checkpoint_2 per task_checkpoint_mapping.json): 12.65 GB (fp32 orbax)
uvx --from huggingface_hub python -c "from huggingface_hub import snapshot_download as s; \
  s('IliaLarchenko/behavior_submission', allow_patterns=['checkpoint_2/*'], local_dir='/home/behavior/ext/rlc/ckpt')"
```
Use `.venv/bin/python` directly, NOT `uv run` (uv run would re-sync and undo the pins). Venv 7.6 GB, ckpt 12 GB at
/home/behavior/ext/rlc/ckpt/checkpoint_2 (norm stats + action correlation matrix in
assets/IliaLarchenko/behavior_224_rgb/norm_stats.json).

Server (`steering/rlc_server.py`, replaces their scripts/serve_b1k.py; same protocol/client as comet_server.py):
```bash
cd /home/behavior/ext/rlc/solution
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.32 .venv/bin/python \
  /home/behavior/BEHAVIOR-1K/steering/rlc_server.py --ckpt /home/behavior/ext/rlc/ckpt/checkpoint_2 --port 8000
# client: unchanged, e.g. python steering/run_rollouts.py --resolution full ... (+ --inject-target for --best-of-n)
```
- Defaults = serve_b1k.py defaults (their challenge setup): chunk 30, execute 26 cubic-interpolated into 20 steps
  (base velocities x1.3; compression skipped when a gripper's range within the chunk > 0.2), last 4 kept for soft
  inpainting of the next chunk (inpainted during the first 70% of denoising, correlation-aware guidance after),
  20 flow steps, stage voting (history 3, 2 votes to advance), eval tricks on. Inference every 20 sim steps.
- Obs used: 3 RGB cameras resize_with_pad to 224 (trained on 224 versions of the 720/480 demo videos, so
  `--resolution full` matches training best; 224 is faster), proprio -> 23-d state: base_qvel(3, world frame as
  2025 -> CompatWrapper fix applies), trunk(4), left arm(7), left gripper width sum normalized 2*w/0.1-1, right arm(7),
  right gripper. Same fields as Comet, so the fork 61-d indices shim works (comet_server.install_omnigibson_shim).
- task_id: RLC uses the 2025 task index from obs["task_id"]; radio = 0 in both the 2025 list and the fork's
  B100_task_misc.csv. The server forces --task-id 0 anyway.
- Action out: absolute 23-d R1Pro (delta mask: torso joints 1-3 and both arms are deltas internally, converted back to
  absolute; base vel, torso joint 4, grippers absolute). Gripper commands can overshoot [-1, 1] (e.g. 1.19);
  the controller clips (command_input_limits default).
- Correction rules (b1k/shared/correction_rules.py, eval tricks): (1) radio-specific
  `task0_stage4_reset_to_stage2`: when the stage tracker reaches the last stage (4 of 0-4) it is forced back to
  stage 2; and from stage >= 2, if a gripper reads fully closed (normalized width < -0.98) while the other one is
  fully open or closed (not mid-way), the whole chunk is replaced by "hold current state, open that gripper" (1.0).
  The authors say it is legacy and probably unimportant. (2) general: gripper closed when the task table says it
  should be open -> same hold-and-open chunk; for radio, closing is only allowed from stage 2.
- Steering knobs: per-episode RNG seed (from `steering_seed`), `--best-of-n N` (N chunks scored by the URDF-FK
  hand-to-button critic, like comet_server), `--num-steps`, `--execute-in-n-steps` / `--actions-to-execute`
  (compression), `--no-eval-tricks`, stage tracker (`--history-len`, `--votes-to-promote`). The model's
  `noise=` argument is ignored on the inpainting path (it draws its own correlated noise from the RNG), so there is
  no comet-style --noise-scale.
- GPU memory: bf16 params are 6.3 GB (3.15B params). Not measured on GPU (busy). Expect about 7-8 GB, so the 0.32
  fraction (7.9 GB on 24 GB) is tight; if it OOMs at the first inference, use 0.36-0.40.
- CPU smoke test (`steering/rlc_smoke_test.py`, real obs from Comet --dump-obs, 45 steps = 3 inferences incl.
  inpainting): OK direct and over websocket (port 8010), bit-identical with the same seed; actions (1, 23),
  first action = current joints within 0.003 rad, later chunks within 0.08 rad; 19 s per inference on CPU;
  host RAM peak 9.0 GB. Seed sweep 0-5: chunk torso deviation <= 0.06 rad.
What broke:
- Host OOM on CPU (31 GB box, about 13 GB free): (a) openpi restore_params has orbax read all fp32 arrays before
  the bf16 cast (> 13 GB) -> rlc_server.py replaces it with a chunked tensorstore reader (peak ~7 GB, same
  params); (b) XLA:CPU has no bf16 dot, so it upcasts and loop-invariant code motion hoists fp32 copies of
  whole stacked weights out of the layer scan -> for CPU runs set
  `XLA_FLAGS=--xla_disable_hlo_passes=while-loop-invariant-code-motion` (temp 1239 -> 31 MB in a toy test).
  Run CPU jobs as `systemd-run --user -p MemoryMax=12G -p MemorySwapMax=0 -p OOMScoreAdjust=1000` so only they die.
- Best-of-N bug (fixed before use): openpi DeltaActions edits `initial_actions` in place, so N infer calls on the
  same array moved candidate k's inpainting targets by -k*state (torso jumps of 1-3 rad). BestOfN now copies.

### RLC freeze debugging (2026-11-20), CPU only
GPU run RLC_B0_baseline (242/s0, 3000 steps): both hands to x 0.31 m in steps 21-60, then frozen; yaw exactly 0.575
for 2000 steps, stage 0 throughout. Tools: `steering/rlc_demo_check.py` (2025 demo ep 40, local parquet + RGB
videos at /home/behavior/ext/rlc/demo{full,224}), `rlc_server.py --log-chunks DIR` (obs + executed chunk per
inference), `--no-inpaint` (A/B knob).
- Format ruled out: on demo observations sent in our client format (720/480 frames -> RLC resize_with_pad, 256-d
  state fields at the fork's 61-d indices), single inferences match the demo (t=20: wz 0.25 vs demo 0.29; t=60
  0.29 vs 0.30; t=900 arm motion 0.39 vs 0.35, predicted stage = true stage). Full-path replay (inpainting,
  compression, voting) of frames 0-300 turns at 0.37 = 1.3 x demo 0.3 rad/s and then drives forward like the demo.
  224-dataset images vs pad-resized full-res: same behaviour. Our sim's first frame matches Comet's on 242
  (view, colours), so RGB order/camera slots are right.
- Not base_qvel: zeroing it at demo t=20 still gives the turn.
- What drives it is the images: same state, sim frames from the RLC video -> wz <= 0.06 decaying to 0; demo frames
  -> 0.2-0.26.
- Closed-loop reconstruction on CPU (real rollout frames in order, state = last command): RLC itself commands
  "both arms forward": seed 1 ends with hands at L (0.317, 0.471) R (0.304, 0.465) at t=60, vs the real
  (0.312, 0.469)/(0.300, 0.464). But all 5 CPU seeds also command wz 0.3-0.4 at steps 20-40, while the real yaw
  never changed. Open: either the GPU run sampled no turn (jax.random is platform-independent, but bf16 GPU vs
  fp32-upcast CPU numerics differ), or the base didn't execute it. Next: 200-step GPU run with --log-chunks, then
  compare logged base commands with the trace's base_yaw.
- Inpainting is not the cause (static-frame A/B: same small, intermittent turn with or without it).

## Overnight steering campaign on Comet + RLC (2026-11-19/20) — steering/steer_campaign.py
Setup: Comet radio SFT, 20 pairs = 10 val setups x seeds 0-19, whole house, default mass, full-res, 3000 steps.
Baseline on the same pairs: massD + massF = 14/20. Runs as systemd user unit `steer-night` (survives editor restarts;
setsid did NOT survive a Claude Code restart). Queue = /home/behavior/ext/runs/steer/queue.json (re-read per
condition; entries ending `_once` are removed by `queue-once.service` when they start).
Summary: `python steering/steer_report.py` (paired flips vs baseline, exact sign test).
- Segmentation (seg_semantic) segfaults this Isaac build at sensor init (deterministic; depth_linear alone is fine).
  Workaround: the radio's 3D AABB projected into each camera = a detector-like 2D box (SteerVisionWrapper).
- S1 best-of-8, oracle FK critic: 10/20 (2 F->S, 6 S->F, p=0.29). Comet's 8 samples are ~identical (critic spread
  median 0.8 cm), so selection has nothing to choose; S1 is effectively baseline with a different RNG stream ->
  the 14 vs 10 gap is mostly run-to-run variance. Grasping unchanged (19/20 vs 20/20). => B0 fresh baseline moved up.
- R1 recovery (stall 500 steps / radio pushed -> let go, back off, home, re-approach; max 2): 13/20 (3 F->S, 4 S->F);
  fired in 9 rollouts, 2 of those then succeeded. Triggers came at stalls 17-19 cm from the button.
- V1 radio tinted red (box overlay) in all 3 cameras: 3/20 (0 F->S, 11 S->F, p=0.001). Overlaying the image breaks
  Comet: visual prompting by editing the input is out-of-distribution for a fine-tuned VLA.
- RLC (1st place 2025) baseline, setup 242 seed 0: robot freezes (base 3.6 cm in 3000 steps, stage 0 throughout).
  CPU debugging (agent): obs/action format verified against 2025 demos (RLC reproduces demo actions incl. base turns).
  GPU chunk log (rlc_server.py --log-chunks): the policy itself commands ~0 base motion (|wz| <= 0.001 except two
  brief 0.17 pulses, which the base executes: yaw +0.017) -> policy-level stall on this start view, not a harness bug.
  Next: RLC baseline on 10 pairs and RLC + scripted approach (shared autonomy) on the same 10.
- Replaced S2 (best-of-8 with depth target) by S3 = noise x2 + best-of-8 (create diversity, then select), since
  Comet's samples have no diversity to select from. S2 moved to the end.
- B0 fresh Comet baseline (same config + seeds as massD/F): 10/20 (vs 14/20 then). Same seeds don't reproduce;
  run-to-run swing is ~+-4/20, so S1 (10/20, = B0) is no effect; R1 13/20 is +3 vs B0 (5 F->S / 2 S->F, p=0.45);
  V1 3/20 vs B0: 1/8, p=0.04 (harmful). steer_report.py now shows both comparisons.
- RLC, 10 pairs (val10 x seeds 0-9): 5/10 (= Comet B0 on the same pairs, 5/10); slower (median success step 2178
  vs Comet 1375; two successes at 2821/2967, so the 2025 4299-step budget would likely raise it). The 242 s0
  freeze did not recur (it succeeded), so the freeze was a stochastic stall.
- RLC + scripted approach (--approach-assist 0.8): 6/10, median success step 1644; solved 197 (Comet ~never) and
  139, lost 242. n=10, not significant, but the only steering that has helped RLC.
- S3 noise x2 + best-of-8 oracle: stopped at 0/7 (B0 same pairs 4/7). Plans now differ (critic spread 8.9 cm vs
  0.8) but never grasp the radio: scaled initial noise is off-distribution for the flow and produces bad plans.
- RLC on all 20 pairs: 8/20 (median success step 2146); RLC + approach 9/20 (step 1748), flips 5 F->S / 4 S->F.
  Setup-dependent: 197 0/2 -> 2/2 (Comet and RLC never solve 197 alone), 139 0/2 -> 2/2, 242 1/2 -> 0/2.
  => the approach fixes start-heading problems and hurts where the policy's own approach was fine.
=> Summary of the night: help that changes WHERE the robot is (approach, stall-retry) is the only thing that
   helped, and only on setups that need it. Changing inputs (tint) or sampling (best-of-N, noise) did nothing or
   hurt. Next idea: a learned/heuristic "needs help?" detector that decides when to intervene (selective shared
   autonomy), evaluated per setup.
- Report: https://claude.ai/artifact/TCmuKtkKRpUiiba4epM2T1 (private). Queue continues: V2 dim, N1, S2.
- V2 dim outside the box (all cams): 10/20 = B0 (4/4 flips). Neutral; at the halfway point it was 3/10 (noise).
  => V1's harm comes from repainting the radio (incl. the wrist close-ups), not from image edits in general.
- Queued (user suggestion): V3 = thin red outline (6 px at 720 -> ~2 px at 224) on the HEAD camera only; V4 = tint on
  the head camera only (`--overlay outline|tint --overlay-cams head`).

## New direction (2026-11-20): fine-tune pi0.5, preserve steerability; WAM vs VLA
See steering/RESEARCH.md (plan, taxonomy, hypotheses) and steering/LITERATURE.md (survey, in progress).
Queue after V3 deferred to /home/behavior/ext/runs/steer/queue_deferred.json (V4, N1, S2) to free the GPU for
fine-tuning. Radio videos (rgb + seg_instance_id, 1200 files) downloading via unit `dl-radio`.

## CyberWAM setup (2026-11-20)
Goal: be able to evaluate CyberWAM (Xiaomi, BEHAVIOR Challenge 2026 WAM; code /home/behavior/ext/cyberwam @3876f7d,
github.com/xiaomi-research/cyberwam) in our harness as soon as the mentor sends `model_a/` (checkpoint.pt,
checkpoint_contract.json, norm_stats.json, model_config.json). Everything up to loading the weights is done and
tested without a GPU.

**Env**: `/home/behavior/ext/cyberwam/.venv` (uv, Python 3.10.x, torch 2.7.1+cu128 / torchvision 0.22.1, the
pyproject pins minus deepspeed (training only) + websockets 15.0.1 / msgpack 1.1.0 / pytest; list in
`/home/behavior/ext/cyberwam/steering_venv_requirements.txt`; fastwam installed `-e --no-deps`). Their test suite:
511 passed on CPU (`CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python -m pytest tests`). Not Docker.

**Base assets** (`/home/behavior/ext/cyberwam/checkpoints` = the Docker image's runtime-assets root, 33.1 GB; script
`steering/cyberwam_assets.py`, unit `dl-cyberwam`, all ungated; only what INFERENCE loads, per
fastwam/models/wan22/helpers/{io,loader}.py, the flex model config and model_factory.py):
| file (under checkpoints/models/) | size | used by |
|---|---|---|
| Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{1..3}-of-00003.safetensors (+index, config), HF @921dbaf | 20.0 GB (fp32) | video expert init; redundant (checkpoint.pt strict-loads `mot`), skip with `--skip-pretrained-dit` |
| DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors, ModelScope (sha256 ok) | 1.41 GB | VAE (redirect_common_files=true default) |
| dinov3/vit_base_patch16_dinov3.lvd1689m/model.safetensors, HF timm @c6a5fb7 | 0.34 GB | frozen DINOv3 target encoder (DINOv3 License, not Apache) |
| Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/* (tokenizer), HF @37ec512 | 0.02 GB | text embeddings only |
| Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth, HF | 11.4 GB | text embeddings only (never loaded by the server) |
Not downloaded: ActionDiT init (training only; `wam_server.py` sets `model.action_dit_pretrained_path=null`, the
action expert is then overwritten by checkpoint.pt). Verified: VAE and DiT key hashes match fastwam's registry,
DINO loads into timm, Hydra config composes with our overrides.

**Text embeddings** (`checkpoints/text_embeds/<sha256(prompt)>.t5_len128.wan22ti2v5b.pt`): the server looks up
`format_b1k_prompt(task_data id)` = "A video recorded from a robot's point of view executing the following
instruction: turning on radio" (task NAME with spaces; the 2026 demos' meta/episodes `tasks` field is the name, not
the long instruction in tasks.jsonl, so training used the same prompt). Their scripts/precompute_text_embeds.py
would build T5 in fp32 (~23 GB RAM) and needs a local dataset tasks.jsonl, so `steering/wam_text_embed.py` does the
same computation (their tokenizer + WanTextEncoder, bf16, CPU) with a meta-device model + mmap'ed weights (~50 s
per prompt, 4 threads). Cached so far: turning_on_radio (official), "Turn on the radio receiver that's on the table
in the living room.", "turn on the radio", "press the power button on the radio", "turning off radio" (negative
control). Arbitrary prompt: `CUDA_VISIBLE_DEVICES= /home/behavior/ext/cyberwam/.venv/bin/python
steering/wam_text_embed.py --text "<instruction>"` (or --file, --raw for an exact prompt). Caveat for language
steering: the model only ever saw 100 fixed task-name prompts, and the per-task prompt carries little information
(mean-token cosine to "turning off radio" is 0.92).

**What CyberWAM needs from the client (verified against challenge_docker/{observation,session,server}.py)**:
- obs keys `*::proprio` (61-d fork layout accepted, collapsed to 23-d with the fork's indices), `task_id` [B,1]
  (fork TASK_NAMES_TO_INDICES == their task_data.json order for all 100 tasks), per camera `*zed*::rgb`,
  `*left_realsense*::rgb`, `*right_realsense*::rgb` and the same with `::depth_linear` (Flex requires depth -> point
  maps; inf/nan clamped server-side). Any extra key (steering_seed, cam_rel_poses) just needs a leading batch dim.
- resolution head 720x720, wrists 480x480 (server resizes to a 384x256 mosaic).
- base_qvel in the ROBOT frame = fork native. `steering/check_qvel_frame.py`: 2026 demos' base_qvel matches the
  robot-frame base action command (rel. err 0.07-0.18, k=0.74 = the 0.75 m/s output limit) and not the rotated one
  (0.17-0.79); 2025 demos are the opposite (world). So NO compat fix #3 for CyberWAM.
- `__action_chunk_size__` = 32 (or 0): the server returns `action_chunk` [B,32,23] and errors if asked for another
  chunk while its queue is not drained, i.e. the client must replay the whole chunk (fork WebsocketClientPolicy
  action_chunk_size=32 does exactly that). speed_2x: each 15 Hz prediction = one 30 Hz step (base velocities
  doubled server-side, seen in the stub test).
- Same robot config as fork eval/r1pro.yaml (controllers, limits, reset pose identical; only proprio_obs differs).
- Sampling seed: fixed 20260719 + chunk index per episode (deterministic given obs); `--episode-seed` makes it
  20260719 + 1000003*steering_seed + chunk index.

**What changed in steering/** (default behavior unchanged):
- `compat_wrapper.CyberWAMWrapper`: CompatFullResWrapper + depth_linear, world_frame_base_qvel False.
- `run_rollouts.py --policy-profile cyberwam`: wrapper `full_rgbd` (= CyberWAMWrapper), robot-frame base_qvel,
  WebsocketPolicy action_chunk_size=32. Rejects --overlay/--depth-target (SteerVisionWrapper strips depth).
  Assist/recovery wrappers + chunk replay are untested (the client resumes a leftover chunk after a pause).
- `wam_server.py`: Dockerfile env + runtime_launcher bundle validation in-process, then challenge_docker.server.main();
  `--preflight-only`, `--stub`, `--episode-seed`, `--prompt-text/--prompt-file` (hot-swappable prompt, "RAW:" =
  exact prompt), `--skip-pretrained-dit`, checkpoint.pt loaded with mmap by default. Tested: preflight on a fake
  bundle (+ their preflight_models.py), stub server <-> fork WebsocketClientPolicy (K=32 replay boundaries at
  32/64, reset, seed hook, full-res RGB-D shapes reach the policy), prompt-file hook (switch, RAW, missing cache).
- `wam_text_embed.py`, `cyberwam_assets.py`, `check_qvel_frame.py`.

**Run once model_a arrives** (put it at e.g. /home/behavior/ext/ckpt/cyberwam/model_a; verify their SHA256SUMS.txt;
never edit norm_stats.json / contract files - hashes are checked):
```bash
V=/home/behavior/ext/cyberwam/.venv/bin/python; M=/home/behavior/ext/ckpt/cyberwam/model_a
CUDA_VISIBLE_DEVICES= $V steering/wam_server.py --model-dir $M --preflight-only     # must print PREFLIGHT OK
# server (port 8000 is used by comet_server; pick another); detached:
systemd-run --user --unit=wam-server --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  /usr/bin/bash -c "$V /home/behavior/BEHAVIOR-1K/steering/wam_server.py --model-dir $M --port 8010 \
  --skip-pretrained-dit --episode-seed --telemetry /home/behavior/ext/logs/wam_telemetry.jsonl \
  > /home/behavior/ext/logs/wam_server.log 2>&1"
curl -s http://127.0.0.1:8010/healthz          # ready + metadata (contract/checkpoint sha256)
# client (behavior env, repo root): 2025 radio val10 x 2 seeds, 2025 budget
OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python steering/run_rollouts.py --policy-profile cyberwam \
  --port 8010 --instances 242 295 211 203 109 181 197 187 214 139 -n 20 --max-steps 4299 \
  --output-dir /home/behavior/ext/runs/wam/base
# wiring smoke test without weights (stub server, zero normalized actions; robot will not do the task):
$V steering/wam_server.py --stub --port 8011   # then run_rollouts ... --policy-profile cyberwam --port 8011 -n 1 --max-steps 100
```
Official-evaluator equivalent (their README): `python -m omnigibson.eval.eval --env-wrapper
omnigibson.eval.wrappers.RGBDFullResWrapper --replay-action-chunk-size 32 ...`; run_rollouts differs only by
steering_seed (ignored unless --episode-seed), released stale grasps, and our instance/seed bookkeeping.

**Resource estimate (not measured; GPU off-limits today)**: ~6B params (video expert 5B + action expert ~1B) +
VAE 0.7B + DINO 0.09B in bf16 = ~13.5 GB weights; 20 denoising steps on a 9-frame 384x256 mosaic x 3 streams ->
expect ~15-18 GB VRAM peak. Our sim at full-res uses ~11 GB, so server + sim on the one 3090 (24 GB) probably does
NOT fit; options: server on another GPU/machine (`--host`), or free sim memory. CPU RAM at startup: the DiT is
built in fp32 on CPU first (~20 GB, +~10 GB if loading the pretrained shards; hence --skip-pretrained-dit) and
this box has 31 GB shared with other jobs -> start it when the fine-tuning job / sims are not running. Speed: their
challenge machines did 20-step inference per 32-step chunk; on a 3090 probably several s per chunk (~40 chunks for a
1300-step radio episode at 2x).

**Open questions (mentor)**:
1. Exactly which bundle: the released `model_a` (all 100 tasks, step 115,884) or another checkpoint; does it come
   with SHA256SUMS.txt and model_config.json (allowed_task_ids must include 0, batch_size 1)?
2. Is checkpoint.pt bf16 weights-only (size?) - drives RAM/VRAM; any optimizer state inside?
3. Which GPU did they serve on and what VRAM/latency per chunk? Is a 24 GB card enough alongside Isaac Sim, or
   should the server run on a separate machine?
4. The README mentions per-task 1x overrides (tasks 55/77/95); radio uses the default speed_2x / horizon 32 - confirm
   that is what their reported radio number used.
5. Is there any prompt variation in training (they say fixed template per task)? If not, language steerability is
   expected to be near zero - worth testing anyway with the cached variants.
- V3 thin red outline, HEAD camera only (radio pixels and wrist views untouched): 6/20 (vs B0 0 F->S / 4 S->F;
  vs old baselines 2/10, p=0.04). Even a minimal visual cue hurts fine-tuned Comet => input-side visual prompting
  is not viable without training on it (motivates the outline-augmented fine-tune, RESEARCH.md M3.5).
  Note: the radio is itself red; a red cue blends with it.
- Perturbation smoke (steering/perturb.py): --clutter 6 --occluder on 242/s0 works (objects rest on the glass
  coffee table around the radio, panel hides part of it; frame checked). Radio-yaw smoke run after it.

## pi0.5 radio fine-tuning pipeline (2026-11-20) — steering/ft/
Goal: fine-tune base pi0.5 (openpi JAX, via openpi-comet's training loop, no edits to openpi-comet/BEHAVIOR) on the
200 2025 radio demos on the one 3090, servable by comet_server.py. Foundation for "fine-tune while preserving
steerability" (outline/box augmentation from the demo segmentation, prompt augmentation).
Files: `preprocess.py` (demos -> compact data), `radio_data.py` (dataset, RadioInputs/Outputs transforms, norm stats),
`augment.py` (prompt/image augmentation hooks), `ft_config.py` (TrainConfigs in code), `ft_loader.py` (our data loader
for openpi's train.py), `train_radio.py` (entry point), `eval_offline.py` (held-out demo action error),
`probe.sh` (speed/memory probe), `launch_ft.sh` (old auto probe+launch script; superseded by the manual run below),
`download_meta.py` (meta/episodes/*.json), `memwatch.sh`.
**Representation** (identical at train and serve time, all in openpi transforms):
- state 23-d in ACTION order (base vel 0:3, torso 3:7, L arm 7:14, L gripper width 14, R arm 15:22, R gripper 22).
  Comet's b1k_policy puts both gripper widths last (state 21,22), which misaligns state and action dims, so we use our
  own `RadioInputs` (accepts 23-d, the fork's 61-d proprio, or 2025 256-d). Base vel is rotated into the ROBOT frame
  in the training data (2025 256-d stores world-frame; rotating by base_qpos yaw (256-d index 246) matches the
  robot-frame action command to 0.004 vs 0.016 unrotated); the fork's eval proprio is robot-frame, so train == eval.
- actions: torso + both arms as deltas from the current state (openpi DeltaActions/AbsoluteActions, mask
  `DELTA_MASK=(-3, 11, -1, 7, -1)`), base vel + grippers absolute. Action horizon 32 (= Comet, for comparability).
- norm: quantile stats of state + DELTA actions over the 200 demos (exact quantiles, every 2nd frame's 32-step chunk;
  dims with q99-q01 < 1e-3 widened, only torso4) -> /home/behavior/ext/data/radio_ft/assets/radio_ft/norm_stats.json;
  openpi copies it into each checkpoint's assets/radio_ft/. Delta arm actions are heavy-tailed: ~1% of normalized
  values beyond |1.5|, max ~9 (fine; same as openpi's own delta configs).
- images: head -> base_0_rgb, left/right wrist; openpi_client resize_with_pad to 224 (PIL bilinear: the exact
  function Comet's B1KPolicyWrapper applies to full-res sim images at serve time). openpi's train-time image aug
  (crop/rotate/color) runs inside the model as usual.
- prompt: Comet's radio prompt via per-sample `prompt_fn(ctx)`; `--prompt-aug paraphrase` exists (50% paraphrases).
- 5 demos held out for offline validation (`ft_config.HOLDOUT` = episodes 2880 2960 2970 2990 3000): train on 195.
**Data** (`/home/behavior/ext/data/radio_ft`, 18 GB total; 430k frames): per episode `ep_<id>.npz` (state23, action,
raw 256-d proprio, task_info), per camera `ep_<id>_<cam>.jpg.npy` + `.off.npy` (JPEG q95 224x224 bytes + offsets,
mmap'd; 17.9 GB), `ep_<id>_seg.npz` (94 MB: radio + toggle-button masks packbits 224x224, boxes, pixel counts per
camera). Loader: 23 batches/s of 16 with 4 workers on CPU (we need 0.15/s; training uses 1 worker).
- Seg decoding: the 2025 seg_instance_id videos encode each id as a palette color (v3.7.2 obs_utils
  `generate_yuv_palette` over the camera's `unique_ins_ids` from meta/episodes/*.json, written as rgb24); nearest
  palette color via per-channel LUT = 100% agreement with the reference cdist decoder. Radio = instances whose object
  name contains "radio" (body + `togglebutton` link, e.g. ids 8/9 in ep 10). H.265 chroma bleed made isolated
  radio-colored pixels at other objects' edges -> 3x3 opening at full res (first seg pass discarded). Radio visible in
  92% of head frames / 26% left wrist / 57% right wrist; button 44 / 13 / 8%. `augment.outline_radio` draws the
  contour (checked visually), `augment.box_radio` the box; `--image-aug outline|outline_p50|box`.
- Preprocessing: `preprocess.py --parts lowdim,rgb --workers 5` 20 min; `--parts seg --workers 5` 45 min.
**Commands**:
```bash
cd /home/behavior/ext/openpi-comet
.venv/bin/python /home/behavior/BEHAVIOR-1K/steering/ft/train_radio.py --compute-norm-stats
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 .venv/bin/python -u /home/behavior/BEHAVIOR-1K/steering/ft/train_radio.py \
  --exp radio_lora_v1 --variant lora --batch-size 16 --steps 5000 --lr 5e-5 --warmup 250 --save-interval 500 \
  --keep-period 1000 --workers 1            # variants: lora | lora_fv (SigLIP frozen) | expert | full
# serve (output = absolute 23-d joint targets, Comet receding-horizon wrapper as before; do NOT add --delta-joints)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 .venv/bin/python /home/behavior/BEHAVIOR-1K/steering/comet_server.py \
  --ckpt /home/behavior/ext/ckpt/ft/radio_lora_v1/<step> --config steering_ft:radio_lora --port 8000
# offline action error on the 5 held-out demos
.venv/bin/python /home/behavior/BEHAVIOR-1K/steering/ft/eval_offline.py --ckpt <ckpt> --config radio_lora
```
**Speed / memory on the 3090** (bf16, remat, XLA mem fraction 0.95, EMA off):
| variant | batch | fits | s/step | samples/s | GPU peak |
|---|---|---|---|---|---|
| lora (SigLIP trained fully, openpi default) | 32 | yes (XLA remat warnings) | 13-15 | ~2.3 | at the 22.4 GiB limit |
| lora | 16 | yes | 6.7-6.9 | 2.4 | 17.8 GiB |
| expert (PaliGemma frozen, action expert full) | 32 | yes | 11 | 2.9 | - |
Compute-bound (GPU 100%, ~250 W): ~2.4 samples/s whatever the batch => ~85k samples in 10 h, i.e. ~0.2 epoch and
~1% of Comet's full fine-tune (30k x 256 = 7.7M samples). Freezing PaliGemma ("expert") only gives ~20%: XLA does
not prune the prefix backward pass. A big speed-up needs a stop-gradient on the prefix KV (model patch in steering/)
or fewer image tokens.
Host RAM: ~7-8 GB steady (1 worker), init loads the 12 GB fp32 base params, checkpoint save peaks the unit at ~22 GB
(page cache included). A first probe with 4 workers was killed by systemd-oomd during the save (swap is full from
desktop apps) -> 1 worker. Checkpoint = 9 GB (params 6 GB + train_state 3 GB).
**Verified**: transforms round trip (delta -> normalize -> unnormalize -> absolute) exact to 6e-11; serving a 6-step
checkpoint through comet_server.py `--config steering_ft:radio_lora` + `rlc_smoke_test.py --mode ws` on saved sim
observations: SMOKE TEST OK, absolute targets, torso within 0.08 rad and arms within 0.44 rad of current proprio
(model ~untrained); eval_offline plumbing OK (6-step model is worse than "hold current joints", as expected).
**Broke / gotchas**: openpi-comet's TorchDataLoader and config import BEHAVIOR v3.7 modules -> own loader +
comet_server's omnigibson shim, installed at import of ft_loader (also in spawned workers); stdin (`python -`)
scripts hang with spawn workers; warmup must be < steps (optax cosine with negative decay steps; clamped);
openpi's `--overwrite` wipes the exp dir (our args json goes to ckpt/ft/<exp>.args.json); `max_to_keep=1` +
keep_period keeps multiples of keep_period plus the latest.
**First run** (systemd unit `ft-radio`, started 2026-11-20 16:50, log /home/behavior/ext/logs/ft_radio.log, host-RAM
log ft_radio_mem.log, `/home/behavior/ext/GPU_TRAINING` exists while it runs): lora, batch 16, 5000 steps (~80k
samples, ~0.19 epoch), peak LR 5e-5 (2x openpi default because of the short budget), 250 warmup, cosine to 5e-6,
checkpoints every 500 -> /home/behavior/ext/ckpt/ft/radio_lora_v1/<step> (kept: multiples of 1000 + latest; ~9 GB
each). 7.0 s/step => expected finish ~03:00 2026-11-21. Step-500 save OK (unit peak 17.5 GB, >= 12.7 GB available).
Loss (flow-matching MSE, normalized units): step 0 0.286, 50 0.217, 100 0.130, 150 0.089, 200 0.066, 300 0.055,
400 0.049, 500 0.046.
Next: eval_offline + closed-loop rollouts (comet_server `--config steering_ft:radio_lora`) on checkpoints when the GPU
is free; if it is undertrained, options: lora_fv/expert with a prefix stop-gradient patch (~3x faster), longer runs.

## ft-radio v1 results (2026-11-27)
- `ft-radio` (LoRA, bs16, 5000 steps) finished 2026-11-21 02:33; final loss 0.029. Checkpoints 1000-4999 in
  /home/behavior/ext/ckpt/ft/radio_lora_v1 (45 GB). Nothing ran on the GPU 11-21 -> 11-27 (no follow-up was queued).
- Offline (eval_offline.py, 5 held-out demos, mean |pred - demo| over 32-step chunks; "hold" = keep current joints):

  | group | step 1000 | step 4999 | hold |
  |---|---|---|---|
  | base vel | 0.027 | 0.023 | 0.031 (zero vel) |
  | torso | 0.013 | 0.009 | 0.011 |
  | left arm | 0.021 | 0.015 | 0.011 |
  | right arm | 0.025 | 0.023 | 0.016 |
  | right grip | 0.20 | 0.09 | - |
- Arms are still worse than "hold still", also on *training* episodes (10-50: 0.051 vs 0.042 at k=31), so it is
  not overfitting. No offset bug: pred[0] is within 0.002 rad of the current state. Reading: undertrained (0.2 epoch)
  and/or sampling noise on slow joints; closed loop decides. Diagnostic: scratchpad arm_diag.py (per-chunk-step error).
- Closed loop: steering/ft/ft_closed_loop.py CKPT NAME [N] [STEPS] (campaign harness, runs/ft/NAME).

## Organisers' released pi0.5 radio checkpoint (= mentor's "vanilla pi0.5", 2026-11-27)
- Source: https://behavior.stanford.edu/challenge/baselines.html "Provided checkpoints", pi0.5 turning_on_radio,
  Google Drive id 1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2 (17.3 GB archive). Trained with github.com/wensi-ai/openpi
  (branch behavior, commit 0cc8e35, config pi05_b1k: full fine-tune, bs64, 50k steps, 2026 demos). behavior26
  (huggingface imaan3/behavior26-pi05-radio) measured it at Q = 0.111 (n=27) and fine-tuned it to 9/20.
- Fetch: unit dl-released -> /home/behavior/ext/mentor/released_pi05_radio. Env: uv sync in
  /home/behavior/ext/mentor/openpi (unit mentor-env).
- Serve: steering/mentor_server.py (their B1KPolicyWrapper, receding horizon K=16; robot_r1 -> robot key rename).
- Deleted (user OK): ckpt/ft/tiny_lora, radio_lora_v1/{2000,3000,4000}.
- Smoke (val pairs 242:0, 295:1, 3000 steps): our lora_v1 step 4999 0/2 (navigates to the table, reaches near the
  radio, drifts off); released baseline 0/2 (reaches the radio fast, right hand on the radio, never toggles, base
  drifts). Full 20-pair eval of the released checkpoint: unit released-val20 -> runs/ft/released_val20.
- **Released checkpoint, 20 val pairs (3000 steps, mentor_server K=16): 2/20** (successes: 211:2@2378 197:16@2450 ). Matches the
  mentor's ~10% and behavior26's Q=0.111. Typical failure: hand on the radio, no toggle, base drifts.
