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
   Same 61-d layout / indices otherwise.
4. Images: Comet authors recommend rendering full-res (head 720x720, wrists 480x480) then
   resize_with_pad to 224 on the server. Fork default renders 224x224 directly.
5. Timeout: 2025 = 2x mean demo length = 4299 steps; fork default = 1.5x = 3224.
6. Instances: 2025 validation = first 10 of `test_instances.csv` for radio
   (242, 295, 211, 203, 109, 181, 197, 187, 214, 139). The 300 train-range radio TRO files are
   byte-identical in the 2025 and 2026 instance zips, so these can be loaded with fork `--mode train`.
   None of them are among Comet's 200 training-demo instances.
7. Sim/assets: Isaac Sim 4.5 + assets 3.7 -> Isaac Sim 5.1 + assets 3.9. Can't fix; expect some drift.
8. Comet's submission zip has no `assets/` (norm stats). Use
   `sunshk/openpi_comet/pi05-b1kpt12-cs32/assets/...` (radio is in pt12's task set).
- Same in both: controllers, reset joint pose, head aperture 40, 23-d action layout, robot start poses.

## Reference numbers
- Comet authors (GitHub issue #18/#19): radio success varies 40-80% across reruns; ~70% after RFT.
