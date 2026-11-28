# Research directions (living document)

Started 2026-11-20 after discussion with postdoc/PhD mentors. NOTES.md = lab log (what ran, what broke);
this file = where we are going and why. LITERATURE.md = paper survey.

## Why we are changing course
Until now we fed policies inputs they were never trained on. Base pi0.5 never saw this robot's 23-number state, its
action layout or these cameras; steering it was "blind" work (0/98, see NOTES 2026-11-17). Comet and RLC were
trained on BEHAVIOR, but we can only steer them at inference time. New plan: **fine-tune pi0.5 ourselves**, so we
control how it is trained, and study what fine-tuning does to steerability.

## Core research question
**Can we fine-tune a policy on one task / embodiment and keep its steerability?**
Same task (turning_on_radio), but the policy should still respond to:
- language (instructions that change *how* the task is done),
- scene perturbations (clutter, partial occlusion, search),
- changed object geometry (button position/size, radio shape) that requires different behavior.

Second question: **WAM vs VLA.** Does a world-action model (CyberWAM, 3rd place BEHAVIOR Challenge 2026) respond
better to scene perturbations and language than a VLA (pi0.5)? Mentor's hypothesis: WAMs handle scene
perturbations (clutter, slight occlusion) better.

## User's notes (2026-11-20, kept verbatim in meaning)
- Fine-tuning balanced with steerability: how to fine-tune on the embodiment only so it stays just as steerable.
- Fine-tune on masked object geometry, e.g. with the radio outline drawn in (connects to tonight's V3 run).
- Cluttered scene; target partially occluded; the robot has to search for it.
- Use the WAM under scene perturbations, and also try language.
- Fine-tune pi0.5 on the dataset.
- Define the different types of steerability, test each, report results.
- Literature study: which steerability papers exist, which type of steerability each claims a contribution on.
- Triviality check: if a VLM can solve it (e.g. "draw a circle around the radio" and feed that in), the problem is
  too easy. Detection + visual prompt is a baseline, not a contribution.
- More interesting: dynamic changes that force the *behavior* to change (button position, button size and
  placement, radio shape) while steerability is preserved.
- Mentor's vanilla pi0.5 fine-tune reached only ~10% on radio. We may tune hard for this task since we only study
  steerability on it. Optional later: our own scripted demos + BC (collection may be expensive).

## Steerability taxonomy (what we will measure)
| Type | Test on turning_on_radio | Metric |
|---|---|---|
| L1 motion-language | "stay still", "turn left/right", "raise left/right arm", "move forward" (existing motor prompts) | paired contrast vs nonsense prompt (motor_contrasts.py) |
| L2 how-to language | "press with your left / right hand", "without picking the radio up", "pick up the radio first" | which hand presses, whether radio lifted, success |
| L3 paraphrase | P1-P4 paraphrases of the task | success unchanged |
| L4 goal-changing language | "pick up the radio", "push the radio away", "point at the radio" (no toggle) | goal-specific checks |
| S1 clutter | 3-10 distractor objects on the radio's table | success, time |
| S2 partial occlusion | occluder in front of the radio from the start view | success, time |
| S3 search | radio out of initial view / moved within the room | success, time, search path |
| S4 radio pose | yaw rotation (button facing away), shifted on table | success, behavior change |
| G1-G3 geometry | button marker moved / resized, radio scaled, other radio model | success; does the hand go to the new button |
| V visual prompting | outline / box drawn on the target (V1-V4 runs) | success; does the policy follow a moved mark |
Same 20 (setup, seed) pairs per cell, paired sign tests (steer_report.py), plus a same-day baseline (run-to-run
swing is about +-4/20, NOTES 2026-11-20).

## Models to compare
| Id | Model | Why |
|---|---|---|
| M0 | base pi0.5 (no BEHAVIOR training) | upper bound on language steerability, lower bound on task skill |
| M1 | mentor's vanilla pi0.5 fine-tune (checkpoint coming) | standard full fine-tune reference |
| M2 | our LoRA fine-tune, radio only (steering/ft) | our controllable baseline |
| M3 | variants of M2 (below) | the steerability-preserving methods |
| C-A / C-pt50 | Comet radio SFT / Comet multi-task pt50 | existing data: task-specific vs multi-task fine-tune |
| RLC | 2025 winner (task embeddings, no text) | control: no language channel at all |
| WAM | CyberWAM | world-action model vs VLA |

## Fine-tuning variants (M3) to test whether steerability survives
1. LoRA rank sweep / LoRA on action expert only (VLM frozen).
2. Knowledge-insulation style: stop gradients from the action expert into the VLM (Driess et al. 2025).
3. Language-augmented fine-tune: relabel segments of the demos with motion-level language (e.g. "turn left" when
   the base turns left, "raise the right arm" when it does) so the instruction channel stays informative.
4. Co-training with a small amount of non-radio BEHAVIOR data (embodiment-general) vs radio-only.
5. Outline/mask-augmented fine-tune: draw the radio outline from the dataset's seg_instance_id videos during
   training, then test whether moving the drawn outline steers the policy (visual-prompt steerability).
6. Geometry/pose augmentation if we can generate data (scripted demos, later).

## Hypotheses
- H1: radio-only full fine-tune kills language steerability (Comet A already ignores prompts); multi-task (pt50),
  LoRA and frozen-VLM fine-tunes keep more. Measurable now with L1-L3 on existing Comet checkpoints.
- H2: at equal clean success, the WAM degrades less under S1-S3 than pi0.5.
- H3: outline-augmented fine-tuning gives a policy steerable by drawing (V tests), unlike Comet (V1 = 3/20).
- H4: interesting cases are behavior-changing (G1-G3); detection + overlay (a VLM's job) should not suffice there.

## Plan
Phase 0 — infrastructure (now)
- [x] Fine-tuning pipeline for pi0.5 on the 200 radio demos (steering/ft, LoRA on one 3090). First run `ft-radio`
  (LoRA, bs16, 5000 steps, ~0.2 epoch) running 2026-11-20 16:50 -> ~03:00; 3090 is compute-bound at ~2.4 samples/s,
  so undertraining is the main risk (next: stop-gradient prefix for a ~3x faster expert-only variant).
- [x] Download the radio videos incl. seg; preprocessed to /home/behavior/ext/data/radio_ft (18 GB, masks for 3 cams).
- [x] Perturbation tooling (steering/perturb.py; run_rollouts --clutter N / --occluder / --radio-yaw-deg): smoke test
  queued after V3. Button offset/size deferred: the visible button is part of the radio mesh (needs asset work).
- [ ] Steerability suite runner: one command runs the taxonomy table for a given model/server.
- [x] Literature survey (LITERATURE.md); CyberWAM weights not public. Env + base assets being prepared.
Phase 1 — baselines on the suite: Comet A, Comet pt50, base pi0.5 (L1 only), RLC, M1 when it arrives.
Phase 2 — our fine-tunes M2/M3 on the suite; plot success vs steerability (the trade-off curve is the result).
Phase 3 — WAM on the same suite once weights are available.
  WAM readiness (CyberWAM; details NOTES "CyberWAM setup (2026-11-20)"):
  - [x] venv /home/behavior/ext/cyberwam/.venv (torch 2.7.1+cu128), their 511 tests pass on CPU
  - [x] inference base assets in /home/behavior/ext/cyberwam/checkpoints (33 GB: Wan2.2 DiT + VAE, DINOv3, T5, tokenizer)
  - [x] radio prompt embedding + 4 steering variants cached; steering/wam_text_embed.py embeds any new prompt (CPU)
  - [x] client: run_rollouts.py --policy-profile cyberwam (RGB-D full res, robot-frame base_qvel, K=32 replay)
  - [x] server: steering/wam_server.py (preflight, stub, episode seed, hot-swappable prompt); stub <-> client tested
  - [ ] model_a from the mentor -> `wam_server.py --preflight-only` -> GPU memory/latency check -> sim stub smoke
  - [ ] baseline radio val10 x 2 seeds, then the steerability suite (prompt variants first)
Decision point after Phase 1: pick the 2-3 steerability types where models differ most and focus there.


## Update 2026-11-27: plan after the first fine-tunes
State: our LoRA-from-base run (ft-radio v1, ~0.2 epoch) 0/2 in closed loop; the organisers' released pi0.5 radio
checkpoint (= the mentor's vanilla fine-tune) ~1/9 so far on our 20 pairs: it reaches the radio and stalls at the
press. Comet ~50%, RLC 40%.
Diagnosis: not a data-quantity problem (Comet reaches ~50% with the same 200 demos; behavior26 took the released
checkpoint to 45% with 6k frozen-VLM steps on the same demos). Initialisation + recipe + last-cm precision.

Workstreams (in order):
1. **Competent controllable base (M2')**: continue-train the released checkpoint with the VLM frozen (behavior26
   recipe) -> unit ft-cont. Target 40-50% on our 20 pairs. Frozen VLM is also the steerability-preserving variant
   we wanted to test (knowledge-insulation-like), so M2' doubles as the first M3 point.
2. **Last-mile steering (PressAssist, steering/press_assist.py)**: policy drives/reaches, scripted IK press when the
   hand stalls within 15 cm of the button. Oracle-localized for now (the button position comes from the sim); if it
   works, swap in a perception estimate (depth + detection, as in S2). Fits the "last-mile deployment" theme
   directly: a failure-triggered, local intervention instead of whole-trajectory steering. Related:
   shared autonomy / residual & corrective policies (see LITERATURE.md: RoboMonkey, V-GPS, FOREWARN; add "last-
   mile" refs).
3. **Steerability suite (Phase 1)** on Comet, released, M2': L1-L3 prompts + S1/S2/S4 perturbations (tools ready).
4. **Scripted demos for perturbed/geometry variants**: ApproachAssist + PressAssist = a full scripted radio policy;
   record demos with radio yaw / clutter / occluder (and later button geometry) -> fine-tune and measure whether
   behavior-changing robustness can be added while steerability is kept (H4). Toggle primitive in OmniGibson is not
   implemented and curobo is not installed, hence our own IK press.

## What the literature says (details: LITERATURE.md, 2026-11-20 survey)
- Closest prior work: "Flatness Preserves Instruction Following in VLAs" (arXiv 2606.23641) names "instruction
  blindness" after fine-tuning and fixes it with sharpness-aware minimisation; Knowledge Insulation (2505.23705),
  VLM2VLA (2509.22195) preserve VLM knowledge. These are our M3 recipes.
- WAM vs VLA robustness: "Do World Action Models Generalize Better than VLAs?" (2603.22078) - WAMs stronger on
  noise/lighting/layout, weaker on camera/initial-state shifts, >= 4.8x slower; tabletop only (LIBERO-Plus,
  RoboTwin2.0-Plus).
- Language grounding metrics: LIBERO-PRO garbage-instruction check, LIBERO-CF / LangGap counterfactual grounding
  (pi0.5 grounds only ~31%); "Learning What to Say to Your VLA" (2606.12299): some tasks are not language-steerable.
- Gaps we can fill: (1) steerability before vs after fine-tuning, across recipes, on one long-horizon mobile task;
  (2) changed object geometry and partial occlusion of the target (essentially unstudied for VLAs/WAMs);
  (3) WAM vs VLA on BEHAVIOR with matched data/embodiment/eval; (4) can a WAM's predicted future flag a
  perturbation (steering / failure signal); can fixed-prompt WAMs be steered by language at all.
- CyberWAM: code public (github.com/xiaomi-research/cyberwam, Apache-2.0); weights "coming soon" (not on HF) ->
  need them from the mentor. Wan2.2-TI2V-5B video backbone + action DiT, trained on the 20k BEHAVIOR-1K 2026 demos,
  RGB+depth input, prompt = fixed template per task (T5), 32x23 chunks; radio 7/10 in their eval; v3.9.3 evaluator.

## Open questions for the mentors
- Mentor's checkpoint: which data (radio only or all 50 tasks), action space (absolute / delta), config?
- CyberWAM weights: public or from the mentor (see LITERATURE.md once the survey finishes).
- Is scripted-demo collection in scope for geometry changes (G1-G3)?
