# Literature: steerability of robot foundation policies, and CyberWAM

Compiled 2026-11-20 by web research only (no GPU jobs). Every arXiv id below was checked against the arXiv API: the
title and first authors returned by the API match the paper named. Venues are from memory and marked `(venue?)`
where unsure. Numbers come from abstracts, project pages or the PDF named. They were not re-measured.

---

## Part A: Xiaomi CyberWAM (BEHAVIOR Challenge 2026)

### Availability (the short answer)

| Item | Status (checked 2026-11-20) |
|---|---|
| **Code** | **Public.** https://github.com/xiaomi-research/cyberwam (Apache-2.0; created 2026-11-12, last push 2026-11-19; no GitHub releases) |
| **Weights** | **Not public.** README "Released weights" table: `model_a` (step 115,884, Q 0.606 on the public 1000-rollout set), Download = **"coming soon"**. The project page also says "Weights coming soon". The HF org `xiaomi-research` has 36 models and none is CyberWAM. HF model search for "cyberwam" returns nothing. No HF id exists yet, so no file size is known. |
| Weights licence (announced) | Apache-2.0 (README: "Code and released weights are under the Apache License 2.0"; fine-tuned from Wan2.2-TI2V-5B, also Apache-2.0). No gating mentioned. |
| Base assets you must fetch yourself | Wan2.2-TI2V-5B + Wan2.2 VAE; DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`; DINOv3 is released under Meta's DINOv3 licence and may need terms accepted on HF; not checked); the Wan2.1-T2V-1.3B tokenizer and T5 for text embeddings; an ActionDiT init (training only, built by `scripts/preprocess_action_dit_backbone.py`) |
| Tech report | https://xiaomi-research.github.io/cyberwam/cyberwam_tech_report.pdf (no arXiv version found) |
| Project page | https://xiaomi-research.github.io/cyberwam/ |
| Training data | HF dataset `behavior-1k/2026-challenge-demos` (LeRobot v3.0, revision `2add61313bac4f1a42363d00ad03bd45949941a8`) |

### What the model is

- **Authors:** Hao Li, Hao Liu, Zhengjun Wang, Diyun Xiang, Hangjun Ye, Tao Kong, Zhiying Du (Xiaomi Robotics, "Team Cyber").
- **Type:** a multi-stream **World Action Model**. It builds on Fast-WAM (arXiv 2603.16666) and follows the multi-stream
  design of Flex-π (arXiv 2608.10860). A video-diffusion transformer jointly denoises future RGB latents, future DINOv3
  features, future point maps (from depth) and a 32×23 action chunk, all on **one shared diffusion timestep**
  (rectified flow, shift λ=5).
- **Backbone:** Wan2.2-TI2V-5B DiT as the "video expert": 30 layers, width 3072, FFN 14,336, 24×128 heads. One set of
  weights is shared by the RGB, DINO and point-map streams, and each stream has only its own input projection and
  output head.
- **Action expert:** "ActionDiT", 30 layers, width 1024, FFN 4096, initialised from Wan2.2 by linear interpolation.
  It is a Mixture-of-Transformers: each expert keeps its own weights, and joint masked self-attention runs over the
  concatenated tokens. Attention is stream-local in the bottom and top 25% of layers and mixed in the middle 50%.
  Action tokens attend to everything, while video tokens cannot see actions.
- **Size:** the report does not state it. My estimate from Table 2 is about 5B (video expert) plus about 1B (action
  expert), so roughly 6B parameters. In bf16 that is about 12 GB of weights, before the frozen VAE and DINOv3-B.
- **Training:** all 20,000 BEHAVIOR-1K 2026 teleop demos (100 tasks, about 1,950 h, 211M frames), with no
  pretraining on other robot data. Settings: AdamW at lr 2e-4 (cosine), bf16, DeepSpeed ZeRO-1, per-GPU batch 32,
  115,884 steps (epoch 27 of 30). The config name `..._256gpu` suggests 256 GPUs. The proprio state is dropped (zeroed)
  on 10% of samples. They also tried SFT/RFT post-training, which "outperformed the pretrained model on only a few
  tasks".
- **Language:** the task prompt is the task name inserted into a fixed template ("A video recorded from a robot's
  point of view executing the following instruction: ..."). It is T5-encoded (128 tokens), precomputed **once per
  task**, and the blocks cross-attend to it. So it is language-*conditioned*, but trained on only 100 fixed
  strings. Free-form or modified instructions are untested and would need new T5 embeddings
  (`scripts/precompute_text_embeds.py`).

### Inference interface

- **Inputs:** RGB **and depth** from the head camera (720×720) and both wrist cameras (480×480). Depth is required
  because the point-map stream needs it at every step. Also the 23-D state (base velocity 3, torso 4, arms 7+7,
  gripper widths 1+1, extracted from the 61-D proprio) and the task prompt. Internally the views are resized and
  tiled into one 384×256 image (head 256², wrists 128²).
- **Outputs:** a 32-step chunk of 23-D actions at 15 Hz (2.1 s): base twist (3, normalised), absolute joint targets
  for torso (4) and arms (7+7), and a binary gripper command per side (±1). It is executed with `speed_2x` (each 15 Hz
  action becomes one 30 Hz control step, and the base path is SE(2)-resampled), with the whole chunk executed before
  re-planning (execution horizon 32). Sampling uses 20 denoising steps. Predicted futures are generated, but only the
  actions are executed.
- **Serving:** a WebSocket policy server in `challenge_docker/` (Docker; the submission used Python 3.11 and torch
  2.6.0+cu124), configured by env vars `FASTWAM_B1K_INFERENCE_STEPS`, `FASTWAM_TEMPORAL_STRATEGY`,
  `FASTWAM_EXECUTE_HORIZON` and `FASTWAM_RESAMPLING`.
- **BEHAVIOR eval:** the official evaluator **v3.9.3**, which is the version of our local checkout (`OmniGibson/setup.py`
  3.9.3). The README command is:
  `python -m omnigibson.eval.eval --task-name <task> --robot-config OmniGibson/omnigibson/eval/r1pro.yaml
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --mode public_test --instance-indices 0 --num-envs 1
  --num-rollouts 1 --replay-action-chunk-size 32 --host <ip> --port 8000 --output-dir <dir> --write-video`.
  `RGBDFullResWrapper` already exists locally.
- **GPU memory:** not stated anywhere. By my estimate it should fit inference on our 24 GB RTX 3090: about 12 GB of
  bf16 weights, plus a small token count (3 latent frames × 3 streams at 384×256), plus DINOv3-B. This is
  **unconfirmed**. Latency is unknown, and Fast-WAM reports 190 ms for its own setup.

### Results

- Public test set, 100 tasks × 10 instances: **Q = 0.606**, 412/1000 successes (Fast-WAM baseline 0.246). On the 50
  tasks of the 2025 edition it reaches 0.587, against 0.26 for the 2025 winner.
- **Turning on radio (task 0): Q 0.70, 7/10 successes.**
- **Official 2026 leaderboard** (`data/results.jsonl` in HF Space `behavior-1k/2026-challenge-leaderboard`, verified
  private-test entries, 20 instances per task): team "Cyber", method "WAM", **Q 0.596, success rate 39.6%**. That is
  **3rd by Q among verified entries**, behind The North Star (Hierarchical Physical RSI, 0.683) and IF1001 (Cosmos-3
  WAM, 0.615; arXiv 2611.13818, "2nd-place solution"). The challenge page does not list the official prize
  placements, so I could not confirm the prize rank.
- **Robustness evidence relevant to us** (report §4.3): in the radio task the radio is on the table in all 300
  training and 20 public instances. They moved it to the **floor** in public instance 301, and CyberWAM still walked
  to it, picked it up and turned it on. That is one anecdotal OOD rollout, not a systematic study, and the authors
  call a systematic OOD study future work. This is directly in our territory.

### Could / could not confirm

- **Confirmed:** architecture, inputs and outputs, training data, eval command and version, public-set scores,
  leaderboard entry, code licence, and that the weights are not yet released.
- **Not confirmed:** parameter count, GPU memory, latency, checkpoint file size, weight release date, official prize
  rank, and behaviour under modified or free-form prompts.

---

## Part B: literature on steerability (2023–2026)

**Steerability type codes:**

- **L**: language/instruction steerability (new or modified instructions, motion-level language, hierarchical
  corrections)
- **IT**: inference-time steering (guided sampling, best-of-N with verifiers or value functions, latent-noise or
  activation steering)
- **R**: robustness to scene perturbations (clutter, distractors, occlusion, lighting, camera, pose)
- **G**: adapting to changed object geometry or affordances
- **P**: preserving pretrained capabilities during fine-tuning
- **V**: visual prompting (marks, traces, masks)
- **W**: world models and WAMs for steering, planning or robustness
- **E**: evaluation/benchmark contribution (added category)

### B1. Inference-time policy steering

| # | Paper | Venue / year | What "steering" means | Types | Method (one line) | How it is measured |
|---|---|---|---|---|---|---|
| 1 | **ITPS**: Inference-Time Policy Steering through Human Interactions. Wang, Wang, Du et al. [arXiv 2411.16627](https://arxiv.org/abs/2411.16627) | ICRA 2025 (venue?) | Bias a frozen generative policy's samples toward a human's intent (point, sketch, physical correction) without fine-tuning | IT, L-adjacent | Guide diffusion sampling with alignment objectives; stochastic sampling gives the best trade-off between alignment and distribution shift | Alignment-to-intent vs. constraint violations or task success across 6 sampling strategies, in sim (Maze2D, Block Push) and real |
| 2 | **V-GPS**: Steering Your Generalists: Improving Robotic Foundation Models via Value Guidance. Nakamoto, Mees, Kumar, Levine. [arXiv 2410.13816](https://arxiv.org/abs/2410.13816) | CoRL 2024 | Re-rank a black-box generalist's sampled actions by a learned value | IT | Offline-RL (Cal-QL/IQL) Q-function over language and observation; pick the argmax of K sampled actions | Success on 12 tasks across 5 policies (Octo, OpenVLA, RT-1-X...) in SIMPLER and real WidowX |
| 3 | **RoboMonkey**: Scaling Test-Time Sampling and Verification for VLAs. Kwok, Agia, Sinha et al. [arXiv 2506.17811](https://arxiv.org/abs/2506.17811) | CoRL 2025 (venue?) | Test-time scaling: sample, perturb and verify | IT, R | Gaussian-perturbed samples plus majority vote, then a VLM verifier trained on synthetic action-preference data | +25% abs on OOD tasks, +9% in-distribution (SIMPLER, LIBERO-Long, real); fits an inference scaling law |
| 4 | **FOREWARN**: From Foresight to Forethought: VLM-In-the-Loop Policy Steering via Latent Alignment. Wu, Tian, Swamy, Bajcsy. [arXiv 2502.01828](https://arxiv.org/abs/2502.01828) | RSS 2025 (venue?) | Choose among policy samples by predicted outcome, judged against a language goal | IT, W, L | A latent world model predicts outcomes of candidate plans; a VLM aligned to those latents selects the plan | Real-robot success under new task descriptions and behaviour preferences, vs. VLM-only and policy-only |
| 5 | **DSRL**: Steering Your Diffusion Policy with Latent Space Reinforcement Learning. Wagenmaker, Nakamoto, Zhang et al. [arXiv 2506.15799](https://arxiv.org/abs/2506.15799) | arXiv 2025 (CoRL 2025?) | Adapt a frozen diffusion/flow policy by choosing its *input noise* | IT | RL over the latent-noise space (black-box, no weight change); also applied to π0 | Sample efficiency and success in sim (Robomimic, OpenAI Gym) and real, including adapting π0 |
| 6 | **Hume**: Introducing System-2 Thinking in VLA. Song, Qu, Yao et al. [arXiv 2505.21432](https://arxiv.org/abs/2505.21432) | arXiv 2025 | Value-guided candidate selection inside the VLA | IT | Value-query head scores sampled action candidates, plus cascaded denoising | LIBERO, SIMPLER, real-robot success |
| 7 | **Mechanistic interpretability for steering VLAs**. Häon, Stocking, Chuang, Tomlin. [arXiv 2509.00328](https://arxiv.org/abs/2509.00328) | CoRL 2025 (venue?) | Activation steering: push internal FFN directions such as "speed" or "up" | IT, L-adjacent | Project FFN activations onto the token basis, find semantic directions, add them at inference (π0, OpenVLA) | Zero-shot change in motion attributes (speed, height) in LIBERO and on a UR5 |
| 8 | **Learning What to Say to Your VLA: Mostly Harmless VLA Steering**. Jeong, Swamy, Bajcsy. [arXiv 2606.12299](https://arxiv.org/abs/2606.12299) | arXiv 2026 | Steer a *frozen* VLA through its language input, choosing prompts that help | L, IT, R | Search for language sequences that improve closed-loop success, distil them into a language-feedback policy, and gate it with a conformal "improvement head" so it does not harm | +24.7% sim, +65% real on seen envs; harmlessness guarantees under visual and semantic perturbations. **Explicitly finds that some tasks are not language-steerable** |
| 9 | **CAG**: When Vision Overrides Language: Counterfactual Failures in VLAs (LIBERO-CF). Fang, Feng, Jing et al. [arXiv 2602.17659](https://arxiv.org/abs/2602.17659) | arXiv 2026 | Strengthen language conditioning at inference | L, IT, E | Classifier-free-guidance-like: amplify (language-conditioned minus language-free) action difference | LIBERO-CF grounding rate and success; π0.5 only 30.8% grounding / 13.2% success; TF-CAG +9.7 grounding |
| 10 | **Real-Time Chunking**: Real-Time Execution of Action Chunking Flow Policies. Black, Galliker, Levine. [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) | arXiv 2025 (NeurIPS 2025?) | Inference-time inpainting guidance so new chunks agree with the executing chunk | IT | Guided flow sampling (inpainting) on π0.5 | Success and smoothness under latency; useful as a guidance mechanism for flow VLAs |

### B2. Language / hierarchical steerability

| # | Paper | Venue / year | Meaning | Types | Method | Measurement |
|---|---|---|---|---|---|---|
| 11 | **RT-H**: Action Hierarchies Using Language. Belkhale, Ding, Xiao et al. [arXiv 2403.01823](https://arxiv.org/abs/2403.01823) | RSS 2024 | Steer with *language motions* ("move arm forward") | L | Predict a language motion, then the action; humans intervene on the motion level | Success gain from motion corrections; generalisation to new tasks |
| 12 | **YAY Robot**: Yell At Your Robot. Shi, Hu, Zhao et al. [arXiv 2403.12910](https://arxiv.org/abs/2403.12910) | RSS 2024 | On-the-fly verbal corrections | L | High-level language policy plus a language-conditioned low-level policy; fine-tune the high level from corrections | Long-horizon bimanual success with and without corrections |
| 13 | **Hi Robot**: Open-Ended Instruction Following with Hierarchical VLAs. Shi, Ichter, Equi et al. (PI). [arXiv 2502.19417](https://arxiv.org/abs/2502.19417) | ICML 2025 (venue?) | Follow complex prompts and mid-task feedback ("no pickles") | L | VLM high-level reasoner emits atomic commands to π0; synthetic interaction data | Human-rated instruction accuracy and task progress on real tasks |
| 14 | **STEER**: Flexible Robotic Manipulation via Dense Language Grounding. Smith, Irpan, Gonzalez Arenas et al. [arXiv 2411.03409](https://arxiv.org/abs/2411.03409) | arXiv 2024 (ICRA 2025?) | Orchestrate modular, language-named skills (grasp angle, approach) | L, G | Dense skill annotation; a human or VLM composes primitives for new situations | Success on novel situations without new data |
| 15 | **Steerable Policies** for Embodied Reasoning and Hierarchical Control. Chen, Bhatia, Glossop et al. [arXiv 2602.13193](https://arxiv.org/abs/2602.13193) | arXiv 2026 | VLA accepting commands at many abstraction levels (subtask, motion, pixel point) | L, V | Train OpenVLA- and π0.5-based VLAs on synthetic multi-level commands; drive them with a learned reasoner or an ICL VLM | Real Bridge/WidowX generalisation and long-horizon tasks vs. MolmoAct and hierarchical baselines |
| 16 | **ECoT**: Robotic Control via Embodied Chain-of-Thought. Zawalski, Chen, Pertsch et al. [arXiv 2407.08693](https://arxiv.org/abs/2407.08693) | CoRL 2024 | Inspectable and editable reasoning steps | L | VLA reasons (plan, subtask, bbox, gripper) before acting; humans can edit the reasoning | OOD success; correction via edited reasoning |
| 17 | **RACER**: Rich Language-Guided Failure Recovery Policies. Dai, Lee, Fazeli, Chai. [arXiv 2409.14674](https://arxiv.org/abs/2409.14674) | arXiv 2024 (ICRA 2025?) | Rich language for recovery from failure | L | Augmented recovery data plus a VLM supervisor emitting rich instructions | RLBench success, including human-in-the-loop corrections |

### B3. Robustness / perturbation benchmarks

| # | Paper | Venue / year | Meaning | Types | Method | Measurement |
|---|---|---|---|---|---|---|
| 18 | **LIBERO-Plus**: In-depth Robustness Analysis of VLAs. Fei, Wang, Shi et al. [arXiv 2510.13626](https://arxiv.org/abs/2510.13626) | arXiv 2025 | Sensitivity to 7 perturbation axes | R, L, E | Controlled perturbations: layout, camera, robot init, language, light, background, sensor noise | Success drops from 95% to <30% for camera and init; **models largely ignore language** |
| 19 | **LIBERO-PRO**: Robust and Fair Evaluation of VLAs Beyond Memorization. Zhou, Xu, Tie et al. [arXiv 2510.03827](https://arxiv.org/abs/2510.03827) | arXiv 2025 | Memorisation vs. understanding | R, L, G, E | Perturb objects, initial states, instructions, environments | 90% falls to 0% under the generalised setting; output unchanged with corrupted instructions; grasps even when the target is replaced |
| 20 | **THE COLOSSEUM**: Benchmark for Evaluating Generalization. Pumacay, Singh, Duan et al. [arXiv 2402.08191](https://arxiv.org/abs/2402.08191) | RSS 2024 | 14 perturbation axes, **including object size**, plus distractors, lighting and camera | R, G, E | RLBench-based perturbation suite with 3D-printable real counterparts | 30–50% drop per axis, ≥75% combined; sim-to-real correlation R²=0.614 |
| 21 | **GemBench**: Towards Generalizable Vision-Language Robotic Manipulation. Garcia, Chen, Schmid. [arXiv 2410.01345](https://arxiv.org/abs/2410.01345) | ICRA 2025 (venue?) | 4 levels: new placements, rigid objects, articulated objects, long-horizon | R, G, E | Benchmark plus an LLM-guided 3D policy (3D-LOTUS++) | Success per generalisation level |
| 22 | **Do World Action Models Generalize Better than VLAs? A Robustness Study**. Zhang, Li, Rahmati et al. [arXiv 2603.22078](https://arxiv.org/abs/2603.22078) | arXiv 2026 | **WAM vs. VLA robustness** | R, W, E | Evaluate VLAs (π0.5...) and WAMs (Cosmos-Policy, LingBot-VA, Motus...) on LIBERO-Plus and the new RoboTwin 2.0-Plus | Cosmos-Policy 82.2% on LIBERO-Plus; LingBot-VA 74.2% vs. π0.5 58.6% on RoboTwin2.0-Plus; secondary summaries say WAMs are weak on camera and robot-init perturbations; WAMs ≥4.8× slower. Caveat: the WAM sets differ across benchmarks |
| 23 | **AGNOSTOS**: Limits of VLA Manipulation in Cross-task Generalization. Zhou, Ye, Liu et al. [arXiv 2505.15660](https://arxiv.org/abs/2505.15660) | arXiv 2025 | Zero-shot unseen tasks | L, G, E | 23 unseen tasks at 2 difficulty levels; X-ICM in-context baseline | Success on unseen tasks |
| 24 | **ArtiBench / ArtiBrain**. Wu, Wei, Wang et al. [arXiv 2511.20330](https://arxiv.org/abs/2511.20330) | arXiv 2025 | Articulated parts that vary in **geometry, scale and joint limits** | G, E | 5-level benchmark; modular VLM plus affordance memory | Success across part, instance and category variation |
| 25 | **VLA Models Are More Generalizable Than You Think** (FTM/FLA). Li, Zhang, Zhai et al. [arXiv 2512.02902](https://arxiv.org/abs/2512.02902) | arXiv 2025 | Camera and visual perturbations are a *spatial-modelling* misalignment | R, P | One-shot adapters on π0.5's vision (4K-param affine; low-rank ViT) beat full FT and match LoRA | LIBERO viewpoint success 48.5% → 90.8% |

### B4. Preserving pretrained capability (forgetting, instruction blindness)

| # | Paper | Venue / year | Meaning | Types | Method | Measurement |
|---|---|---|---|---|---|---|
| 26 | **RT-2**: VLAs Transfer Web Knowledge to Robotic Control. Brohan et al. [arXiv 2307.15818](https://arxiv.org/abs/2307.15818) | CoRL 2023 | Keep web semantics by co-training | P, L | Co-fine-tune the VLM on web VQA plus robot actions as tokens | Emergent semantic or symbol tasks; co-training vs. robot-only ablation |
| 27 | **π0**: A VLA Flow Model for General Robot Control. Black et al. [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) | RSS 2025 | Language-following base model | L, P | PaliGemma VLM plus a flow-matching action expert; pre-train then post-train | Language-following rate (with and without high-level VLM prompts) on real tasks |
| 28 | **π0.5**: VLA with Open-World Generalization. Physical Intelligence et al. [arXiv 2504.16054](https://arxiv.org/abs/2504.16054) | CoRL 2025 (venue?) | Generalise to new homes and follow instructions | L, R, P | Co-training on multi-robot, web, subtask-prediction and detection data; hierarchical inference | Success and **language following** in unseen homes; ablations removing web and cross-embodiment data |
| 29 | **Knowledge Insulating VLAs**: Train Fast, Run Fast, Generalize Better. Driess, Springenberg, Ichter et al. (PI). [arXiv 2505.23705](https://arxiv.org/abs/2505.23705) | arXiv 2025 (NeurIPS 2025?) | Adding an action expert *harms* VLM knowledge; insulate it | P, L | Stop gradients from the action expert into the VLM; train the VLM on FAST tokens plus VLM data | Language following, generalisation and training speed vs. naive π0-style training |
| 30 | **VLM2VLA**: Actions as Language, without Catastrophic Forgetting. Hancock, Wu, Zha et al. [arXiv 2509.22195](https://arxiv.org/abs/2509.22195) | arXiv 2025 | Keep VQA and multilingual instruction following after action FT | P, L | Express actions as natural language so LoRA-only FT stays in-distribution | VQA benchmarks plus 800+ real trials with new or multilingual instructions |
| 31 | **InstructVLA**: VLA Instruction Tuning. Yang, Li, Wang et al. [arXiv 2507.17520](https://arxiv.org/abs/2507.17520) | arXiv 2025 | Keep multimodal reasoning while learning actions | P, L | MoE adaptation plus VLA-IT co-training (650K samples) | Multimodal benchmarks plus SimplerEnv-Instruct (instruction-following variant) |
| 32 | **ChatVLA**: Unified Multimodal Understanding and Robot Control. Zhou, Zhu, Zhu et al. [arXiv 2502.14420](https://arxiv.org/abs/2502.14420) | arXiv 2025 | Avoid forgetting and task interference | P | Phased alignment training plus MoE | VQA scores plus real manipulation success |
| 33 | **Flatness Preserves Instruction Following in VLAs** (SAM). Zhang, Bisk. [arXiv 2606.23641](https://arxiv.org/abs/2606.23641) | arXiv 2026 | **"Instruction blindness" after fine-tuning**, which is exactly our question | P, L | Sharpness-aware minimisation during VLA fine-tuning on the same data | >55% better instruction following on LIBERO-CF, LIBERO-PRO, LangGap and real; beats weight-anchoring regularisers |
| 34 | **LangGap**: Diagnosing and Closing the Language Gap in VLAs. Hou, Zhao. [arXiv 2603.00592](https://arxiv.org/abs/2603.00592) | arXiv 2026 | π0.5 ignores language under fixed layouts | L, E | 4-D semantic perturbation; many tasks per identical layout | 0% → 90% (single-task) with augmentation; 0% → 28% multi-task |
| 35 | **BayesVLA**: Seeing to Act, Prompting to Specify. Xu, Zhu, Chen et al. [arXiv 2512.11218](https://arxiv.org/abs/2512.11218) | arXiv 2025 | Language forgetting from modality imbalance | P, L | Factorise into a visual-action prior and a language likelihood | Unseen instructions, objects, environments |
| 36 | **Bring Back Language to Actions**: parameter merging. Aissi, Sigaud, Soulier et al. [arXiv 2611.18891](https://arxiv.org/abs/2611.18891) | arXiv 2026 | Restore VLM grounding after VLA FT | P | SVD of the VLA–VLM task vector with learned per-component mixing | VLM benchmarks plus LIBERO success (3–4B) |
| 37 | **What Can RL Bring to VLA Generalization?** Liu, Gao, Wei et al. [arXiv 2505.19789](https://arxiv.org/abs/2505.19789) | arXiv 2025 (NeurIPS 2025?) | SFT vs. RL fine-tuning and OOD generalisation | P, R, L | PPO fine-tuning of OpenVLA vs. SFT | Visual, semantic and execution OOD splits: RL helps semantic and execution, equal on visual |
| 38 | **Continual learning for VLAs in the real world**. Zhu, Hong, Sun et al. [arXiv 2605.26820](https://arxiv.org/abs/2605.26820) | arXiv 2026 | Forgetting across sequential tasks | P | Experience replay with action normalisation | 10-task real stream; ER beats joint training |
| 39 | **OpenVLA-OFT**: Fine-Tuning VLAs: Optimizing Speed and Success. Kim, Finn, Liang. [arXiv 2502.19645](https://arxiv.org/abs/2502.19645) | RSS 2025 (venue?) | FT recipe; *negative example*: OFT shows the worst vision shortcuts in LIBERO-CF (#9) | P (neg.) | Parallel decoding, continuous actions, FiLM for language | LIBERO SR; FiLM was added *because* language was ignored on ALOHA |

### B5. Visual prompting

| # | Paper | Venue / year | Meaning | Types | Method | Measurement |
|---|---|---|---|---|---|---|
| 40 | **TraceVLA**: Visual Trace Prompting. Zheng, Liang, Huang et al. [arXiv 2412.10345](https://arxiv.org/abs/2412.10345) | ICLR 2025 | Overlay past point traces on the image | V, R | Fine-tune OpenVLA (and Phi-3-V 4B) on 150K trajectories with trace overlays | 137 SimplerEnv configs (+10%), real WidowX (3.5×) |
| 41 | **RoboPoint**: VLM for Spatial Affordance Prediction. Yuan, Duan, Blukis et al. [arXiv 2406.10721](https://arxiv.org/abs/2406.10721) | CoRL 2024 | Language to points (affordance) | V, G | Instruction-tune a VLM on synthetic point data | Point accuracy plus downstream manipulation |
| 42 | **MOKA**: Mark-Based Visual Prompting. Liu, Fang, Abbeel, Levine. [arXiv 2403.03174](https://arxiv.org/abs/2403.03174) | RSS 2024 | VLM chooses keypoints and waypoints on marked images | V, G | Mark-based VQA, then motion primitives | Real open-world task success |
| 43 | **PIVOT**: Iterative Visual Prompting. Nasiriany, Xia, Yu et al. [arXiv 2402.07872](https://arxiv.org/abs/2402.07872) | ICML 2024 | VLM selects among annotated candidate actions | V, IT | Iterative sample, annotate, select (like CEM) | Navigation and manipulation success |
| 44 | **ARRO**: Augmented Reality for RObots. Mirjalili, Jülg, Walter, Burgard. [arXiv 2505.08627](https://arxiv.org/abs/2505.08627) | arXiv 2025 | Mask distractors and overlay virtual guides | V, R | Open-vocab segmentation keeps gripper and target on a virtual grid background (train and test) | Success under background and distractor shifts; DP, Octo, OpenVLA, π0 |
| 45 | **MolmoAct**: Action Reasoning Models. Lee, Duan, Fang et al. [arXiv 2508.07917](https://arxiv.org/abs/2508.07917) | arXiv 2025 | **Trajectory-trace steering**: the user edits the predicted 2-D trace | V, L | Depth tokens, then a visual trace, then actions; open weights | SimplerEnv, LIBERO, OOD +23.3%; human-preference "trajectory steering" eval |

### B6. World models and World Action Models

| # | Paper | Venue / year | Meaning | Types | Method | Measurement |
|---|---|---|---|---|---|---|
| 46 | **UniPi**: Universal Policies via Text-Guided Video Generation. Du, Yang, Dai et al. [arXiv 2302.00111](https://arxiv.org/abs/2302.00111) | NeurIPS 2023 | Language-conditioned video planning | W, L | Text-to-video plan plus inverse dynamics | Combinatorial language generalisation |
| 47 | **VPP**: Video Prediction Policy. Hu, Guo, Wang et al. [arXiv 2412.14803](https://arxiv.org/abs/2412.14803) | ICML 2025 (venue?) | Policy on predictive video features | W | Fine-tuned video diffusion representations feed an action head | CALVIN, MetaWorld, real |
| 48 | **UVA**: Unified Video Action Model. Li, Gao, Sadigh, Song. [arXiv 2503.00200](https://arxiv.org/abs/2503.00200) | RSS 2025 (venue?) | Joint video-action latent | W | Decoupled video and action diffusion heads; can skip video at test time | Multi-task success, forward and inverse dynamics |
| 49 | **Cosmos Policy**: Fine-Tuning Video Models for Visuomotor Control and Planning. Kim, Gao, Lin et al. [arXiv 2601.16163](https://arxiv.org/abs/2601.16163) | arXiv 2026 | Video model as policy plus value for **test-time planning** | W, IT | Actions, future images and values encoded as latent frames in Cosmos-Predict2 | LIBERO 98.5%, RoboCasa 67.1%; best robustness on LIBERO-Plus in #22 |
| 50 | **Fast-WAM**: Do WAMs Need Test-time Future Imagination? Yuan, Dong, Liu, Zhao. [arXiv 2603.16666](https://arxiv.org/abs/2603.16666) | arXiv 2026 | Is the benefit from video *training* or from test-time imagination? | W | Video co-training, skip generation at test time (CyberWAM's base) | Removing co-training hurts much more than removing generation; 190 ms latency |
| 51 | **Flex-π**: Multi-Stream WAM with Compute Flexibility. Yan, Liu, Fan et al. [arXiv 2608.10860](https://arxiv.org/abs/2608.10860) | arXiv 2026 | RGB, point-map and DINO streams; run any subset | W | MoT with per-stream dropout | (Not read in detail) |
| 52 | **Genie Envisioner**. Liao, Zhou, Huang et al. [arXiv 2508.05635](https://arxiv.org/abs/2508.05635); **Motus**. Bi, Tan, Xie et al. [arXiv 2512.13030](https://arxiv.org/abs/2512.13030); **LingBot-VA** (Causal World Modeling for Robot Control). Li, Zhang, Luo et al. [arXiv 2601.21998](https://arxiv.org/abs/2601.21998); **WorldVLA**. Cen, Yu, Yuan et al. [arXiv 2506.21539](https://arxiv.org/abs/2506.21539) | 2025–26 | Joint video and action generation | W | MoT or AR video-action models | Robustness comparison in #22 |
| 53 | **DreamGen**: Generalization through Video World Models. Jang, Ye, Lin et al. [arXiv 2505.12705](https://arxiv.org/abs/2505.12705) | arXiv 2025 (CoRL 2025?) | Generate synthetic trajectories for new verbs and environments | W, L, R | Fine-tune a video model, generate "neural trajectories", label with IDM or latent actions | New-behaviour and new-environment success from a single-task dataset |
| 54 | **Ctrl-World**: Controllable Generative World Model. Guo, Shi, Chen, Finn. [arXiv 2510.10125](https://arxiv.org/abs/2510.10125) | arXiv 2025 | Evaluate and improve *instruction following* of a generalist in imagination | W, L, E | Multi-view action-conditioned WM (DROID); imagined rollouts for eval plus synthetic FT | Correlation with real eval; instruction-following gains after imagined-data FT (π0.5) |
| 55 | **Latent Policy Steering (LPS)**. Wang, Verghese, Schneider. [arXiv 2507.13340](https://arxiv.org/abs/2507.13340) | arXiv 2025 | WM plus value selects base-policy candidates | W, IT | Optical-flow-action WM pretrained cross-embodiment | Robomimic +10.6%, real |
| 56 | **V-JEPA 2**. Assran, Bardes, Fan et al. [arXiv 2506.09985](https://arxiv.org/abs/2506.09985) | arXiv 2025 | Latent WM planning (MPC) toward image goals | W | Action-conditioned latent predictor plus CEM | Zero-shot pick-and-place in new labs |

### B7. BEHAVIOR Challenge reports (context)

| Paper | Notes |
|---|---|
| Larchenko et al. 2025 1st place, [arXiv 2512.06951](https://arxiv.org/abs/2512.06951) | π0.5 task adaptation, correlated noise, Q 0.26 |
| Openpi Comet, [arXiv 2512.10071](https://arxiv.org/abs/2512.10071) | 2025 runner-up π0.5; **our current policy** |
| IF1001, WAM 2nd place 2026, [arXiv 2611.13818](https://arxiv.org/abs/2611.13818) | Cosmos-3 WAM plus LoRA adapters, progress routing; Q 0.615 private; code and weights not public |
| Larchenko et al. 2026, [arXiv 2611.14432](https://arxiv.org/abs/2611.14432) | One π0.5 for 100 tasks, 9th, Q 0.392; 14 tasks routed to a coding agent |
| CyberWAM tech report (Part A) | WAM, Q 0.596 private / 0.606 public; radio-on-floor OOD anecdote |

---

## Synthesis

### (i) Coverage

- **Well covered:**
  - **Inference-time steering by sampling plus selection** (V-GPS, RoboMonkey, FOREWARN, LPS, Hume, ITPS, DSRL).
  - **Visual-perturbation robustness benchmarks** on *tabletop* LIBERO and RoboTwin (LIBERO-Plus, LIBERO-PRO,
    COLOSSEUM, RoboTwin2.0-Plus).
  - **Forgetting of VLM knowledge**, measured mostly by **VQA scores** (RT-2, KI, VLM2VLA, ChatVLA, InstructVLA,
    merging).
  - The 2026 wave of **"VLAs ignore language"** diagnostics (LIBERO-Plus, LIBERO-PRO, LIBERO-CF, LangGap).
- **Under-explored:**
  1. **Changed object geometry or affordance of the *same* object** (button moved or resized, different radio body).
     COLOSSEUM varies size, and GemBench and ArtiBench vary articulated instances, but no VLA/WAM paper studies
     *within-task* affordance relocation for a fine-tuned generalist.
  2. **Partial occlusion of the target** is almost never an explicit axis. Clutter or distractors and camera are, but
     not occlusion.
  3. **Long-horizon mobile manipulation**: every robustness benchmark above is a short tabletop task. Nothing covers
     BEHAVIOR-scale, mobile bimanual tasks (CyberWAM's single floor-radio rollout is the only data point).
  4. **Steerability as a property that is *lost* during task fine-tuning.** Only Flatness/SAM (#33) and
     VLM2VLA/KI measure *instruction following* (not VQA) before vs. after FT. None does so for π0.5 on a
     long-horizon embodied task, and none tracks *motion-level* language or perturbation adaptation as a function of
     FT recipe.
  5. **WAM steerability.** #22 compares WAM vs. VLA robustness, but nobody has measured whether WAMs are
     *language-steerable* (CyberWAM conditions on 100 fixed T5 prompts), or whether WAM imagination can be used to
     *steer* (FOREWARN-style) inside the same model. Cosmos Policy planning is closest.

### (ii) Where our questions would be novel

- **"Fine-tune π0.5 on one task/embodiment while preserving steerability."** Novel if we (a) define a *steerability
  battery* on BEHAVIOR turning_on_radio, covering motion-language prompts, counterfactual instructions, clutter,
  target occlusion, and moved or resized button or radio geometry, and (b) measure it **before vs. after** fine-tuning
  under recipes the literature proposes but never compares on one long-horizon embodied task:
  - full FT
  - LoRA
  - frozen VLM (expert-only)
  - Knowledge Insulation (stop-grad plus FAST plus VQA co-training)
  - SAM/flatness (#33)
  - weight merging (#36)
  - vision-only adapters (#25)
  - RL fine-tuning (#37)
  - DSRL noise steering (#5) as a no-weight-change baseline

  The claim to test is that "task success and steerability trade off, and recipe X keeps steerability at equal
  success". No paper reports that curve.
- **"WAM vs. VLA under scene perturbation."** Novel on BEHAVIOR because #22 is tabletop-only and uses disjoint WAMs.
  On BEHAVIOR we would have *same data, same embodiment, same eval*: CyberWAM (once weights drop) or IF1001 (not
  released) vs. Comet/π0.5, with the same perturbation battery. Two further angles are new:
  - CyberWAM predicts future frames, DINO features and point maps. We can test whether its *predicted future* detects
    the perturbation (prediction error as an OOD or failure signal), giving a steering hook the VLA lacks.
  - Prompt steerability of a WAM trained on fixed prompts vs. π0.5.

  Weights are needed before this can run: watch the repo. CyberWAM already scores 0.70 Q on radio, so a fair baseline
  exists.

### (iii) Reusable evaluation protocols

1. **LIBERO-Plus axes** (#18): perturb one factor at a time at graded severities (layout, camera, robot init, language,
   light, background, noise). Report success per axis and the drop vs. clean. Port to BEHAVIOR radio as distractor
   clutter, camera jitter, robot init pose, lighting, and background or texture.
2. **LIBERO-PRO language sanity checks** (#19): corrupted or garbage instructions and a target-replaced scene. If
   success is unchanged, the policy is not language-conditioned. This is a cheap test of whether our fine-tuned π0.5
   still listens.
3. **Counterfactual instruction grounding rate** (LIBERO-CF #9, LangGap #34): same scene, alternative feasible
   instruction (e.g. "pick up the radio and put it on the floor", "turn on the lamp"). Measure the *grounding rate*
   (it touched the right object) separately from success. CAG (language minus no-language guidance) is a ready
   inference-time baseline for flow π0.5.
4. **Modality ablation** (#9): language-only vs. vision-only inputs, as a quick diagnostic of shortcut reliance.
5. **Motion-level steerability** (RT-H #11, activation steering #7, Steerable Policies #15): issue "move left/higher/
   slower" and measure displacement of the end-effector or base along the commanded direction vs. no command (a
   signed effect size). This is an objective steerability metric we could log from the sim state.
6. **Harmlessness of steering** (#8): report whether a steering intervention *reduces* success vs. the original
   prompt, with conformal gating. Useful for "last-mile" claims.
7. **Inference-time scaling curve** (RoboMonkey #3, V-GPS #2): success vs. number of samples K with a
   verifier or value. Compare VLA vs. WAM (CyberWAM's predicted futures could act as its own verifier, as in Cosmos
   Policy planning).
8. **Geometry axis** (COLOSSEUM size #20, GemBench/ArtiBench instance levels #21/#24): scale the button or radio, move
   the button on the radio body, swap the radio model. Report success by severity. In BEHAVIOR, object scale and
   object swap are possible through the task instance or object config.
9. **CyberWAM-style single OOD relocation** (radio moved to the floor), scaled up to a systematic grid of
   placements, which is the study the CyberWAM authors explicitly leave open.

## Sources (non-arXiv)

- CyberWAM project page: https://xiaomi-research.github.io/cyberwam/
- CyberWAM tech report: https://xiaomi-research.github.io/cyberwam/cyberwam_tech_report.pdf
- CyberWAM code: https://github.com/xiaomi-research/cyberwam
- 2026 leaderboard data: https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard (files `data/results.jsonl`, `data/published_self_reported_results.jsonl`)
- Challenge page: https://behavior.stanford.edu/challenge/index.html
- RoboTwin 2.0-Plus: https://robot-robustness.github.io/RoboTwin2.0-Plus/ (linked from #22; not opened)
