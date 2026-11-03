"""
Zero-shot pi0.5-DROID, wrapped for OmniGibson.

Checkpoint: lerobot/pi05_droid = pi0.5 fine-tuned by Physical Intelligence on DROID (a large Franka Panda dataset).
It was never trained on our simulator, our scene, or our camera placement; "zero-shot" means we use it as released.

DROID conventions (verified against the released normalization statistics; the action scale is from memory of the
DROID codebase, see JOINT_DELTA_PER_UNIT):
  observation   base_0_rgb        exterior camera image (224x224)
                left_wrist_0_rgb  wrist camera image
                right_wrist_0_rgb (not used by DROID; passed as a masked, empty camera)
                state             8 numbers: 7 joint angles (rad) + gripper position (0 open .. 1 closed)
                prompt            language instruction
  action        15-step chunk at 15 Hz; each step 8 numbers:
                7 normalized joint velocities in about [-1, 1] and 1 gripper position (0 open .. 1 closed)

Pipeline (same as openpi): state -> quantile-normalized to [-1, 1] -> discretized into 256 bins and written into the
text prompt ("Task: ..., State: ...;\\nAction: ") -> tokenized with the PaliGemma tokenizer; images -> [0, 1] floats;
model outputs normalized actions -> un-normalized with the DROID quantiles.

The tokenizer file is the public copy of PaliGemma's sentencepiece model (the Hugging Face repo is license-gated).
"""
import json
import os

import numpy as np
import sentencepiece as spm
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
PI0_DIR = os.path.join(HERE, "data", "pi0")
CKPT_DIR = os.path.join(PI0_DIR, "pi05_droid")
STATS_PATH = os.path.join(PI0_DIR, "droid_norm_stats.json")
TOKENIZER_PATH = os.path.join(PI0_DIR, "paligemma_tokenizer.model")

IMAGE_KEYS = {"exterior": "observation.images.base_0_rgb", "wrist": "observation.images.left_wrist_0_rgb"}
MAX_TOKENS = 200
EPS = 1e-6  # openpi's quantile-normalization epsilon
# DROID turns a normalized joint-velocity action v in [-1, 1] into a joint-position step of v * 0.2 rad per control
# step (control at 15 Hz). This constant is from memory of the DROID robot code, not verified; it is a CLI option.
JOINT_DELTA_PER_UNIT = 0.2


class PI05DroidZeroShot:
    def __init__(self, device="cuda", dtype="bfloat16"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        cfg = PreTrainedConfig.from_pretrained(CKPT_DIR)
        cfg.device, cfg.dtype = device, dtype
        self.policy = PI05Policy.from_pretrained(CKPT_DIR, config=cfg)
        self.policy.to(device).eval()
        self.device = device
        self.chunk = cfg.chunk_size
        self.sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
        stats = json.load(open(STATS_PATH))
        stats = stats.get("norm_stats", stats)
        self.s_q01, self.s_q99 = (np.array(stats["state"][k][:8], dtype=np.float64) for k in ("q01", "q99"))
        self.a_q01, self.a_q99 = (np.array(stats["actions"][k][:8], dtype=np.float64) for k in ("q01", "q99"))

    # ---- inputs -----------------------------------------------------------------------------------------------
    def _tokens(self, prompt, state_norm_padded):
        """openpi's pi0.5 prompt: the normalized state is discretized to 256 bins and spelled out in the text."""
        bins = np.linspace(-1, 1, 256 + 1)[:-1]
        discretized = np.digitize(state_norm_padded, bins=bins) - 1
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        text = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "
        ids = [self.sp.bos_id()] + self.sp.encode(text)
        ids = ids[:MAX_TOKENS]
        n = len(ids)
        tokens = th.zeros(1, MAX_TOKENS, dtype=th.long)
        mask = th.zeros(1, MAX_TOKENS, dtype=th.bool)
        tokens[0, :n] = th.tensor(ids)
        mask[0, :n] = True
        return tokens.to(self.device), mask.to(self.device), text

    def normalize_state(self, state8):
        s = (np.asarray(state8, dtype=np.float64) - self.s_q01) / (self.s_q99 - self.s_q01 + EPS) * 2.0 - 1.0
        return np.concatenate([s, np.zeros(32 - 8)])  # openpi zero-pads AFTER normalizing

    # ---- inference --------------------------------------------------------------------------------------------
    @th.inference_mode()
    def predict(self, exterior_rgb, wrist_rgb, state8, prompt):
        """
        exterior_rgb, wrist_rgb: uint8 (H, W, 3);  state8: 7 joint angles + gripper position (0 open .. 1 closed).
        Returns a (chunk, 8) float array: 7 normalized joint velocities + gripper position per 15 Hz step.
        """
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        def img(a):
            return th.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float().div(255)[None].to(self.device)

        tokens, mask, self.last_text = self._tokens(prompt, self.normalize_state(state8))
        batch = {
            IMAGE_KEYS["exterior"]: img(exterior_rgb),
            IMAGE_KEYS["wrist"]: img(wrist_rgb),  # right_wrist_0_rgb is omitted -> LeRobot pads it as an empty camera
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: mask,
        }
        norm_actions = self.policy.predict_action_chunk(batch)[0, :, :8].float().cpu().numpy().astype(np.float64)
        # quantile un-normalization (inverse of the above)
        actions = (norm_actions + 1.0) / 2.0 * (self.a_q99 - self.a_q01 + EPS) + self.a_q01
        return actions
