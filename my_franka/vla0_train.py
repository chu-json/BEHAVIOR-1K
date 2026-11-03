"""
Experiment 2: a VLA-0-style policy (Goyal et al., "VLA-0: Building State-of-the-Art VLAs with Zero Modification",
arXiv 2510.13054): a stock VLM, Qwen2.5-VL-3B-Instruct, fine-tuned to WRITE robot actions as text. No action head,
no new tokens: the answer is a string of integers.

Same data as our diffusion policy (the 962 successful wrist-camera demos, data/vision/demos_w*_c*.hdf5):
    input   wrist RGB (128x128, upscaled to 252x252 so Qwen gets 81 visual tokens) + the 7 proprio numbers as text
    output  the next 8 actions x 4 numbers (dx, dy, dz, gripper), each mapped from [-1, 1] to an integer 0..1000

VLA-0 prompt template (plus a state line):
    "Analyze the input image and predict robot actions for the next 8 timesteps. Each action has 4 dimensions.
     Robot state: s1 ... s7. Output a single sequence of 32 integers (0 - 1000 each), representing the 8 timesteps
     sequentially. Provide only space-separated numbers. Nothing else."

Scaled down to one RTX 3090: LoRA on the language model (vision encoder frozen) instead of VLA-0's full fine-tune
(64 epochs, batch 192, 8 x A100, ~32 h), and a fraction of an epoch-to-a-few-epochs budget. Stated as such in results.

    python my_franka/vla0_train.py --steps 3000
Output: my_franka/data/vla0/<run>/ (LoRA adapter, state normalization, training log)
"""
import argparse
import glob
import json
import math
import os
import time

import cv2
import h5py
import numpy as np
import torch as th
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.path.join(HERE, "data", "vlm", "Qwen2.5-VL-3B-Instruct")
DEMOS = os.path.join(HERE, "data", "vision", "demos_w*_c*.hdf5")
OUT_ROOT = os.path.join(HERE, "data", "vla0")
HORIZON, ADIM, BINS = 8, 4, 1000
IMG = 252  # 9 x 28: no internal resize


def prompt_for(state_ints):
    return (f"Analyze the input image and predict robot actions for the next {HORIZON} timesteps. Each action has "
            f"{ADIM} dimensions. Robot state: {' '.join(map(str, state_ints))}. Output a single sequence of "
            f"{HORIZON * ADIM} integers (0 - {BINS} each), representing the {HORIZON} timesteps sequentially. "
            f"Provide only space-separated numbers. Nothing else.")


def actions_to_text(chunk):
    """(H, 4) floats in [-1, 1] -> 'i i i ...'"""
    ints = np.clip(np.round((np.asarray(chunk) + 1.0) / 2.0 * BINS), 0, BINS).astype(int)
    return " ".join(map(str, ints.reshape(-1)))


def text_to_actions(text):
    """Inverse of actions_to_text; tolerant of short / garbled answers (pads with the last parsed action)."""
    nums = [int(x) for x in text.replace(",", " ").split() if x.lstrip("-").isdigit()]
    if not nums:
        return None
    nums = nums[: HORIZON * ADIM]
    while len(nums) % ADIM:
        nums.append(BINS // 2)
    a = np.array(nums, dtype=np.float32).reshape(-1, ADIM) / BINS * 2.0 - 1.0
    return a


class StateNorm:
    """Proprio -> integers 0..1000 using the dataset's min/max per dimension."""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)

    def ints(self, s):
        x = (np.asarray(s, np.float32) - self.lo) / np.maximum(self.hi - self.lo, 1e-6)
        return np.clip(np.round(x * BINS), 0, BINS).astype(int).tolist()

    def to_json(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


def upscale(img):
    return cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_LINEAR)


def build_messages(img_pil, prompt, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img_pil}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


class DemoFrames(th.utils.data.Dataset):
    """Every @every-th frame of every successful demo; returns (image uint8, state, 8-step action chunk)."""

    def __init__(self, every=3):
        self.index, self.files = [], sorted(glob.glob(DEMOS))
        lo, hi = None, None
        for fi, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                for key in f.keys():
                    g = f[key]
                    if not g.attrs.get("success", False):
                        continue
                    st = g["state"][:]
                    lo = st.min(0) if lo is None else np.minimum(lo, st.min(0))
                    hi = st.max(0) if hi is None else np.maximum(hi, st.max(0))
                    for t in range(0, len(st), every):
                        self.index.append((fi, key, t))
        self.norm = StateNorm(lo, hi)
        self._open = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, key, t = self.index[i]
        if fi not in self._open:
            self._open[fi] = h5py.File(self.files[fi], "r")
        g = self._open[fi][key]
        chunk = g["action"][t:t + HORIZON]
        if len(chunk) < HORIZON:  # repeat the last action past the episode end
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], HORIZON - len(chunk), axis=0)])
        return g["images"][t], g["state"][t], chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000, help="optimizer steps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--every", type=int, default=3, help="use every Nth frame")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--run", default="lora_r32")
    parser.add_argument("--max-hours", type=float, default=6.0)
    args = parser.parse_args()
    out = os.path.join(OUT_ROOT, args.run)
    os.makedirs(out, exist_ok=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    ds = DemoFrames(every=args.every)
    json.dump(ds.norm.to_json(), open(os.path.join(out, "state_norm.json"), "w"))
    print(f"{len(ds)} training frames from {len(ds.files)} files (every {args.every}th frame of successful demos)", flush=True)

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=th.bfloat16, device_map="cuda",
                                                               attn_implementation="sdpa")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      exclude_modules=r".*visual.*")  # language model only; vision encoder frozen
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def collate(batch):
        pils, full_texts, prompt_texts = [], [], []
        for img, st, chunk in batch:
            pil = Image.fromarray(upscale(img))
            prompt = prompt_for(ds.norm.ints(st))
            answer = actions_to_text(chunk)
            pils.append(pil)
            full_texts.append(processor.apply_chat_template(build_messages(pil, prompt, answer), tokenize=False))
            prompt_texts.append(processor.apply_chat_template(build_messages(pil, prompt), tokenize=False,
                                                              add_generation_prompt=True))
        enc = processor(text=full_texts, images=pils, return_tensors="pt", padding=True)
        enc_p = processor(text=prompt_texts, images=pils, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        for i in range(len(batch)):
            n_prompt = int(enc_p["attention_mask"][i].sum())
            labels[i, :n_prompt] = -100  # loss only on the answer (the action integers + end token)
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    loader = th.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6, collate_fn=collate,
                                      drop_last=True, persistent_workers=True)
    opt = th.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warm = 100
    sched = th.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    model.train()
    step, t0, log = 0, time.time(), []
    it = iter(loader)
    while step < args.steps:
        tot = 0.0
        for _ in range(args.accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            tot += loss.item()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0:
            el = time.time() - t0
            samples = step * args.batch * args.accum
            log.append(dict(step=step, loss=tot, lr=sched.get_last_lr()[0], hours=el / 3600))
            print(f"step {step}/{args.steps} loss {tot:.4f} lr {sched.get_last_lr()[0]:.2e} | {samples} samples "
                  f"({samples / len(ds):.2f} epochs) | {samples / el:.1f} samples/s | {el / 60:.0f} min", flush=True)
        if step % 500 == 0 or step == args.steps:
            model.save_pretrained(os.path.join(out, "adapter"))
            json.dump(dict(args=vars(args), log=log), open(os.path.join(out, "train_log.json"), "w"))
        if (time.time() - t0) / 3600 > args.max_hours:
            print("time budget reached", flush=True)
            model.save_pretrained(os.path.join(out, "adapter"))
            break
    print(f"saved adapter to {out}/adapter", flush=True)


if __name__ == "__main__":
    main()
