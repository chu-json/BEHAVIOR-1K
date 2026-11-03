"""
Offline check of a VLA-0-style checkpoint, no simulator: on demo frames, does the fine-tuned Qwen write 32 parseable
integers, and how close are they to the demonstrated actions? (Training frames, so this checks the format and that
learning happened, not generalization; generalization is what vla0_eval.py measures.)

    python my_franka/vla0_offline_check.py --run lora_r32 --n 6 --device cpu
"""
import argparse
import json
import os
import time

import numpy as np
import torch as th
from PIL import Image

from vla0_train import BASE_MODEL, OUT_ROOT, DemoFrames, StateNorm, actions_to_text, build_messages, prompt_for, text_to_actions, upscale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="lora_r32")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    run_dir = os.path.join(OUT_ROOT, args.run)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=th.bfloat16, device_map=args.device)
    model = PeftModel.from_pretrained(model, os.path.join(run_dir, "adapter")).merge_and_unload().eval()
    norm = StateNorm(**json.load(open(os.path.join(run_dir, "state_norm.json"))))
    ds = DemoFrames(every=3)
    rng = np.random.default_rng(0)
    errs = []
    for i in rng.choice(len(ds), args.n, replace=False):
        img, st, chunk = ds[int(i)]
        pil = Image.fromarray(upscale(img))
        text = processor.apply_chat_template(build_messages(pil, prompt_for(norm.ints(st))), tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt").to(args.device)
        t0 = time.time()
        with th.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        ans = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        pred = text_to_actions(ans)
        n_ok = 0 if pred is None else len(pred)
        err = None if pred is None else float(np.abs(pred[:len(chunk)] - chunk[:len(pred)]).mean())
        errs.append(err)
        print(f"frame {int(i)}: {time.time() - t0:.0f}s | parsed {n_ok}/8 actions | mean abs error {err}")
        print(f"   predicted: {ans[:120]}")
        print(f"   demo:      {actions_to_text(chunk)[:120]}")
    good = [e for e in errs if e is not None]
    print(f"\nmean abs error over {len(good)} parsed answers: {np.mean(good):.3f} (actions are in [-1, 1]; "
          f"predicting all zeros + the right gripper would give roughly {np.mean([np.abs(ds[int(i)][2][:, :3]).mean() * 0.75 for i in rng.choice(len(ds), 50)]):.3f})")


if __name__ == "__main__":
    main()
