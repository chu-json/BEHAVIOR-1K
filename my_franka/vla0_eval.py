"""
Evaluate the VLA-0-style fine-tuned Qwen2.5-VL-3B (vla0_train.py) exactly like the diffusion policy (step11_eval_vision):
same 50 unseen apple layouts, same wrist camera + proprio, same success test. Every 8 env steps the model writes the
next 8 actions as integers; we parse and execute them (open loop for those 8 steps, like the DP's action chunks).

    OMNIGIBSON_HEADLESS=1 python my_franka/vla0_eval.py --run lora_r32 --episodes 50
"""
import argparse
import json
import os
import time

import numpy as np
import torch as th
from PIL import Image

import omnigibson as og
from pick_common import MAX_EPISODE_STEPS, MAX_STEP, apple_z, apply_layout, check_success, get_state, sample_layout, to_env_action
from step7_eval import EVAL_SEED
from step9_collect_vision import build_vision_cfg, proprio, wrist_rgb
from vla0_train import BASE_MODEL, HORIZON, OUT_ROOT, StateNorm, build_messages, prompt_for, text_to_actions, upscale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="lora_r32")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument("--start", type=int, default=0, help="first layout number")
    parser.add_argument("--out", default="eval.json")
    args = parser.parse_args()
    run_dir = os.path.join(OUT_ROOT, args.run)

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=th.bfloat16, device_map="cuda",
                                                               attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, os.path.join(run_dir, "adapter")).merge_and_unload().eval()
    norm = StateNorm(**json.load(open(os.path.join(run_dir, "state_norm.json"))))

    @th.inference_mode()
    def predict(img, state):
        pil = Image.fromarray(upscale(img))
        text = processor.apply_chat_template(build_messages(pil, prompt_for(norm.ints(state))), tokenize=False,
                                             add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    env = og.Environment(configs=build_vision_cfg(num_envs=1))
    rows, n_ok = [], 0
    for ep in range(args.start, args.start + args.episodes):
        layout = sample_layout(np.random.default_rng(args.seed + ep))
        apply_layout(env, layout)
        rest_z = apple_z(env)
        prev_grip, chunk, k, calls, bad, t_inf, first_close = 1.0, None, 0, 0, 0, 0.0, None
        close_dist, ever_grasped, max_eef_z_after_grasp = None, False, None  # diagnostics (privileged, logging only)
        for t in range(MAX_EPISODE_STEPS):
            s = get_state(env, prev_grip)
            if chunk is None or k >= len(chunk):
                t0 = time.time()
                answer = predict(wrist_rgb(env, 0), proprio(s))
                t_inf += time.time() - t0
                calls += 1
                chunk = text_to_actions(answer)
                if chunk is None:  # unparseable: hold still this step
                    bad += 1
                    chunk = np.array([[0.0, 0.0, 0.0, prev_grip]], dtype=np.float32)
                k = 0
            a = chunk[k]
            k += 1
            grip = 1.0 if a[3] >= 0 else -1.0
            if grip < 0 and first_close is None:
                first_close = t
                close_dist = round(float(th.norm(s["apple"] - s["eef"])) * 100, 1)
            if s["grasping"] > 0:
                ever_grasped = True
                z = float(s["eef"][2])
                max_eef_z_after_grasp = z if max_eef_z_after_grasp is None else max(max_eef_z_after_grasp, z)
            env.step(to_env_action(env, s, th.cat([th.tensor(a[:3]).clamp(-1, 1) * MAX_STEP, th.tensor([grip])])))
            prev_grip = grip
            if check_success(env, rest_z)[0] and s["eef"][2] > 0.17:
                break
        success, lifted = check_success(env, rest_z)
        n_ok += success
        rows.append(dict(ep=ep, success=bool(success), steps=t + 1, calls=calls, unparseable=bad, first_close=first_close,
                         close_dist_cm=close_dist, ever_grasped=ever_grasped,
                         max_eef_z_after_grasp=None if max_eef_z_after_grasp is None else round(max_eef_z_after_grasp, 3),
                         seconds_per_call=round(t_inf / max(calls, 1), 2)))
        print(f"eval ep {ep}: success={success} lifted={lifted * 100:.1f}cm steps={t + 1} first_close={first_close} "
              f"| close at {close_dist} cm, grasped {ever_grasped}, max eef z after grasp {max_eef_z_after_grasp} "
              f"| {calls} calls ({t_inf / max(calls, 1):.1f} s each, {bad} unparseable) | running {n_ok}/{len(rows)}", flush=True)
        json.dump(rows, open(os.path.join(run_dir, args.out), "w"), indent=1)
    print(f"FINAL VLA-0-style policy success rate: {n_ok}/{args.episodes} = {100 * n_ok / args.episodes:.1f}%", flush=True)
    og.shutdown()


if __name__ == "__main__":
    main()
