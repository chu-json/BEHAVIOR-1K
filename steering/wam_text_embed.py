"""Precompute CyberWAM (FastWAM) T5 prompt embeddings for the policy server's text cache, on CPU.

The server never runs T5: experiments/b1k_2026/policy_adapter.py looks up
    <cache>/<sha256(prompt)>.t5_len128.wan22ti2v5b.pt     ({"context": bf16 [128, 4096], "mask": bool [128]})
where prompt = format_b1k_prompt(task) = "A video recorded from a robot's point of view executing the following
instruction: " + task with "_" -> " " (training used the task NAME, e.g. "turning on radio", not the long
instruction in tasks.jsonl). This writes exactly that payload, using CyberWAM's own tokenizer wrapper and T5 module.

Why not scripts/precompute_text_embeds.py: it builds the 5.7B-param encoder in fp32 first (~23 GB RAM) and needs a
dataset tasks.jsonl; this box has ~10 GB free. Here the model is built on the meta device and the bf16 .pth is
mmap'ed and assigned (file-backed pages, ~0.5 GB RSS + page cache), forward in bf16 like the official script on CPU.

Run with the CyberWAM venv (CPU only):
    CUDA_VISIBLE_DEVICES= /home/behavior/ext/cyberwam/.venv/bin/python steering/wam_text_embed.py \
        --task turning_on_radio                                   # official task-name prompt (what the server uses)
        --text "turn on the radio on the coffee table"            # free text, wrapped in the B1K template
        --raw "A video recorded ... exact prompt"                 # exact prompt string, no template
        [--file prompts.txt]                                      # one free-text instruction per line
    -> prints sha256 + path per prompt and appends to <cache>/prompts.jsonl.
Use the free text with steering/wam_server.py --prompt-text / --prompt-file.
"""

import argparse
import hashlib
import json
import os
import sys
import time

CYBERWAM = os.environ.get("CYBERWAM_ROOT", "/home/behavior/ext/cyberwam")
ASSETS = os.path.join(CYBERWAM, "checkpoints")
T5_PATH = os.path.join(ASSETS, "models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth")
TOKENIZER = os.path.join(ASSETS, "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl")
CACHE = os.path.join(ASSETS, "text_embeds")
CONTEXT_LEN = 128
ENC_ID = "wan22ti2v5b"  # scripts/precompute_text_embeds._model_id_to_enc_id("Wan-AI/Wan2.2-TI2V-5B")


def cache_path(prompt, cache_dir=CACHE):
    return os.path.join(
        cache_dir, f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}.t5_len{CONTEXT_LEN}.{ENC_ID}.pt"
    )


def build_prompts(args):
    sys.path[:0] = [CYBERWAM, os.path.join(CYBERWAM, "src")]
    from fastwam.datasets.b1k.text import format_b1k_prompt

    prompts = [format_b1k_prompt(t) for t in args.task] + [format_b1k_prompt(t) for t in args.text] + list(args.raw)
    if args.file:
        with open(args.file) as f:
            prompts += [format_b1k_prompt(line.strip()) for line in f if line.strip()]
    return list(dict.fromkeys(prompts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="*", default=[], help="Task names (format_b1k_prompt), e.g. turning_on_radio")
    ap.add_argument("--text", nargs="*", default=[], help="Free-text instructions, wrapped in the B1K template")
    ap.add_argument("--raw", nargs="*", default=[], help="Exact prompt strings")
    ap.add_argument("--file", help="File with one free-text instruction per line")
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--t5", default=T5_PATH)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (be polite to the sim / fine-tuning jobs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
    prompts = build_prompts(args)
    if not prompts:
        ap.error("give at least one --task / --text / --raw / --file")
    todo = [p for p in prompts if args.overwrite or not os.path.exists(cache_path(p, args.cache_dir))]
    for p in prompts:
        if p not in todo:
            print(f"cached  {cache_path(p, args.cache_dir)}  {p!r}")
    if not todo:
        return

    import torch

    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder

    torch.set_num_threads(args.threads)
    t0 = time.time()
    with torch.device("meta"):
        model = WanTextEncoder()
    state = torch.load(args.t5, map_location="cpu", mmap=True, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    assert all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in model.parameters())
    tokenizer = HuggingfaceTokenizer(name=args.tokenizer, seq_len=CONTEXT_LEN, clean="whitespace")
    print(f"T5 ready in {time.time() - t0:.1f}s; encoding {len(todo)} prompt(s) on CPU", flush=True)

    os.makedirs(args.cache_dir, exist_ok=True)
    with torch.no_grad(), open(os.path.join(args.cache_dir, "prompts.jsonl"), "a") as log:
        for prompt in todo:
            t = time.time()
            ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
            mask = mask.to(dtype=torch.bool)
            context = model(ids, mask)
            payload = {
                "context": context[0].detach().to(dtype=torch.bfloat16).contiguous(),
                "mask": mask[0].contiguous(),
            }
            out = cache_path(prompt, args.cache_dir)
            tmp = out + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, out)
            n = int(mask.sum())
            print(
                f"wrote   {out}  tokens={n}{' (TRUNCATED)' if n == CONTEXT_LEN else ''}  {time.time() - t:.1f}s  {prompt!r}"
            )
            log.write(json.dumps({"prompt": prompt, "path": out, "tokens": n, "time": time.time()}) + "\n")


if __name__ == "__main__":
    main()
