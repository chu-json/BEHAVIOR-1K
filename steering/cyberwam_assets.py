"""Paced, resumable download of the CyberWAM (FastWAM Flex) base assets needed for INFERENCE.

Layout = the Docker image's runtime-assets root (DIFFSYNTH_MODEL_BASE_PATH = <dest>/models):

  models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors (+ index, config)  20.0 GB
  models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors                        1.4 GB
  models/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth                                          11.4 GB
  models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/{spiece.model,tokenizer.json,...}                        0.02 GB
  models/dinov3/vit_base_patch16_dinov3.lvd1689m/model.safetensors                                         0.34 GB

The DiffSynth converted VAE (fastwam redirect_common_files=true, its default) lives on ModelScope only, so it
comes from modelscope.cn (sha256-checked; slow, ~2 MB/s); the rest from Hugging Face. The T5 encoder comes as the
original .pth from HF (same weights/keys as the converted safetensors, which is ModelScope-only and slow). The ActionDiT init is training-only.
The T5 encoder is only needed to precompute text embeddings (steering/wam_text_embed.py), never by the server.

    python steering/cyberwam_assets.py --dest /home/behavior/ext/cyberwam/checkpoints [--skip-dit] [--dry-run]
"""

import argparse
import hashlib
import os
import time

import requests

HF = [  # (repo, file in repo, path under dest/models)
    ("timm/vit_base_patch16_dinov3.lvd1689m", "model.safetensors", "dinov3/vit_base_patch16_dinov3.lvd1689m"),
    ("timm/vit_base_patch16_dinov3.lvd1689m", "config.json", "dinov3/vit_base_patch16_dinov3.lvd1689m"),
    ("timm/vit_base_patch16_dinov3.lvd1689m", "LICENSE.md", "dinov3/vit_base_patch16_dinov3.lvd1689m"),
] + [
    ("Wan-AI/Wan2.1-T2V-1.3B", f"google/umt5-xxl/{f}", "Wan-AI/Wan2.1-T2V-1.3B")
    for f in ("special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json")
]
DIT = [
    ("Wan-AI/Wan2.2-TI2V-5B", f, "Wan-AI/Wan2.2-TI2V-5B")
    for f in (
        "config.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
    )
]
MS_REPO = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
MS = ["Wan2.2_VAE.safetensors"]
T5 = [("Wan-AI/Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth", "Wan-AI/Wan2.2-TI2V-5B")]
MS_API = f"https://www.modelscope.cn/api/v1/models/{MS_REPO}/repo/files?Revision=master&Recursive=true"
MS_URL = f"https://www.modelscope.cn/models/{MS_REPO}/resolve/master/{{}}"


def hf_download(repo, fname, local_dir):
    from huggingface_hub import hf_hub_download

    for attempt in range(40):
        try:
            return hf_hub_download(repo, fname, local_dir=local_dir)
        except Exception as e:
            wait = 30 if "429" in str(e) else 10
            print(f"  retry {repo}/{fname} in {wait}s ({attempt}): {str(e)[:120]}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {repo}/{fname}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ms_download(name, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out, part = os.path.join(out_dir, name), os.path.join(out_dir, name + ".part")
    if os.path.exists(out) and os.path.getsize(out) == meta["Size"]:
        print(f"  have {out}", flush=True)
        return
    for attempt in range(40):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        try:
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(MS_URL.format(name), headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 429:
                    raise RuntimeError("429")
                r.raise_for_status()
                if have and r.status_code != 206:  # server ignored the range: restart
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
            if os.path.getsize(part) == meta["Size"]:
                break
        except Exception as e:
            wait = 30 if "429" in str(e) else 10
            print(f"  retry {name} at {have} B in {wait}s ({attempt}): {str(e)[:120]}", flush=True)
            time.sleep(wait)
    digest = sha256(part)
    if digest != meta["Sha256"]:
        raise RuntimeError(f"sha256 mismatch for {name}: {digest} != {meta['Sha256']}")
    os.replace(part, out)
    print(f"  ok {out} (sha256 verified)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="/home/behavior/ext/cyberwam/checkpoints")
    ap.add_argument("--skip-dit", action="store_true", help="Skip the 20 GB Wan2.2 DiT (redundant given checkpoint.pt)")
    ap.add_argument("--skip-t5", action="store_true", help="Skip the 11.4 GB T5 encoder (text embeddings only)")
    ap.add_argument("--dry-run", action="store_true", help="Only list files and sizes")
    args = ap.parse_args()
    models = os.path.join(args.dest, "models")

    from huggingface_hub import HfApi

    api = HfApi()
    t5 = [] if args.skip_t5 else T5
    hf = HF + t5 + ([] if args.skip_dit else DIT)
    for attempt in range(30):
        try:
            ms_meta = {f["Path"]: f for f in requests.get(MS_API, timeout=60).json()["Data"]["Files"]}
            break
        except Exception as e:
            print(f"  ModelScope listing failed, retry in 30s: {str(e)[:120]}", flush=True)
            time.sleep(30)
    ms = MS
    total = 0
    for repo in sorted({r for r, _, _ in hf}):
        info = api.model_info(repo, files_metadata=True)
        print(f"{repo} @ {info.sha} gated={info.gated}", flush=True)
        assert not info.gated, f"{repo} is gated; not downloading"
        sizes = {s.rfilename: s.size for s in info.siblings}
        for r, f, _ in hf:
            if r == repo:
                print(f"  {sizes[f] / 1e9:7.3f} GB  {f}")
                total += sizes[f]
        time.sleep(0.35)
    print(f"ModelScope {MS_REPO}")
    for n in ms:
        print(f"  {ms_meta[n]['Size'] / 1e9:7.3f} GB  {n}")
        total += ms_meta[n]["Size"]
    print(f"TOTAL {total / 1e9:.2f} GB -> {models}", flush=True)
    if args.dry_run:
        return
    # small files first, then T5 (needed first, for text embeddings), the DiT shards, and the slow ModelScope VAE
    for repo, fname, sub in HF + t5:
        print(f"HF {repo}/{fname}", flush=True)
        hf_download(repo, fname, os.path.join(models, sub))
        time.sleep(0.35)
    for repo, fname, sub in [] if args.skip_dit else DIT:
        print(f"HF {repo}/{fname}", flush=True)
        hf_download(repo, fname, os.path.join(models, sub))
        time.sleep(0.35)
    for n in ms:
        print(f"MS {n}", flush=True)
        ms_download(n, ms_meta[n], os.path.join(models, MS_REPO))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
