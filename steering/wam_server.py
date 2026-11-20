"""Launch CyberWAM's BEHAVIOR-1K challenge policy server (challenge_docker/server.py) outside Docker, from the CyberWAM
venv, with optional steering hooks. No CyberWAM source is modified: this reproduces challenge_docker/Dockerfile's
environment + runtime_launcher.py's bundle validation in-process, then calls challenge_docker.server.main().

    # 1) preflight (no GPU, no model load): bundle contract/norm-stats hashes, base assets, text cache
    CUDA_VISIBLE_DEVICES= /home/behavior/ext/cyberwam/.venv/bin/python steering/wam_server.py \
        --model-dir /home/behavior/ext/ckpt/cyberwam/model_a --preflight-only

    # 2) serve (GPU; ~16-20 GB VRAM, see NOTES "CyberWAM setup")
    /home/behavior/ext/cyberwam/.venv/bin/python steering/wam_server.py \
        --model-dir /home/behavior/ext/ckpt/cyberwam/model_a --port 8000 [--episode-seed] [--prompt-file F]

    # 3) client (behavior env): python steering/run_rollouts.py --policy-profile cyberwam ...

Steering hooks (all off by default = exactly the submission server):
  --episode-seed      take the sampling seed from the client's `steering_seed` obs key at the first chunk of each
                      episode (CompatWrapper injects it); default server seed is fixed (20260719 + chunk index).
  --prompt-text T     replace the task prompt by format_b1k_prompt(T) (needs a cached embedding:
                      steering/wam_text_embed.py --text T).
  --prompt-file F     same, but re-read from file F at every new chunk (change prompts without reloading the 6B
                      model). Empty / missing file = official prompt. A line starting with "RAW:" is an exact prompt.
  --skip-pretrained-dit  build the video/action experts randomly instead of loading Wan2.2 / ActionDiT init; they
                      are fully overwritten by checkpoint.pt's strict `mot` load anyway (saves ~20 GB of reads
                      and ~10 GB of peak CPU RAM).
  --no-mmap-checkpoint  by default checkpoint.pt is torch.load'ed with mmap=True (same values, file-backed pages
                      instead of a full anonymous-RAM copy; this box has 31 GB RAM shared with other jobs).
  --stub              no model: serve zero normalized actions through the real ChallengePolicy/PolicySession/
                      protocol (observation validation, depth, K=32 chunking). For wiring tests; needs no GPU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

CYBERWAM = Path(os.environ.get("CYBERWAM_ROOT", "/home/behavior/ext/cyberwam"))
ASSETS = CYBERWAM / "checkpoints"
SEED_KEY = "steering_seed"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, help="Released bundle: checkpoint.pt, checkpoint_contract.json, ...")
    ap.add_argument("--assets", type=Path, default=ASSETS, help="runtime-assets root (models/, text_embeds/)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--task", default="turning_on_radio", help="Task whose text cache preflight checks")
    ap.add_argument("--inference-steps", type=int, default=20, help="FASTWAM_B1K_INFERENCE_STEPS (submission: 20)")
    ap.add_argument("--episode-seed", action="store_true")
    ap.add_argument("--prompt-text")
    ap.add_argument("--prompt-file", type=Path)
    ap.add_argument("--skip-pretrained-dit", action="store_true")
    ap.add_argument("--no-mmap-checkpoint", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--telemetry", type=Path, help="Per-request server timing JSONL")
    args, extra = ap.parse_known_args()  # extra args go to challenge_docker.server (e.g. --seed, --execute-horizon)
    if not args.stub and args.model_dir is None:
        ap.error("--model-dir is required (unless --stub)")
    return args, extra


def setup_env(args):
    """challenge_docker/Dockerfile ENV, with our paths."""
    models = args.assets / "models"
    env = {
        "PYTHONUNBUFFERED": "1",
        "FASTWAM_B1K_INFERENCE_STEPS": str(args.inference_steps),
        "FASTWAM_EXECUTE_HORIZON": "32",
        "FASTWAM_TEMPORAL_STRATEGY": "speed_2x",
        "FASTWAM_RESAMPLING": "zero_order_hold",
        "FASTWAM_RUNTIME_ASSETS": str(args.assets),
        "FASTWAM_B1K_MODEL_ID": str(models / "Wan-AI/Wan2.2-TI2V-5B"),
        "FASTWAM_B1K_DINO_PATH": str(models / "dinov3/vit_base_patch16_dinov3.lvd1689m/model.safetensors"),
        "DIFFSYNTH_MODEL_BASE_PATH": str(models),
        "FASTWAM_B1K_TEXT_CACHE_DIR": str(args.assets / "text_embeds"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        "FASTWAM_TASK_DATA": str(CYBERWAM / "challenge_docker/task_data.json"),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
    # No ActionDiT init on disk (training-only asset): the action expert starts random and checkpoint.pt's strict
    # `mot` load overwrites it. Optionally also skip the (equally overwritten) Wan2.2 DiT pretrained load.
    overrides = json.loads(os.environ.get("FASTWAM_B1K_CONFIG_OVERRIDES", "[]"))
    overrides.append("model.action_dit_pretrained_path=null")
    if args.skip_pretrained_dit:
        overrides.append("model.skip_dit_load_from_pretrain=true")
    os.environ["FASTWAM_B1K_CONFIG_OVERRIDES"] = json.dumps(overrides)
    for p in (CYBERWAM / "challenge_docker", CYBERWAM / "src", CYBERWAM):
        sys.path.insert(0, str(p))


def resolve_bundle(args):
    """runtime_launcher.py, in-process."""
    from runtime_contract import validate_bundle

    info = validate_bundle(args.model_dir.expanduser().resolve())
    os.environ["FASTWAM_MODEL_DIR"] = str(info["root"])
    os.environ["FASTWAM_CHECKPOINT"] = str(info["checkpoint"])
    os.environ["FASTWAM_CONTRACT"] = str(info["contract"])
    os.environ["FASTWAM_STATS"] = str(info["stats"])
    os.environ["FASTWAM_B1K_HYDRA_TASK"] = str(info["hydra_task"])
    if info.get("model_config_path") is None:
        os.environ.pop("FASTWAM_MODEL_CONFIG", None)
    else:
        os.environ["FASTWAM_MODEL_CONFIG"] = str(info["model_config_path"])
    if info["model_target"] == "fastwam.runtime.create_fastwam_joint":
        os.environ.pop("FASTWAM_B1K_DINO_PATH", None)
    return info


def preflight(args, info):
    from fastwam.datasets.b1k.text import format_b1k_prompt
    from wam_text_embed import cache_path

    models = args.assets / "models"
    need = {
        "VAE": models / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
        "DINOv3": Path(os.environ["FASTWAM_B1K_DINO_PATH"]) if "FASTWAM_B1K_DINO_PATH" in os.environ else None,
    }
    if not args.skip_pretrained_dit:
        need.update(
            {
                f"Wan2.2 DiT shard {i}": models
                / f"Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-0000{i}-of-00003.safetensors"
                for i in (1, 2, 3)
            }
        )
    prompts = [format_b1k_prompt(args.task)] + ([format_b1k_prompt(args.prompt_text)] if args.prompt_text else [])
    for p in prompts:
        need[f"text cache {p!r}"] = Path(cache_path(p, os.environ["FASTWAM_B1K_TEXT_CACHE_DIR"]))
    ok = True
    print(
        json.dumps(
            {k: str(v) for k, v in info.items() if k != "model_config"} | {"model_config": info["model_config"]},
            indent=1,
        )
    )
    if info["hydra_task"] != "b1k_2026_fastwam_flex_statedrop10_sharedt_ashift5_256gpu":
        print(f"NOTE: bundle maps to {info['hydra_task']}, not the released model_a config")
    allowed = (info["model_config"] or {}).get("allowed_task_ids")
    if allowed is not None and 0 not in allowed:
        print("WARN: model_config.json does not allow task 0 (turning_on_radio)")
    for name, path in need.items():
        exists = path is not None and path.is_file()
        ok &= exists
        print(f"{'OK  ' if exists else 'MISS'} {name}: {path}")
    print("PREFLIGHT", "OK" if ok else "FAILED")
    return ok


def install_hooks(args):
    import challenge_docker.server as server
    import experiments.b1k_2026.policy_adapter as adapter
    from fastwam.datasets.b1k.text import format_b1k_prompt

    if not args.no_mmap_checkpoint and not args.stub:
        import torch

        import fastwam.models.wan22.fastwam as fastwam_mod

        orig_torch_load = torch.load

        def mmap_load(f, *a, **kw):
            if str(f) == os.environ.get("FASTWAM_CHECKPOINT"):
                kw.setdefault("mmap", True)
            return orig_torch_load(f, *a, **kw)

        fastwam_mod.torch.load = mmap_load  # same torch module object: affects every torch.load caller

    if args.episode_seed:
        orig_act = server.ChallengePolicy.act

        def act(self, observation):
            if self.step == 0 and not self.queue and SEED_KEY in observation:
                import numpy as np

                s = int(np.asarray(observation[SEED_KEY]).reshape(-1)[0])
                self.seed = int(self.seed) + 1_000_003 * s  # fresh ChallengePolicy per episode (session.reset)
                print(f"[wam_server] episode seed {s} -> sampling seed {self.seed}", flush=True)
            return orig_act(self, observation)

        server.ChallengePolicy.act = act

    if args.prompt_text or args.prompt_file:
        orig_load = adapter.load_cached_text_context
        last = [None]

        def override_prompt():
            text = args.prompt_text
            if args.prompt_file is not None:
                try:
                    text = args.prompt_file.read_text().strip() or None
                except OSError:
                    text = None
            if not text:
                return None
            return text[4:].strip() if text.startswith("RAW:") else format_b1k_prompt(text)

        def load_cached_text_context(task):
            prompt = override_prompt()
            if prompt is None:
                out = orig_load(task)
                prompt = adapter.format_b1k_prompt(task)
            else:
                # orig_load formats its argument, so temporarily make format_b1k_prompt the identity.
                fmt = adapter.format_b1k_prompt
                adapter.format_b1k_prompt = lambda _t, _p=prompt: _p
                try:
                    out = orig_load(task)
                finally:
                    adapter.format_b1k_prompt = fmt
            if prompt != last[0]:
                print(f"[wam_server] prompt: {prompt!r} ({out[2].name})", flush=True)
                last[0] = prompt
            return out

        adapter.load_cached_text_context = load_cached_text_context


def serve_stub(args):
    """Real ChallengePolicy/PolicySession/protocol around a fake model (zero normalized actions)."""
    import functools

    import numpy as np
    import torch

    import challenge_docker.server as server
    from challenge_docker.protocol import WebsocketPolicyServer
    from fastwam.b1k_execution import B1KExecutionConfig

    class StubNormalizer:
        proprio_mean = torch.zeros(23)
        proprio_std = torch.ones(23)

        @staticmethod
        def denormalize_action(a):
            return a

    calls = []

    def policy_fn(model, *, observation, normalized_proprio, step, seed):
        calls.append(1)
        shapes = {k: tuple(np.asarray(v).shape) for k, v in observation["images"].items()}
        dshapes = {k: tuple(np.asarray(v).shape) for k, v in observation.get("depths", {}).items()}
        print(
            f"[stub] chunk {step} seed {seed} task {observation['task']!r} rgb {shapes} depth {dshapes} "
            f"proprio[:3] {np.round(observation['proprio'][:3], 3).tolist()}",
            flush=True,
        )
        a = torch.zeros(32, 23)
        a[:, 0] = 0.1 * (step % 2)  # alternate so the client can tell chunks apart
        return a

    factory = functools.partial(
        server.ChallengePolicy,
        model=None,
        policy_fn=policy_fn,
        normalizer=StubNormalizer(),
        task_lookup=server._task_lookup(Path(os.environ["FASTWAM_TASK_DATA"])),
        require_depth=True,
        seed=20260719,
        execution_config=B1KExecutionConfig(
            prediction_horizon=32, execute_horizon=32, temporal_strategy="speed_2x", resampling="zero_order_hold"
        ),
    )
    meta = {"protocol": "behavior_1k_websocket.v393", "stub": True, "requires_depth": True, "action_chunk_size": 32}
    WebsocketPolicyServer(factory, host=args.host, port=args.port, metadata=meta).serve_forever()


def main():
    args, extra = parse_args()
    setup_env(args)
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # wam_text_embed
    import logging

    logging.basicConfig(level=logging.INFO)
    if args.stub:
        install_hooks(args)
        return serve_stub(args)
    info = resolve_bundle(args)
    if args.preflight_only:
        sys.exit(0 if preflight(args, info) else 2)
    install_hooks(args)
    import challenge_docker.server as server

    argv = ["challenge_docker.server", "--host", args.host, "--port", str(args.port), *extra]
    if args.telemetry:
        argv += ["--telemetry", str(args.telemetry)]
    sys.argv = argv
    server.main()


if __name__ == "__main__":
    main()
