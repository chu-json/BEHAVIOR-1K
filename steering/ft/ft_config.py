"""openpi TrainConfigs for fine-tuning base pi0.5 on the radio demos, built in code (no openpi-comet edits).

Config names: radio_<variant>, variant in
    lora      LoRA on PaliGemma (gemma_2b_lora) + action expert (gemma_300m_lora); SigLIP, projections and the
              time/action MLPs train fully (openpi's standard LoRA recipe, Pi0Config.get_freeze_filter()).
    lora_fv   as lora, but SigLIP (PaliGemma/img) frozen too ("frozen vision").
    expert    PaliGemma (LLM + SigLIP) frozen, action expert + projections trained fully.
    full      everything trained (does not fit a 24 GB GPU; for reference).
Used by steering/ft/train_radio.py (training) and steering/comet_server.py --config steering_ft:<name> (serving).

Representation (same at train and serve time):
  state   23-d in action order (radio_data.RadioInputs; accepts the fork's 61-d proprio at serve time)
  actions 23-d; torso + both arms as DELTAS from the current state (openpi DeltaActions/AbsoluteActions with
          radio_data.DELTA_MASK, as in base pi0.5 pre-training); base velocity and grippers absolute
  norm    quantile norm stats from radio_data.compute_norm_stats (state + delta actions), stored as
          <assets_dir>/radio_ft/norm_stats.json and copied into every checkpoint's assets/radio_ft/
  images  head -> base_0_rgb, left wrist, right wrist; 224x224 resize_with_pad (preprocess.py / Comet wrapper)
  prompt  Comet's radio prompt by default (radio_data.RADIO_PROMPT)
"""

import dataclasses
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flax.nnx as nnx  # noqa: E402
from openpi import transforms as _transforms  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from openpi.training import config as _config  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402
from openpi.training import weight_loaders  # noqa: E402

import radio_data  # noqa: E402

PI05_BASE_PARAMS = "/home/behavior/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
ASSETS_DIR = "/home/behavior/ext/data/radio_ft/assets"  # holds radio_ft/norm_stats.json
ASSET_ID = "radio_ft"
CKPT_BASE = "/home/behavior/ext/ckpt/ft"
VARIANTS = ("lora", "lora_fv", "expert", "full")
# 5 of the 200 radio demos held out for offline validation (the 5 highest episode indices)
HOLDOUT = (2880, 2960, 2970, 2990, 3000)


@dataclasses.dataclass(frozen=True)
class RadioDataConfig(_config.DataConfigFactory):
    repo_id: str = "steering/radio_ft"
    assets: _config.AssetsConfig = dataclasses.field(
        default_factory=lambda: _config.AssetsConfig(assets_dir=ASSETS_DIR, asset_id=ASSET_ID)
    )
    root: str = radio_data.DATA
    # held-out demo episodes (not trained on; steering/ft/eval_offline.py scores them)
    exclude_episodes: tuple[int, ...] = HOLDOUT

    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        mask = _transforms.make_bool_mask(*radio_data.DELTA_MASK)
        data_transforms = _transforms.Group(inputs=[radio_data.RadioInputs()], outputs=[radio_data.RadioOutputs()])
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(mask)], outputs=[_transforms.AbsoluteActions(mask)]
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
            use_quantile_norm=True,
        )


def model_and_freeze(variant, action_horizon=32):
    if variant in ("lora", "lora_fv"):
        model = pi0_config.Pi0Config(
            pi05=True,
            action_horizon=action_horizon,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        freeze = model.get_freeze_filter()
        if variant == "lora_fv":
            freeze = nnx.Any(freeze, nnx_utils.PathRegex(".*img.*"))
    elif variant == "expert":
        model = pi0_config.Pi0Config(pi05=True, action_horizon=action_horizon)
        freeze = nnx.Any(
            nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
            nnx_utils.PathRegex(".*img.*"),
        )
    elif variant == "full":
        model = pi0_config.Pi0Config(pi05=True, action_horizon=action_horizon)
        freeze = nnx.Nothing
    else:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    return model, freeze


def make_config(
    variant="lora",
    exp_name="debug",
    batch_size=16,
    num_train_steps=20_000,
    peak_lr=2.5e-5,
    warmup_steps=1_000,
    save_interval=2_000,
    keep_period=6_000,
    log_interval=50,
    num_workers=4,
    action_horizon=32,
    seed=42,
    resume=False,
    overwrite=False,
    weight_path=PI05_BASE_PARAMS,
):
    model, freeze = model_and_freeze(variant, action_horizon)
    warmup_steps = min(warmup_steps, max(1, num_train_steps // 10))  # short (probe) runs
    return _config.TrainConfig(
        name=f"radio_{variant}",
        exp_name=exp_name,
        project_name="steering_ft",
        model=model,
        data=RadioDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(weight_path),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps, peak_lr=peak_lr, decay_steps=num_train_steps, decay_lr=peak_lr / 10
        ),
        freeze_filter=freeze,
        ema_decay=None,
        assets_base_dir=ASSETS_DIR,
        checkpoint_base_dir=CKPT_BASE,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        log_interval=log_interval,
        save_interval=save_interval,
        keep_period=keep_period,
        seed=seed,
        resume=resume,
        overwrite=overwrite,
        wandb_enabled=False,
    )


def get_config(name):
    """Serving: `radio_<variant>` -> TrainConfig with the same model/transforms (training-only fields irrelevant)."""
    assert name.startswith("radio_"), name
    return make_config(variant=name[len("radio_") :])
