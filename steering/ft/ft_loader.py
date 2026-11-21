"""Training data loader for the radio fine-tune: RadioDataset -> openpi transforms -> batched (Observation, actions).

Replaces openpi-comet's create_behavior_data_loader (which needs BEHAVIOR v3.7's omnigibson.learning). Importing this
module installs steering/comet_server.py's small omnigibson shim first, because openpi.training.config imports
openpi-comet's b1k_policy, which imports omnigibson.learning.utils.eval_utils. That also has to happen inside the
spawned torch workers, which is why the dataset wrapper and worker init function live here.
"""

import logging
import multiprocessing
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [p for p in (_HERE, os.path.dirname(_HERE)) if p not in sys.path]

if "omnigibson.learning.utils.eval_utils" not in sys.modules:
    from comet_server import install_omnigibson_shim

    install_omnigibson_shim()

import jax  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import augment  # noqa: E402
import radio_data  # noqa: E402


class TransformedRadioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = list(transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x = self.dataset[i]
        for t in self.transforms:
            x = t(x)
        return x


def _collate(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init(worker_id):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    torch.set_num_threads(1)


class RadioLoader:
    def __init__(self, dataset, batch_size, sharding, num_workers, seed, num_batches=None):
        self._sharding = sharding
        self._num_batches = num_batches
        g = torch.Generator()
        g.manual_seed(seed)
        self._loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=_collate,
            worker_init_fn=_worker_init,
            drop_last=True,
            generator=g,
        )

    def __iter__(self):
        n = 0
        while True:
            for batch in self._loader:
                if self._num_batches is not None and n >= self._num_batches:
                    return
                n += 1
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield batch


def make_dataset(config, data_config, prompt_aug="none", image_aug="none", skip_norm_stats=False):
    from openpi import transforms as _transforms

    ds = radio_data.RadioDataset(
        root=config.data.root,
        action_horizon=config.model.action_horizon,
        exclude=tuple(config.data.exclude_episodes),
        prompt_fn=augment.PROMPT_AUGS[prompt_aug],
        image_fn=augment.IMAGE_AUGS[image_aug],
        seed=config.seed,
    )
    norm = {} if skip_norm_stats else data_config.norm_stats
    if norm is None:
        raise ValueError("norm stats missing; run steering/ft/train_radio.py --compute-norm-stats")
    return TransformedRadioDataset(
        ds,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def make_create_fn(prompt_aug="none", image_aug="none"):
    """Drop-in for openpi.training.data_loader.create_behavior_data_loader (same signature as used in train.py)."""

    def create(config, *, sharding=None, shuffle=False, num_batches=None, skip_norm_stats=False, seed_shift=0):
        from openpi.training import data_loader as _dl

        data_config = config.data.create(config.assets_dirs, config.model)
        ds = make_dataset(config, data_config, prompt_aug, image_aug, skip_norm_stats)
        logging.info(f"radio dataset: {len(ds.dataset.episodes)} episodes, {len(ds)} frames; "
                     f"prompt_aug={prompt_aug} image_aug={image_aug}")
        if sharding is None:
            sharding = jax.sharding.NamedSharding(jax.sharding.Mesh(jax.devices(), ("B",)), jax.sharding.PartitionSpec("B"))
        loader = RadioLoader(ds, config.batch_size, sharding, config.num_workers, config.seed + seed_shift, num_batches)
        return _dl.DataLoaderImpl(data_config, loader)

    return create
