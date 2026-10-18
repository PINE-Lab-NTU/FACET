"""Refinement data pipeline: corpus frames + credit labels -> (observation, Y^data, info)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

from facet0.data import frames as _frames
from facet0.policy import guidance

INFO_KEYS = ("label", "contact_intensity", "action_valid", "wrench_valid", "guided_prompt", "guided_prompt_mask")


def build_dataset(config: _config.TrainConfig, credit_path: Path, norm_stats: dict) -> tuple[_config.DataConfig, Any]:
    data_config = dataclasses.replace(config.data.create(config.assets_dirs, config.model), norm_stats=norm_stats)
    frames = _frames.CorpusFrames(
        Path(data_config.repo_id), config.model.action_horizon, config.model.wrench_window, credit=credit_path
    )
    dataset = _data_loader.TransformedDataset(
        frames,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *guidance.swap_tokenizer(list(data_config.model_transforms.inputs)),
        ],
    )
    return data_config, dataset


class RefinementDataLoader:
    def __init__(self, data_config: _config.DataConfig, torch_loader: Any):
        self._data_config = data_config
        self._loader = torch_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._loader:
            yield _model.Observation.from_dict(batch), batch["actions"], {k: batch[k] for k in INFO_KEYS}


def create_data_loader(
    config: _config.TrainConfig,
    credit_path: Path,
    norm_stats: dict,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
    corpora: list[dict] | None = None,
) -> RefinementDataLoader:
    data_config, dataset = build_dataset(config, credit_path, norm_stats)
    if corpora:
        import torch

        datasets = []
        for corpus in corpora:
            root = Path(corpus["dataset"])
            current = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=str(root)))
            _, part = build_dataset(current, root / "meta" / f"credit_{corpus['credit_tag']}.parquet", norm_stats)
            datasets.append(part)
        dataset = torch.utils.data.ConcatDataset(datasets)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    return RefinementDataLoader(data_config, loader)
