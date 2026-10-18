"""Deployable Facet-0 policy and its wrench-history buffer."""

from __future__ import annotations

import functools
import logging
import pathlib
from collections import deque
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training import config as _config
from openpi_client import base_policy as _base_policy

from facet0 import conventions
from facet0.model import Facet0Config, Facet0Model
from facet0.policy import checkpoint, guidance
from facet0.policy.transforms import Facet0DataConfig
from facet0.contracts import read_manifest, model_metadata


def model_config(**kwargs) -> Facet0Config:
    return Facet0Config(**kwargs)


def data_config_factory(**kwargs) -> Facet0DataConfig:
    return Facet0DataConfig(
        base_config=_config.DataConfig(prompt_from_task=True, action_sequence_keys=["action"]), **kwargs
    )


class WrenchHistory:
    def __init__(self, window: int = conventions.WRENCH_WINDOW):
        self._buf: deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, wrench) -> None:
        self._buf.append(np.asarray(wrench, dtype=np.float32).reshape(conventions.WRENCH_DIM))

    def window(self) -> np.ndarray:
        if not self._buf:
            raise ValueError("push at least one wrench reading first")
        rows = list(self._buf)
        return np.stack([rows[0]] * (self._buf.maxlen - len(rows)) + rows)


class Facet0Policy(_base_policy.BasePolicy):
    def __init__(
        self,
        model: Facet0Model,
        norm_stats: dict[str, _transforms.NormStats],
        *,
        guidance_scale: float = 1.0,
        default_prompt: str | None = None,
        num_steps: int = 10,
        seed: int = 0,
        config: Facet0Config | None = None,
        tokenizer_path: str | None = None,
    ):
        data_config = data_config_factory(tokenizer_path=tokenizer_path).create(pathlib.Path("assets"), config or model_config())
        use_quantiles = data_config.use_quantile_norm
        self._inputs = _transforms.compose(
            [
                _transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=use_quantiles),
                *guidance.swap_tokenizer(list(data_config.model_transforms.inputs)),
            ]
        )
        self._outputs = _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles),
                *data_config.data_transforms.outputs,
            ]
        )
        graphdef, state = nnx.split(model)

        def sample(state, rng, observation, guided, guided_mask, noise, scale):
            return guidance.sample_with_guidance(
                nnx.merge(graphdef, state),
                rng,
                observation,
                guided,
                guided_mask,
                num_steps=num_steps,
                guidance_scale=scale,
                noise=noise,
            )[0]

        self._sample = functools.partial(jax.jit(sample), state)
        self._rng = jax.random.key(seed)
        self._guidance_scale = guidance_scale
        self.action_horizon = model.action_horizon
        self.action_dim = model.action_dim
        self._features = nnx.jit(model.extract_contact_features)

    def contact_features(self, obs):
        inputs = self._inputs(dict(obs))
        inputs.pop("guided_prompt")
        inputs.pop("guided_prompt_mask")
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        tokens, mask = self._features(_model.Observation.from_dict(batch))
        return np.asarray(tokens[0], np.float32), np.asarray(mask[0], bool)

    def infer(self, obs: dict, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        if "observation/wrench_history" not in obs:
            raise KeyError("observation/wrench_history [K, 6] is required; see facet0.policy.WrenchHistory")
        inputs = self._inputs(dict(obs))
        guided = jnp.asarray(inputs.pop("guided_prompt"))[None]
        guided_mask = jnp.asarray(inputs.pop("guided_prompt_mask"))[None]
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        self._rng, rng = jax.random.split(self._rng)
        if noise is None:
            noise = jax.random.normal(rng, (1, self.action_horizon, self.action_dim))
        noise = jnp.asarray(noise).reshape(1, self.action_horizon, self.action_dim)
        chunk = self._sample(rng, _model.Observation.from_dict(batch), guided, guided_mask, noise, self._guidance_scale)
        return self._outputs({"state": np.asarray(inputs["state"]), "actions": np.asarray(chunk[0], dtype=np.float32)})

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "wrench_window": conventions.WRENCH_WINDOW,
            "guidance_scale": self._guidance_scale,
        }


def load(
    model_dir: str | pathlib.Path, guidance_scale: float = 1.0, config: Facet0Config | None = None, **kwargs
) -> Facet0Policy:

    model_dir = pathlib.Path(model_dir)
    logging.info("Loading Facet-0 policy from %s", model_dir)
    manifest = read_manifest(model_dir)
    if config is not None and model_metadata(config) != manifest["model"]:
        raise ValueError("requested model differs from checkpoint")
    config = config or model_config(**manifest["model"])
    model = config.load(checkpoint.restore_params(model_dir / "params", dtype=jnp.bfloat16))
    return Facet0Policy(
        model, checkpoint.load_norm_stats(model_dir / "assets"), guidance_scale=guidance_scale, config=config,
        tokenizer_path=str(model_dir / "assets/policy_tokenizer.model") if manifest.get("tokenizer_sha256") else None,
        **kwargs
    )
