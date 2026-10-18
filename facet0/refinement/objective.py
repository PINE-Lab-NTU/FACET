"""Compute policy training objectives."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import openpi.models.model as _model

from facet0.model import flow_matching_errors
from facet0.losses import per_sample_loss


@dataclasses.dataclass(frozen=True)
class RefinementObjective:
    contact_weight: float = 1.0
    wrench_weight: float = 0.1


def action_wrench_loss(errors: jax.Array, action_valid: jax.Array, wrench_valid: jax.Array, wrench_weight: float):

    return per_sample_loss(errors, action_valid, wrench_valid, wrench_weight)


def refinement_loss(
    model,
    rng: jax.Array,
    observation: _model.Observation,
    target: jax.Array,
    info: dict[str, jax.Array],
    objective: RefinementObjective,
) -> tuple[jax.Array, dict[str, jax.Array]]:

    uncond_rng, cond_rng = jax.random.split(rng)
    label = info["label"].astype(jnp.float32)
    weight = 1.0 + objective.contact_weight * info["contact_intensity"].astype(jnp.float32)
    tagged = dataclasses.replace(
        observation, tokenized_prompt=info["guided_prompt"], tokenized_prompt_mask=info["guided_prompt_mask"]
    )

    def branch(rng, obs):
        errors = flow_matching_errors(model, rng, obs, target, train=True)
        return action_wrench_loss(errors, info["action_valid"], info["wrench_valid"], objective.wrench_weight)

    l_uncond, a_uncond, w_uncond = branch(uncond_rng, observation)
    l_cond, _, _ = branch(cond_rng, tagged)
    loss = jnp.mean(weight * (l_uncond + label * l_cond))
    positives = jnp.maximum(jnp.sum(label), 1.0)
    metrics = {
        "loss": loss,
        "unconditional_loss": jnp.mean(l_uncond),
        "conditional_loss": jnp.sum(label * l_cond) / positives,
        "action_loss": jnp.mean(a_uncond),
        "wrench_loss": jnp.mean(w_uncond),
        "positive_fraction": jnp.mean(label),
        "contact_weight_mean": jnp.mean(weight),
    }
    return loss, metrics
