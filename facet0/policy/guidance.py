"""Instruction-tag conditioning on b_t and classifier-free-guidance sampling (App. D.2)."""

from __future__ import annotations

import dataclasses
from typing import Any

import einops
import jax
import jax.numpy as jnp
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.models import pi0 as _pi0

from facet0 import conventions


@dataclasses.dataclass(frozen=True)
class TokenizeWithGuidance(_transforms.DataTransformFn):
    """Drop-in replacement for ``TokenizePrompt`` that also emits the tagged prompt."""

    tokenizer: Any

    def __call__(self, data: dict) -> dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        tokens, mask = self.tokenizer.tokenize(prompt, None)
        guided_tokens, guided_mask = self.tokenizer.tokenize(f"{prompt}{conventions.GUIDANCE_TAG}", None)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": mask,
            "guided_prompt": guided_tokens,
            "guided_prompt_mask": guided_mask,
        }


def swap_tokenizer(model_inputs: list[_transforms.DataTransformFn]) -> list[_transforms.DataTransformFn]:
    """Replace the ``TokenizePrompt`` step of an openpi model-transform stack."""
    out, found = [], False
    for t in model_inputs:
        if isinstance(t, _transforms.TokenizePrompt):
            if t.discrete_state_input:
                raise ValueError("Facet-0 feeds the state to the action expert, not to the prompt")
            out.append(TokenizeWithGuidance(t.tokenizer))
            found = True
        else:
            out.append(t)
    if not found:
        raise ValueError("expected a TokenizePrompt transform in the model input stack")
    return out


def sample_with_guidance(
    model,
    rng: jax.Array,
    observation: _model.Observation,
    guided_prompt: jax.Array,
    guided_prompt_mask: jax.Array,
    *,
    num_steps: int = 10,
    guidance_scale: float = 1.0,
    noise: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Integrate the guided velocity field from noise (tau = 1) to a proposal (tau = 0).

    ``observation.tokenized_prompt`` is the unconditional branch and ``guided_prompt`` the
    tagged branch. Returns the normalized proposal Y [b, H, 13] and the semantic-contact
    representation h^c [b, 2 * width]: the expert's outputs at the E_s and E_w tokens under
    the untagged instruction at tau = 1. They cannot attend to the chunk, so h^c does not
    depend on the noise or on the sampled proposal.
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    guided = dataclasses.replace(observation, tokenized_prompt=guided_prompt, tokenized_prompt_mask=guided_prompt_mask)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    def prefill(obs):
        tokens, mask, ar_mask = model.embed_prefix(obs)
        positions = jnp.cumsum(mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([tokens, None], mask=_pi0.make_attn_mask(mask, ar_mask), positions=positions)
        return mask, kv_cache

    uncond_prefix = prefill(observation)
    guided_prefix = prefill(guided)

    def expert(obs, prefix, x_t, time):
        prefix_mask, kv_cache = prefix
        tokens, mask, ar_mask, adarms_cond = model.embed_suffix(obs, x_t, jnp.broadcast_to(time, batch_size))
        attn = jnp.concatenate(
            [einops.repeat(prefix_mask, "b p -> b s p", s=tokens.shape[1]), _pi0.make_attn_mask(mask, ar_mask)], axis=-1
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        (_, out), _ = model.PaliGemma.llm(
            [None, tokens], mask=attn, positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
        )
        return out

    def velocity(obs, prefix, x_t, time):
        return model.action_out_proj(expert(obs, prefix, x_t, time)[:, -model.action_horizon :])

    def step(carry):
        x_t, time = carry
        v_uncond = velocity(observation, uncond_prefix, x_t, time)
        v_cond = velocity(guided, guided_prefix, x_t, time)
        return x_t + dt * (v_uncond + guidance_scale * (v_cond - v_uncond)), time + dt

    first = expert(observation, uncond_prefix, noise, 1.0)
    h_c = first[:, :2].reshape(batch_size, -1)
    x_0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
    return x_0, h_c
