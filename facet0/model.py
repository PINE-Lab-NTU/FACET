"""Facet-0 policy: PaliGemma prefix and a flow-matching action expert with E_s and E_w tokens (Sec. 4.1)."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from typing_extensions import override

from facet0 import conventions


@dataclasses.dataclass(frozen=True)
class Facet0Config(_pi0_config.Pi0Config):
    pi05: bool = True
    discrete_state_input: bool = False
    action_dim: int = conventions.PROPOSAL_DIM
    action_horizon: int = conventions.ACTION_HORIZON
    wrench_window: int = conventions.WRENCH_WINDOW

    @property
    def state_dim(self) -> int:
        return conventions.KIN_DIM + self.wrench_window * conventions.WRENCH_DIM

    @override
    def create(self, rng) -> "Facet0Model":
        return Facet0Model(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        obs, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs = dataclasses.replace(obs, state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32))
        return obs, actions

    def vl_backbone_filter(self) -> nnx.filterlib.Filter:
        """Parameters of the vision-language backbone (SigLIP and the PaliGemma LLM expert)."""
        return nnx.Any(
            nnx_utils.PathRegex("PaliGemma/img/.*"),
            nnx.All(nnx_utils.PathRegex("PaliGemma/llm/.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*"))),
        )


class Facet0Model(_pi0.Pi0):
    def __init__(self, config: Facet0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        width = _gemma.get_config(config.action_expert_variant).width
        self.wrench_window = config.wrench_window
        self.state_encoder = nnx.Linear(conventions.KIN_DIM, width, rngs=rngs)  # E_s
        self.wrench_encoder_in = nnx.Linear(config.wrench_window * conventions.WRENCH_DIM, width, rngs=rngs)  # E_w
        self.wrench_encoder_out = nnx.Linear(width, width, rngs=rngs)

    def contact_tokens(self, state: jax.Array) -> jax.Array:
        """[E_s(s^kin); E_w(W)] as two expert tokens: [b, 2, width]."""
        kin = state[:, : conventions.KIN_DIM]
        window = state[:, conventions.KIN_DIM : conventions.KIN_DIM + self.wrench_window * conventions.WRENCH_DIM]
        s_token = self.state_encoder(kin)
        w_token = self.wrench_encoder_out(nnx.swish(self.wrench_encoder_in(window)))
        return jnp.stack([s_token, w_token], axis=1)

    @override
    def embed_suffix(self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: jax.Array):
        tokens, input_mask, ar_mask, adarms_cond = super().embed_suffix(obs, noisy_actions, timestep)
        contact = self.contact_tokens(obs.state).astype(tokens.dtype)
        tokens = jnp.concatenate([contact, tokens], axis=1)
        input_mask = jnp.concatenate([jnp.ones(contact.shape[:2], dtype=jnp.bool_), input_mask], axis=1)
        # [E_s, E_w] form one block after the prefix; the chunk forms the next block.
        ar_mask = jnp.concatenate([jnp.array([True, False]), ar_mask])
        return tokens, input_mask, ar_mask, adarms_cond


def flow_matching_errors(
    model: Facet0Model, rng: jax.Array, observation: _model.Observation, target: jax.Array, *, train: bool = True
) -> jax.Array:
    """Per-element squared velocity error ||v_theta(X^tau, tau | h^c) - U^tau||^2: [b, H, 13].

    X^tau = (1 - tau) Y + tau eps and U^tau = eps - Y, with tau ~ Beta(1.5, 1).
    """
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    noise = jax.random.normal(noise_rng, target.shape)
    tau = jax.random.beta(time_rng, 1.5, 1, target.shape[:-2]) * 0.999 + 0.001
    x_tau = tau[..., None, None] * noise + (1 - tau[..., None, None]) * target
    u_tau = noise - target
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, adarms_cond = model.embed_suffix(observation, x_tau, tau)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    attn = _pi0.make_attn_mask(input_mask, jnp.concatenate([prefix_ar, suffix_ar], axis=0))
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=attn, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    return jnp.square(v - u_tau)
