"""Alternating optimizer updates and VQA supervision."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from facet0.model import flow_matching_errors
from facet0.losses import masked_terms
from openpi.models.pi0 import make_attn_mask
from openpi.shared import nnx_utils


def vqa_filter():
    return nnx.All(nnx.Param, nnx_utils.PathRegex("PaliGemma/llm/.*"), nnx.Not(nnx_utils.PathRegex(".*_1.*")))


def vqa_loss(model, batch):
    image, _ = model.PaliGemma.img(batch["image"], train=False)
    text = model.PaliGemma.llm(batch["tokens"], method="embed")
    tokens = jnp.concatenate([image, text], axis=1)
    im = jnp.ones(image.shape[:2], dtype=bool)
    mask = jnp.concatenate([im, batch["token_mask"]], axis=1)
    ar = jnp.concatenate([jnp.zeros_like(im), batch["ar_mask"]], axis=1)
    (hidden, _), _ = model.PaliGemma.llm(
        [tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1
    )
    hidden = hidden[:, image.shape[1] :]
    predictors = jnp.take_along_axis(hidden, jnp.maximum(batch["target_indices"] - 1, 0)[..., None], axis=1)
    logits = model.PaliGemma.llm(predictors, method=lambda m, x: m.embedder.decode(x))
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target_tokens"])
    valid = batch["target_mask"]
    return jnp.sum(jnp.where(valid, loss, 0.0)) / jnp.maximum(jnp.sum(valid), 1)


def aw_loss(model, rng, observation, targets, info, *, microbatch_size, wrench_weight=0.1, train=True):
    batch = targets.shape[0]
    if batch % microbatch_size:
        raise ValueError("effective batch must be divisible by microbatch size")
    a_count = jnp.maximum(jnp.sum(info["action_valid"]) * 7, 1)
    w_count = jnp.maximum(jnp.sum(info["wrench_valid"]) * 6, 1)

    def step(total, index):
        start = index * microbatch_size
        obs = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, start, microbatch_size), observation)
        target = jax.lax.dynamic_slice_in_dim(targets, start, microbatch_size)
        a = jax.lax.dynamic_slice_in_dim(info["action_valid"], start, microbatch_size)
        w = jax.lax.dynamic_slice_in_dim(info["wrench_valid"], start, microbatch_size)
        errors = flow_matching_errors(model, jax.random.fold_in(rng, index), obs, target, train=train)
        a_sum, _, w_sum, _ = masked_terms(errors, a, w)
        return total + a_sum / a_count + wrench_weight * w_sum / w_count, None

    value, _ = jax.lax.scan(jax.checkpoint(step), jnp.array(0.0), jnp.arange(batch // microbatch_size))
    return value


def update(model, optimizer, opt_state, loss_fn, parameter_filter):
    loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, parameter_filter))(model)
    if not bool(jnp.isfinite(loss)) or not bool(jnp.isfinite(optax.global_norm(grads))):
        raise FloatingPointError("nonfinite optimization step")
    selected = nnx.state(model, parameter_filter)
    updates, opt_state = optimizer.update(grads, opt_state, selected)
    nnx.update(model, optax.apply_updates(selected, updates))
    return loss, opt_state
