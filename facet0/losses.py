"""Masked action and wrench regression reductions."""

import jax.numpy as jnp


def masked_terms(errors, action_valid, wrench_valid):
    action = jnp.broadcast_to(action_valid[..., None], errors[..., :7].shape)
    wrench = jnp.broadcast_to(wrench_valid[..., None], errors[..., 7:13].shape)
    return (
        jnp.sum(jnp.where(action, errors[..., :7], 0.0)),
        jnp.sum(action),
        jnp.sum(jnp.where(wrench, errors[..., 7:13], 0.0)),
        jnp.sum(wrench),
    )


def reduced_terms(terms, wrench_weight=0.1):
    a_sum, a_count, w_sum, w_count = terms
    a, w = a_sum / jnp.maximum(a_count, 1), w_sum / jnp.maximum(w_count, 1)
    return a + wrench_weight * w, a, w


def per_sample_loss(errors, action_valid, wrench_valid, wrench_weight=0.1):
    import jax

    return jax.vmap(lambda e, a, w: reduced_terms(masked_terms(e, a, w), wrench_weight))(
        errors, action_valid, wrench_valid
    )
