"""Tests for the action-wrench loss and the refinement objective."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("openpi")

from facet0.refinement import objective  # noqa: E402


def test_slices_are_normalized_by_their_own_masks():
    errors = jnp.zeros((1, 4, 13))
    errors = errors.at[..., :7].set(1.0).at[..., 7:].set(2.0)  # ||.||^2 = 7 per action row, 12 per wrench row
    m_a = jnp.array([[1, 1, 1, 0]], bool)
    m_w = jnp.array([[1, 0, 0, 0]], bool)
    total, l_a, l_w = objective.action_wrench_loss(errors, m_a, m_w, wrench_weight=0.1)
    np.testing.assert_allclose(l_a, [7.0])
    np.testing.assert_allclose(l_w, [12.0])  # one valid row is not down-weighted
    np.testing.assert_allclose(total, [7.0 + 1.2])


@dataclasses.dataclass
class _Obs:
    tokenized_prompt: jax.Array
    tokenized_prompt_mask: jax.Array


def test_all_frames_unconditional_and_positives_also_conditional(monkeypatch):
    # stub errors: 1 per element untagged, 3 per element tagged
    def fake_errors(model, rng, obs, target, train=True):
        tagged = obs.tokenized_prompt[:, 0].astype(jnp.float32)
        return jnp.ones(target.shape) * (1.0 + 2.0 * tagged)[:, None, None]

    monkeypatch.setattr(objective, "flow_matching_errors", fake_errors)
    b, h = 3, 4
    obs = _Obs(jnp.zeros((b, 2), jnp.int32), jnp.ones((b, 2), bool))
    info = {
        "label": jnp.array([True, False, True]),
        "contact_intensity": jnp.array([1.0, 0.5, 0.0]),
        "action_valid": jnp.ones((b, h), bool),
        "wrench_valid": jnp.ones((b, h), bool),
        "guided_prompt": jnp.ones((b, 2), jnp.int32),
        "guided_prompt_mask": jnp.ones((b, 2), bool),
    }
    obj = objective.RefinementObjective(contact_weight=1.0, wrench_weight=0.1)
    loss, m = objective.refinement_loss(None, jax.random.key(0), obs, jnp.zeros((b, h, 13)), info, obj)
    l_u = 7 * 1 + 0.1 * 6 * 1  # per-row squared norms, averaged over valid rows
    l_c = 7 * 3 + 0.1 * 6 * 3
    weights = np.array([2.0, 1.5, 1.0])
    expected = np.mean(weights * (l_u + np.array([1, 0, 1]) * l_c))
    np.testing.assert_allclose(float(loss), expected, rtol=1e-6)
    np.testing.assert_allclose(float(m["conditional_loss"]), l_c, rtol=1e-6)
    np.testing.assert_allclose(float(m["contact_weight_mean"]), weights.mean(), rtol=1e-6)
