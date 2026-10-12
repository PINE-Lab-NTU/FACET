"""Tests for the Facet-0 policy model at dummy size."""

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("openpi")

from facet0.model import Facet0Config, flow_matching_errors  # noqa: E402
from facet0.policy import guidance  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    cfg = Facet0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(2)
    obs = obs.replace(state=jax.random.normal(jax.random.key(1), obs.state.shape))
    return cfg, model, obs


def test_layout_and_errors_match_openpi_loss(setup):
    cfg, model, obs = setup
    assert obs.state.shape == (2, 7 + 10 * 6) and cfg.action_dim == 13
    target = jax.random.normal(jax.random.key(2), (2, cfg.action_horizon, 13))
    errors = flow_matching_errors(model, jax.random.key(3), obs, target, train=False)
    assert errors.shape == (2, cfg.action_horizon, 13)
    np.testing.assert_allclose(errors.mean(-1), model.compute_loss(jax.random.key(3), obs, target), atol=1e-6)


def test_guidance_zero_is_unconditional_and_hc_ignores_the_chunk(setup):
    cfg, model, obs = setup
    noise = jax.random.normal(jax.random.key(4), (2, cfg.action_horizon, 13))
    tagged = jnp.ones_like(obs.tokenized_prompt)
    y0, h0 = guidance.sample_with_guidance(
        model, None, obs, tagged, obs.tokenized_prompt_mask, num_steps=2, guidance_scale=0.0, noise=noise
    )
    y1, h1 = guidance.sample_with_guidance(
        model, None, obs, tagged, obs.tokenized_prompt_mask, num_steps=2, guidance_scale=1.0, noise=noise * 2
    )
    np.testing.assert_allclose(y0, model.sample_actions(None, obs, num_steps=2, noise=noise), atol=1e-5)
    np.testing.assert_allclose(h0, h1, atol=1e-6)  # h^c does not depend on the chunk or the tag
    assert h0.shape == (2, 2 * model.action_in_proj.out_features)
    assert not np.allclose(y0, y1)


def test_frozen_backbone_excludes_the_action_expert_and_contact_encoders(setup):
    cfg, model, _ = setup
    frozen = traverse_util.flatten_dict(nnx.state(model, cfg.vl_backbone_filter()).to_pure_dict(), sep="/")
    assert frozen and all(k.startswith("PaliGemma/") for k in frozen)
    assert not any("_1" in k.split("/")[-2] or k.endswith("_1") for k in frozen if "/llm/" in k)
    trainable = traverse_util.flatten_dict(
        nnx.state(model, nnx.Not(cfg.vl_backbone_filter())).to_pure_dict(), sep="/"
    )
    for name in ("state_encoder", "wrench_encoder_in", "wrench_encoder_out", "action_in_proj", "action_out_proj"):
        assert any(k.startswith(name + "/") for k in trainable)
    assert any("_1" in k for k in trainable if k.startswith("PaliGemma/llm"))
