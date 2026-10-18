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
    cfg = Facet0Config(paligemma_variant="dummy", action_expert_variant="dummy", vision_variant="mu/32", max_token_len=48)
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
    np.testing.assert_allclose(h0, h1, atol=1e-6)
    assert h0.shape == (2, 2 * model.action_in_proj.out_features)
    assert not np.allclose(y0, y1)


def test_frozen_backbone_excludes_the_action_expert_and_contact_encoders(setup):
    cfg, model, _ = setup
    frozen = traverse_util.flatten_dict(nnx.state(model, cfg.vl_backbone_filter()).to_pure_dict(), sep="/")
    assert frozen and all(k.startswith("PaliGemma/") for k in frozen)
    assert not any("_1" in k.split("/")[-2] or k.endswith("_1") for k in frozen if "/llm/" in k)
    trainable = traverse_util.flatten_dict(nnx.state(model, nnx.Not(cfg.vl_backbone_filter())).to_pure_dict(), sep="/")
    for name in ("state_encoder", "wrench_encoder_in", "wrench_encoder_out", "action_in_proj", "action_out_proj"):
        assert any(k.startswith(name + "/") for k in trainable)
    assert any("_1" in k for k in trainable if k.startswith("PaliGemma/llm"))


def test_contact_feature_api_reads_state_and_history(setup):
    cfg, model, obs = setup
    tokens, mask = model.extract_contact_features(obs)
    assert tokens.shape == (2, 2, model.action_in_proj.out_features) and mask.all()
    changed = obs.replace(state=obs.state.at[:, :7].add(1.0))
    next_tokens, _ = model.extract_contact_features(changed)
    assert not np.allclose(tokens, next_tokens)
    changed = obs.replace(state=obs.state.at[:, 7:].add(1.0))
    next_tokens, _ = model.extract_contact_features(changed)
    assert not np.allclose(tokens, next_tokens)


def test_three_stage_checkpoint_and_adaptation(setup, tmp_path, monkeypatch):
    import optax
    import torch
    from openpi.models import tokenizer
    from openpi.shared import normalize
    from openpi.shared import nnx_utils
    from facet0.pretraining import core
    from facet0.pretraining.train import save
    from facet0.policy.checkpoint import restore_params
    from facet0.contracts import write_manifest, read_manifest, identity
    from facet0.refinement.objective import refinement_loss, RefinementObjective
    from facet_adaptation.integrations.serve import FrozenPolicyServer
    from facet_adaptation.token_model import RLTokenModel
    from facet_adaptation.config import LearnerConfig
    from facet_adaptation.learner import Learner

    class Tokenizer:
        def __init__(self, max_len):
            self.n = max_len

        def tokenize(self, prompt, state=None):
            tokens = np.zeros(self.n, np.int32)
            tokens[:3] = [1, 2, 3]
            return tokens, np.arange(self.n) < 3

    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", Tokenizer)
    cfg, model, obs = setup
    selection = nnx.All(nnx.Param, nnx_utils.PathRegex("action_out_proj/.*"))
    tx = optax.adam(1e-3)
    opt = tx.init(nnx.state(model, selection))
    target = jnp.ones((2, cfg.action_horizon, 13)) * 0.01
    valid = jnp.ones((2, cfg.action_horizon), bool)
    info = dict(action_valid=valid, wrench_valid=valid)
    loss, opt = core.update(
        model, tx, opt, lambda m: core.aw_loss(m, jax.random.key(12), obs, target, info, microbatch_size=1), selection
    )
    assert np.isfinite(float(loss))
    norm = {
        name: normalize.NormStats(mean=np.zeros(n), std=np.ones(n), q01=-np.ones(n), q99=np.ones(n))
        for name, n in [("state", 67), ("actions", 13)]
    }
    state = dict(step=1, aw_steps=1, vqa_steps=0, aw_opt=opt, vqa_opt=(), rng={})
    aligned = tmp_path / "aligned"
    save(aligned, model, cfg, norm, state, {"split": "synthetic"}, "")
    parent = read_manifest(aligned)
    restored = cfg.load(restore_params(aligned / "params"))
    info.update(
        label=jnp.array([True, False]),
        contact_intensity=jnp.array([1.0, 0.0]),
        guided_prompt=obs.tokenized_prompt,
        guided_prompt_mask=obs.tokenized_prompt_mask,
    )
    loss, opt = core.update(
        restored,
        tx,
        opt,
        lambda m: refinement_loss(m, jax.random.key(13), obs, target, info, RefinementObjective())[0],
        selection,
    )
    assert np.isfinite(float(loss))
    refined = tmp_path / "refined"
    save(refined, restored, cfg, norm, state, {}, identity(parent))
    write_manifest(refined, cfg, stage="post_training", parent=identity(parent))
    server = FrozenPolicyServer("facet0", refined, emit_prefix=True)
    image = np.zeros((224, 224, 3), np.uint8)
    state13 = np.r_[np.zeros(6), 0.5, np.zeros(6)].astype(np.float32)
    request = {
        "observation/state": state13,
        "observation/wrench_history": np.zeros((10, 6), np.float32),
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/right_wrist_image": image,
        "prompt": "insert the module",
    }
    result = server.infer(request)
    assert result["actions"].shape == (50, 7) and result["predicted_wrench"].shape == (50, 6)
    features, mask = result["prefix_embeddings"], result["prefix_mask"]
    assert features.shape == (2, 64)
    bottleneck = RLTokenModel(embedding_dim=64, encoder_layers=1, encoder_heads=2, decoder_layers=1, decoder_heads=2)
    optimizer = torch.optim.Adam(bottleneck.parameters(), lr=1e-3)
    reconstruction, _, _ = bottleneck(torch.tensor(features[None]), torch.tensor(mask[None]))
    reconstruction.backward()
    optimizer.step()
    bottleneck.eval()
    latent = bottleneck.encode(torch.tensor(features[None]), torch.tensor(mask[None]))[0].detach().numpy()
    learner = Learner(
        LearnerConfig(
            embedding_dim=64,
            projection_dim=8,
            hidden_dim=16,
            hidden_layers=1,
            batch_size=4,
            capacity=16,
            reference_dropout=0.0,
            updates_per_transition=1,
            critic_updates_per_actor=1,
            intervention_updates_per_episode=0,
        ),
        mode="rl",
    )
    x = learner.state_vector(latent, state13)
    reference = result["actions"][0]
    action = learner.act(x, reference)
    rows = [
        dict(
            x=x.copy(),
            next_x=x.copy(),
            reference=reference.copy(),
            next_reference=reference.copy(),
            action=action.copy(),
            human=False,
        )
        for _ in range(4)
    ]
    learner.finish_episode(rows, success=True)
    assert learner.updates == 4
    learner.save(tmp_path / "local.pt", {"policy": identity(read_manifest(refined))})
