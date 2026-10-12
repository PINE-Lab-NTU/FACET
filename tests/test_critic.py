"""Tests for the categorical projection and critic training."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("flax")

from facet0.refinement import critic  # noqa: E402

CFG = critic.CriticConfig(num_atoms=21, v_min=-1.0, v_max=1.0, gamma=0.5)


def test_projection_conserves_mass_and_shifts_support():
    probs = jax.nn.softmax(jax.random.normal(jax.random.key(0), (4, CFG.num_atoms)))
    reward = jnp.array([0.0, 0.3, -2.0, 1.0])
    done = jnp.array([0.0, 0.0, 0.0, 1.0])
    m = critic.project_distribution(probs, reward, done, CFG)
    np.testing.assert_allclose(m.sum(-1), 1.0, rtol=1e-6)
    mean_in = jnp.sum(probs * CFG.atoms, -1)
    mean_out = jnp.sum(m * CFG.atoms, -1)
    np.testing.assert_allclose(mean_out[0], 0.5 * mean_in[0], atol=1e-5)  # r + gamma z, no clipping
    np.testing.assert_allclose(mean_out[1], 0.3 + 0.5 * mean_in[1], atol=1e-5)
    np.testing.assert_allclose(m[2], jax.nn.one_hot(0, CFG.num_atoms), atol=1e-6)  # clipped at v_min
    np.testing.assert_allclose(mean_out[3], 1.0, atol=1e-6)  # terminal: the reward alone


def _synthetic(n_episodes=8, length=20, rep_dim=16, seed=0) -> critic.CriticData:
    rng = np.random.default_rng(seed)
    n = n_episodes * length
    ep = np.repeat(np.arange(n_episodes), length)
    fr = np.tile(np.arange(length), n_episodes)
    success = ep % 2 == 0
    progress = fr / (length - 1)
    rep = rng.normal(size=(n, rep_dim)).astype(np.float32)
    rep[:, 0] = progress
    rep[:, 1] = success
    reward = np.where((fr == length - 1) & success, 1.0, 0.0).astype(np.float32)
    done = (fr == length - 1).astype(np.float32)
    next_index = np.where(done > 0, np.arange(n), np.arange(n) + 1)
    return critic.CriticData(
        rep=rep, proposal=rng.normal(size=(n, 2, 5, 13)).astype(np.float32), next_index=next_index,
        reward=reward, done=done, success=success, progress=progress.astype(np.float32),
        contact=(fr > length // 2).astype(np.float32), regime=fr > length // 2,
        near_wrench=rng.normal(size=(n, 6)).astype(np.float32), episode=ep, frame=fr,
    )


def test_training_fits_values_and_roundtrips(tmp_path):
    data = _synthetic()
    cfg = dataclasses.replace(CFG, gamma=0.99, steps=300, batch_size=64, hidden=64, lr=1e-3)
    model, history = critic.train(data, cfg, log_every=100)
    assert history[-1]["td"] < history[0]["td"]
    v = critic.value(model, data, cfg)
    late_success = (data.frame > 15) & data.success
    late_failure = (data.frame > 15) & ~data.success
    assert v[late_success].mean() > v[late_failure].mean() + 0.2
    critic.save(model, cfg, tmp_path / "c", history)
    loaded, cfg2 = critic.load(tmp_path / "c")
    np.testing.assert_allclose(critic.value(loaded, data, cfg2), v, rtol=1e-5, atol=1e-6)
