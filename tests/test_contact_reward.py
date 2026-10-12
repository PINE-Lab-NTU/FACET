"""Tests for contact intensity, contact regimes and the interaction reward."""

import numpy as np
import pandas as pd

from facet0 import contact
from facet0.data import lerobot
from facet0.refinement import reward

from conftest import write_corpus


def test_causal_filter_never_looks_ahead():
    x = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32)
    y = x.copy()
    y[20:] += 100.0
    np.testing.assert_array_equal(contact.causal_filter(x, 5)[:20], contact.causal_filter(y, 5)[:20])
    np.testing.assert_array_equal(contact.causal_filter(x, 1), x)


def test_regime_thresholds_intensity(corpus):
    states = lerobot.load_episode_states(corpus)
    for threshold in (1.0, 0.5):
        spec = contact.ContactSpec(regime_threshold=threshold)
        for ep in contact.episode_contact(states, spec).values():
            assert 0.0 <= ep.intensity.min() and ep.intensity.max() <= 1.0
            np.testing.assert_array_equal(ep.regime, ep.intensity >= threshold)


def test_corpus_floor_raises_scale_of_gentle_episode():
    loud = np.zeros((50, 13), np.float32)
    loud[:, 9] = np.linspace(0, 50, 50)
    gentle = np.zeros((50, 13), np.float32)
    gentle[:, 9] = np.random.default_rng(1).normal(0, 0.01, 50)
    floored = contact.episode_contact({0: loud, 1: gentle, 2: loud.copy()}, contact.ContactSpec())
    unfloored = contact.episode_contact({1: gentle}, contact.ContactSpec())
    assert floored[1].scale > unfloored[1].scale
    assert floored[1].regime.sum() < unfloored[1].regime.sum()


def test_outcome_only_reward_has_no_failure_penalty(corpus):
    table = reward.compute_returns(corpus, reward.RewardConfig(gamma=0.9))
    success = lerobot.episode_success(corpus)
    for ep, g in table.groupby("episode_index"):
        r = g["reward"].to_numpy()
        assert r[-1] == (1.0 if success[ep] else 0.0) and np.all(r[:-1] == 0.0)
        np.testing.assert_allclose(g["return"].to_numpy(), r[-1] * 0.9 ** np.arange(len(r))[::-1], rtol=1e-5)


def test_speed_bonus_and_force_penalty():
    complete = np.array([0, 0, 1, 0, 1], bool)
    elapsed = np.array([1, 2, 3, 1, 2], float)
    wrench = np.zeros((5, 6))
    wrench[3, 2] = 30.0  # 1.5 x the 20 N envelope on f_z
    cfg = reward.RewardConfig(alpha=0.5, beta=1.0, time_budget=4.0)
    r = reward.interaction_reward(complete, elapsed, wrench, np.full((5, 6), 20.0), cfg)
    np.testing.assert_allclose(r, [0, 0, 1 + 0.25, -0.25, 1 + 0.5])


def test_judge_segmentation_resets_subgoal_time(tmp_path):
    states = np.zeros((6, 13), np.float32)
    root = write_corpus(tmp_path / "c", [{"states": states, "success": True}])
    f = next((root / "data").rglob("*.parquet"))
    df = pd.read_parquet(f)
    df["subgoal_index"] = [0, 0, 0, 1, 1, 1]
    df["subgoal_complete"] = [0, 0, 1, 0, 0, 1]
    df["phase"] = ["align"] * 3 + ["press"] * 3
    df.to_parquet(f, index=False)
    r = reward.compute_returns(root, reward.RewardConfig(beta=1.0, time_budget=6.0))["reward"].to_numpy()
    np.testing.assert_allclose(r, [0, 0, 1 + 0.5, 0, 0, 1 + 0.5])
