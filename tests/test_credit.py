"""Tests for the N-step credit and the per-regime positive labels."""

import numpy as np
import pandas as pd

from facet0.refinement import credit, reward


def _scores(corpus, value_fn) -> pd.DataFrame:
    table = reward.compute_returns(corpus, reward.RewardConfig())
    table["value"] = [value_fn(e, f) for e, f in zip(table["episode_index"], table["frame_index"])]
    return table


def test_undiscounted_credit_with_terminal_bootstrap():
    values = np.linspace(0.0, 1.0, 11)
    rewards = np.zeros(11)
    rewards[-1] = 1.0
    delta, v_next, reward_sum = credit.episode_credit(values, rewards, horizon=3)
    np.testing.assert_allclose(delta[:7], 0.3, atol=1e-12)  # linear V: constant, position-independent
    assert v_next[-1] == 0.0 and reward_sum[-1] == 1.0
    np.testing.assert_allclose(delta[-1], 1.0 - 1.0)  # r_T + 0 - V_T
    np.testing.assert_allclose(delta[8], 1.0 + 0.0 - 0.8)  # window reaches the terminal reward, no discount


def test_same_top_fraction_in_each_regime(corpus):
    returns = reward.compute_returns(corpus, reward.RewardConfig())
    rng = np.random.default_rng(0)
    noise = {k: rng.normal() for k in zip(returns.episode_index, returns.frame_index)}
    scores = _scores(corpus, lambda e, f: noise[(e, f)])
    table = credit.compute_credit(corpus, scores, returns, credit.CreditConfig(horizon=5, positive_fraction=0.25))
    regime = table["contact_regime"].to_numpy()
    assert regime.any() and (~regime).any()
    assert abs(table.loc[regime, "label"].mean() - 0.25) < 0.05
    assert abs(table.loc[~regime, "label"].mean() - 0.25) < 0.05
    np.testing.assert_array_equal(table["contact_regime"], table["contact_intensity"] >= 1.0)
