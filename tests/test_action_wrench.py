"""Tests for the next-step wrench pairing, validity masks and causal wrench history."""

import numpy as np

from facet0 import action_wrench


def _states(n=12):
    s = np.zeros((n, 13), np.float32)
    s[:, 7:13] = np.arange(n * 6, dtype=np.float32).reshape(n, 6)
    return s


def test_row_k_pairs_with_the_following_step():
    states = _states()
    w_bar = action_wrench.filtered_wrench(states, window=1)
    targets, valid = action_wrench.wrench_targets(w_bar, frame=3, horizon=5)
    np.testing.assert_array_equal(targets, states[4:9, 7:13])
    assert valid.all()


def test_masks_at_episode_end():
    states = _states(12)
    targets, valid = action_wrench.wrench_targets(action_wrench.filtered_wrench(states), frame=9, horizon=5)
    np.testing.assert_array_equal(valid, [True, True, False, False, False])
    np.testing.assert_array_equal(action_wrench.action_valid(9, 5, 12), [True, True, True, False, False])


def test_history_is_causal_oldest_first_and_clamped():
    states = _states()
    h = action_wrench.wrench_history(states, frame=2, window=4)
    np.testing.assert_array_equal(h, states[[0, 0, 1, 2], 7:13])
    h = action_wrench.wrench_history(states, frame=8, window=4)
    np.testing.assert_array_equal(h, states[5:9, 7:13])


def test_targets_are_causally_filtered():
    states = _states()
    w_bar = action_wrench.filtered_wrench(states, window=3)
    targets, _ = action_wrench.wrench_targets(w_bar, frame=5, horizon=1)
    np.testing.assert_allclose(targets[0], states[4:7, 7:13].mean(0))
