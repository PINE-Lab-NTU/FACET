import json
import numpy as np
import pandas as pd
import pytest

from facet0.contracts import write_manifest, read_manifest, identity
from facet0.data.split import build, load
from facet0.refinement import reward, credit
from conftest import write_corpus, contact_episode


def test_parameter_and_normalization_contract(tmp_path):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="bfloat16",
        action_dim=13,
        action_horizon=50,
        wrench_window=10,
        max_token_len=200,
        pi05=True,
        discrete_state_input=False,
        vision_variant="So400m/14",
    )
    (tmp_path / "params").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "params/weights").write_bytes(b"first")
    (tmp_path / "assets/norm_stats.json").write_text("{}")
    first = write_manifest(tmp_path, cfg, stage="pretraining")
    assert identity(read_manifest(tmp_path)) == identity(first)
    (tmp_path / "params/weights").write_bytes(b"second")
    with pytest.raises(ValueError, match="parameter checksum"):
        read_manifest(tmp_path)


def test_split_groups_and_unknown_tiers(tmp_path):
    (tmp_path / "meta").mkdir()
    rows = [{"episode_index": i, "source_demo_id": f"source{i // 2}"} for i in range(8)]
    (tmp_path / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = build(tmp_path)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(result))
    load(path)
    seen = {}
    for row in result["episodes"]:
        assert seen.setdefault(row["source_demo_id"], row["split"]) == row["split"]


def test_missing_phase_and_partial_judge_fail():
    state = np.zeros((2, 13))
    state[:, 7] = 20
    frame = pd.DataFrame(
        {
            "observation.state": list(state),
            "is_success": [0, 0],
            "subgoal_index": [0, 0],
            "subgoal_complete": [0, 0],
            "phase": ["align", "missing"],
        }
    )
    cfg = reward.RewardConfig(alpha=1.0, time_budget=20.0, envelopes={"align": (10.0,) * 6})
    with pytest.raises(ValueError, match="envelope"):
        reward._episode_reward(frame, cfg)
    with pytest.raises(ValueError, match="incomplete judge"):
        reward._episode_reward(frame.drop(columns="phase"), cfg)
    with pytest.raises(ValueError, match="require"):
        reward.RewardConfig().validate()


def test_ties_have_bounded_positive_budget(tmp_path):
    root = write_corpus(tmp_path / "c", [contact_episode(60, 30, False, 1)])
    returns = reward.compute_returns(root, reward.RewardConfig(sparse=True))
    scores = returns[["episode_index", "frame_index"]].assign(value=0.0)
    result = credit.compute_credit(root, scores, returns, credit.CreditConfig())
    for _, part in result.groupby("contact_regime"):
        assert part.label.sum() == max(1, int(np.ceil(len(part) * 0.2)))
    with pytest.raises(ValueError, match="incomplete"):
        credit.compute_credit(root, scores.iloc[:10], returns, credit.CreditConfig())


def test_regression_uses_behavior_and_bootstrap_uses_policy():
    from test_critic import _synthetic

    data = _synthetic()
    with pytest.raises(ValueError, match="recorded behavior"):
        data.batch(np.zeros(4, int), np.random.default_rng(0))
    data.behavior = np.ones(data.proposal[:, 0].shape, np.float32) * 123
    batch = data.batch(np.zeros(4, int), np.random.default_rng(0))
    np.testing.assert_array_equal(batch["proposal"], 123)
    assert not np.allclose(batch["next_proposal"], 123)
