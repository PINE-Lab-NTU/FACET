from dataclasses import replace
import numpy as np
import torch

from facet_adaptation.learner import Learner
from facet_adaptation.safety import SafetyGuard, RobotState
import time


def row(learner, human=False):
    x = learner.state_vector(np.ones(learner.config.embedding_dim), np.zeros(13))
    ref = np.r_[np.zeros(6), 1.0].astype(np.float32)
    action = learner.act(x, ref)
    nxt = x.copy()
    nxt[-6:] = 1.0
    return dict(x=x, next_x=nxt, reference=ref, next_reference=ref.copy(), action=action, human=human)


def test_human_reference_and_success_filter(learner_config):
    learner = Learner(replace(learner_config, intervention_updates_per_episode=0))
    item = row(learner, human=True)
    learner.finish_episode([item], success=False)
    np.testing.assert_array_equal(learner.replay._a_tilde[0], item["reference"])
    assert not learner.interventions
    item["bc_confirmed"] = True
    learner.finish_episode([item], success=False)
    assert len(learner.interventions) == 1


def test_wrench_head_updates_and_target_actor_is_saved(learner_config, tmp_path):
    learner = Learner(
        replace(
            learner_config,
            intervention_updates_per_episode=0,
            updates_per_transition=0,
            reference_dropout=0.0,
        ),
        mode="rl",
    )
    rows = [row(learner) for _ in range(4)]
    learner.finish_episode(rows, success=True)
    last = [m for m in learner.actor.mlp.net if isinstance(m, torch.nn.Linear)][-1]
    before = last.weight[7:].detach().clone()
    metrics = learner.update_td3()
    assert metrics["wrench_valid_rows"] == 4 and metrics["wrench_loss"] > 0
    assert not torch.equal(before, last.weight[7:])
    path = tmp_path / "actor.pt"
    learner.save(path, {})
    loaded = Learner(learner.config, mode="eval")
    loaded.load(path, {})
    for a, b in zip(learner.target_actor.parameters(), loaded.target_actor.parameters()):
        torch.testing.assert_close(a, b)


def test_training_execution_map_matches_guard(learner_config):
    learner = Learner(learner_config)
    item = row(learner)
    proposed = np.array([0.5, 0.3, 0.1, 0.2, 0.3, 0.4, 0.8], np.float32)
    state = RobotState(np.zeros(6), 0.0, np.zeros(6), time.monotonic())
    actual = SafetyGuard(learner.safety).project(proposed, state, learner.control_dt)
    np.testing.assert_allclose(learner.applied(proposed, item["x"]), actual)
    value = torch.tensor(proposed, requires_grad=True)
    result = learner.applied(value, torch.tensor(item["x"]))
    np.testing.assert_allclose(result.detach(), actual)
    result.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_explicit_rewards_are_preserved(learner_config):
    learner = Learner(replace(learner_config, updates_per_transition=0), mode="rl")
    rows = [row(learner), row(learner)]
    rows[0]["reward"], rows[1]["reward"] = -0.5, 1.5
    learner.finish_episode(rows, success=True)
    np.testing.assert_array_equal(learner.replay._rewards[:2, 0], [-0.5, 1.5])


def test_resume_rejects_changed_td_rule(learner_config, tmp_path):
    learner = Learner(learner_config)
    path = tmp_path / "actor.pt"
    learner.save(path, {})
    import pytest

    with pytest.raises(ValueError, match="configuration mismatch"):
        Learner(replace(learner_config, critic_target="mc")).load(path, {})
