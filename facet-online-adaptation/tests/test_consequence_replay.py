from dataclasses import replace

import numpy as np
import torch

from facet_adaptation.learner import Learner


def test_interventions_different_from_actor_still_train_wrench(learner_config):
    learner = Learner(replace(
        learner_config, intervention_updates_per_episode=0, updates_per_transition=0, reference_dropout=0,
        critic_gate=False, critic_updates_per_actor=1,
    ), mode="rl")
    x = learner.state_vector(np.ones(learner_config.embedding_dim), np.r_[np.zeros(6), 0.5, np.zeros(6)])
    reference = np.r_[np.zeros(6), 0.5].astype(np.float32)
    executed = learner.act(x, reference)
    executed[6] = 0.95 if executed[6] < 0.7 else 0.05
    nxt = x.copy()
    nxt[-6:] = 1
    rows = [dict(x=x, next_x=nxt, action=executed, reference=reference, next_reference=reference, human=True)] * 4
    learner.finish_episode(rows, success=True)
    before = {k: p.detach().clone() for k, p in learner.actor.wrench_head.named_parameters()}
    metrics = learner.update_td3()
    assert metrics["wrench_valid_rows"] == 4 and metrics["wrench_coverage"] == 1
    assert metrics["wrench_loss"] > 0
    assert any(not torch.equal(before[k], p) for k, p in learner.actor.wrench_head.named_parameters())


def test_consequence_depends_on_executed_action(learner_config):
    learner = Learner(learner_config)
    x = torch.ones(2, learner_config.embedding_dim + 13)
    ref = torch.zeros(2, 7)
    commands = torch.zeros(2, 7)
    commands[1, 6] = 1
    learner.actor.eval()
    _, wrench = learner.actor.outputs(x, ref, executed_action=commands, reference_dropout=False)
    assert not torch.allclose(wrench[0], wrench[1])
