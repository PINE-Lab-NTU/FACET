"""Critic-gated TD3+BC: hold the BC seed, open on held-out Critic checks, anneal BC on the RL clock."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from facet_adaptation.config import LearnerConfig
from facet_adaptation.gate import CriticReadinessGate, GateLimits, annealed_beta
from facet_adaptation.learner import Learner, critic_lr_at


def episode(learner, count=6, human=True, offset=0.0):
    x = learner.state_vector(np.linspace(-1, 1, learner.config.embedding_dim), np.zeros(13))
    rows = []
    for i in range(count):
        reference = np.array([0.1 + 0.001 * i + offset, 0.2, 0.3, 0, 0, 0, 1], np.float32)
        action = reference.copy()
        action[2] -= 0.001
        nxt = x.copy()
        nxt[-6:] = 0.1 * i
        rows.append(dict(x=x.copy(), reference=reference, action=action, next_x=nxt,
                         next_reference=reference.copy(), human=human))
    return rows


def actor_vector(learner):
    return torch.cat([p.detach().reshape(-1).clone() for p in learner.actor.parameters()])


def test_defaults_follow_the_validated_recipe():
    c = LearnerConfig()
    assert (c.critic_target, c.td3_bc_target, c.demo_success_only) == ("mc", "demo", True)
    assert (c.actor_lr, c.critic_lr, c.critic_updates_per_actor, c.updates_per_transition) == (1e-6, 1e-4, 20, 5)
    assert (c.bc_beta, c.bc_beta_end, c.bc_anneal_critic_updates) == (1.0, 0.3, 3000)
    assert c.critic_gate and c.hold_actor_until_critic_ready and c.actor_q_weight == 1.0
    assert (c.gate_min_updates, c.gate_consecutive_passes, c.gate_seed_drift_max_m) == (750, 3, 0.001)
    assert c.rl_intervention_updates_per_episode == 0


def test_invalid_schedules_are_rejected():
    with pytest.raises(ValueError, match="monitor"):
        replace(LearnerConfig(), monitor_every=0).validate()
    replace(LearnerConfig(), monitor_every=0, critic_gate=False).validate()
    with pytest.raises(ValueError, match="bc_beta_end"):
        replace(LearnerConfig(), bc_beta_end=2.0).validate()
    with pytest.raises(ValueError, match="actor_q_weight"):
        replace(LearnerConfig(), actor_q_weight=-1.0).validate()


def test_annealed_beta_and_critic_lr_schedules():
    assert annealed_beta(1.0, 0.3, 3000, 0) == 1.0
    assert annealed_beta(1.0, 0.3, 3000, 1500) == pytest.approx(0.65)
    assert annealed_beta(1.0, 0.3, 3000, 9000) == pytest.approx(0.3)
    assert annealed_beta(1.0, 0.3, 0, 100) == 1.0
    c = replace(LearnerConfig(), critic_lr_schedule="cosine", gate_min_updates=10, critic_lr_decay_updates=20)
    assert critic_lr_at(c, 5) == pytest.approx(1e-4)
    assert critic_lr_at(c, 20) == pytest.approx((1e-4 + 2e-5) / 2)
    assert critic_lr_at(c, 100) == pytest.approx(2e-5)
    assert critic_lr_at(LearnerConfig(), 10_000) == 1e-4


def passing():
    return dict(success_terminals=5, failure_terminals=5, terminal_brier=0.05, success_terminal_q=0.8,
                failure_terminal_q=0.1, q_disagreement=0.01, seed_drift_p95_m=0.0005, bc_degradation=1.0)


def test_gate_needs_consecutive_passes_and_closes_on_any_failure():
    gate = CriticReadinessGate(GateLimits(min_updates=10, consecutive_passes=3))
    probe = (np.ones(4), np.ones(4))
    assert gate.observe(10, passing(), probe)["gate_check_passed"] == 0.0  # no previous probe yet
    for _ in range(2):
        gate.observe(10, passing(), probe)
    assert not gate.active
    gate.observe(10, passing(), probe)
    assert gate.active and gate.events[-1]["event"] == "opened"
    bad = dict(passing(), seed_drift_p95_m=0.002)
    gate.observe(10, bad, probe)
    assert not gate.active and gate.last_failed == ["actor_seed_drift"] and gate.pass_streak == 0
    assert gate.events[-1]["event"] == "paused"


@pytest.mark.parametrize(
    "change, failed",
    [
        (dict(failure_terminals=2), "monitor_coverage"),
        (dict(success_terminal_q=0.6), "success_terminal_q"),
        (dict(failure_terminal_q=-0.4), "failure_terminal_q"),
        (dict(terminal_brier=float("nan")), "terminal_calibration"),
        (dict(bc_degradation=1.2), "successful_behavior"),
    ],
)
def test_each_check_blocks_the_gate(change, failed):
    gate = CriticReadinessGate(GateLimits(min_updates=0, consecutive_passes=1))
    probe = (np.ones(3), np.ones(3))
    gate.observe(0, passing(), probe)
    gate.observe(0, dict(passing(), **change), probe)
    assert failed in gate.last_failed and not gate.active


def test_minimum_updates_and_probe_change_block_the_gate():
    gate = CriticReadinessGate(GateLimits(min_updates=100, consecutive_passes=1))
    gate.observe(50, passing(), (np.ones(3), np.ones(3)))
    gate.observe(50, passing(), (np.ones(3), np.ones(3)))
    assert gate.last_failed == ["minimum_critic_updates"]
    gate.observe(100, passing(), (np.ones(3), np.ones(3) + 0.05))
    assert gate.last_failed == ["target_q_stability"]
    gate.observe(100, passing(), (np.ones(5), np.ones(5)))  # monitor grew: not comparable
    assert set(gate.last_failed) == {"fixed_q_stability", "target_q_stability"}


def test_every_nth_labelled_episode_is_held_out_of_training(learner_config):
    learner = Learner(replace(learner_config, monitor_every=2))
    learner.finish_episode(episode(learner), success=True)
    assert (learner.replay.size, len(learner.monitor), len(learner.interventions)) == (6, 0, 6)
    metrics = learner.finish_episode(episode(learner), success=False)
    assert (learner.replay.size, len(learner.monitor), len(learner.interventions)) == (6, 1, 6)
    assert metrics["held_out_for_monitor"] == 1.0 and metrics["episode_bc_updates"] == 0


def test_intervention_bc_only_runs_for_new_confirmed_labels(learner_config):
    learner = Learner(replace(learner_config, intervention_updates_per_episode=3))
    learner.finish_episode(episode(learner), success=True)
    assert learner.bc_updates == 3
    learner.finish_episode(episode(learner, human=False), success=True)  # autonomous: no new labels
    assert learner.bc_updates == 3


def test_demo_rules_are_wired(learner_config):
    loose = Learner(replace(learner_config, demo_success_only=False, intervention_updates_per_episode=0))
    loose.finish_episode(episode(loose), success=False)
    assert loose.demo.size == 6 and not loose.interventions
    seeded = Learner(replace(learner_config, demo_seed_success_episodes=1, intervention_updates_per_episode=0))
    seeded.finish_episode(episode(seeded, human=False), success=True)
    seeded.finish_episode(episode(seeded, human=False), success=True)
    assert seeded.demo.size == 6 and seeded.demo_successes == 1


def rl_learner(config, **overrides):
    settings = dict(intervention_updates_per_episode=0, updates_per_transition=0, reference_dropout=0.0,
                    critic_updates_per_actor=1, gate_eval_every=10_000)
    learner = Learner(replace(config, **{**settings, **overrides}), mode="rl")
    learner.finish_episode(episode(learner), success=True)
    learner.finish_episode(episode(learner, human=False, offset=0.01), success=False)
    return learner


def test_actor_is_held_at_its_seed_until_the_gate_opens(learner_config):
    learner = rl_learner(learner_config)
    before = actor_vector(learner)
    critic_before = torch.cat([p.detach().reshape(-1).clone() for p in learner.critic.q1.parameters()])
    for _ in range(4):
        metrics = learner.update_td3()
    assert metrics["actor_held_for_critic_gate"] == 1.0 and learner.held_actor_updates == 4
    torch.testing.assert_close(actor_vector(learner), before, rtol=0, atol=0)
    assert not torch.equal(critic_before, torch.cat([p.reshape(-1) for p in learner.critic.q1.parameters()]))
    assert learner.rl_active_critic_updates == 0


def test_open_gate_applies_q_gradient_and_anneals_only_while_open(learner_config):
    learner = rl_learner(learner_config, bc_anneal_critic_updates=4)
    learner.update_td3()  # creates the BC seed
    learner.gate.active = True
    before = actor_vector(learner)
    betas = [learner.update_td3()["bc_beta_effective"] for _ in range(2)]
    assert betas == pytest.approx([1.0 - 0.7 / 4, 1.0 - 1.4 / 4])
    assert learner.td3_actor_updates == 2 and not torch.equal(before, actor_vector(learner))
    learner.gate.active = False
    paused = actor_vector(learner)
    metrics = learner.update_td3()
    assert metrics["bc_beta_effective"] == 1.0 and learner.rl_active_critic_updates == 2
    torch.testing.assert_close(actor_vector(learner), paused, rtol=0, atol=0)


def test_without_hold_the_closed_gate_still_trains_bc_without_q(learner_config):
    learner = rl_learner(learner_config, hold_actor_until_critic_ready=False)
    before = actor_vector(learner)
    metrics = learner.update_td3()
    assert metrics["actor_uses_q_gradient"] == 0.0 and learner.td3_actor_updates == 0
    assert not torch.equal(before, actor_vector(learner))


def run_open(config, q_weight, perturb_critic):
    learner = rl_learner(config, critic_gate=False, actor_q_weight=q_weight)
    if perturb_critic:
        generator = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for p in learner.critic.parameters():
                p.add_(torch.randn(p.shape, generator=generator))
    np.random.seed(3)
    torch.manual_seed(3)
    for _ in range(3):
        learner.update_td3()
    return actor_vector(learner)


def test_matched_bc_control_differs_only_by_the_q_term(learner_config):
    # q weight 0: identical batches/BC/wrench steps, so the Actor cannot depend on the Critic.
    torch.testing.assert_close(run_open(learner_config, 0.0, False), run_open(learner_config, 0.0, True),
                               rtol=0, atol=0)
    # With the Q term the same perturbation changes the Actor.
    assert not torch.equal(run_open(learner_config, 1.0, False), run_open(learner_config, 1.0, True))


def test_gate_metrics_use_the_held_out_monitor(learner_config):
    learner = rl_learner(learner_config, monitor_every=1, gate_eval_every=1, gate_min_terminals=1)
    assert learner.replay.size == 0 and len(learner.monitor) == 2
    learner.finish_episode(episode(learner), success=True)  # monitor too; replay stays empty
    x = episode(learner)[0]["x"]
    for _ in range(learner.config.batch_size):  # unrelated training rows so an update can run
        learner.replay.add(x=x, a=np.zeros(7), a_tilde=np.zeros(7), next_a_tilde=np.zeros(7),
                           rewards=np.zeros(1), next_x=x, done=0.0)
    metrics = learner.update_td3()
    assert metrics["gate/success_terminals"] == 2 and metrics["gate/failure_terminals"] == 1
    assert metrics["gate/seed_drift_p95_m"] == 0.0 and metrics["gate/bc_degradation"] == pytest.approx(1.0)
    assert "minimum_critic_updates" in metrics["gate_failed_checks"]


def test_checkpoint_keeps_gate_seed_monitor_and_clock(learner_config, tmp_path):
    learner = rl_learner(learner_config, monitor_every=2)
    learner.update_td3()
    learner.gate.active, learner.gate.pass_streak = True, 4
    learner.update_td3()
    path = tmp_path / "v4.pt"
    learner.save(path, {})
    restored = Learner(learner.config, mode="rl")
    restored.load(path, {})
    assert restored.gate.active and restored.gate.pass_streak == 4
    assert restored.rl_active_critic_updates == learner.rl_active_critic_updates == 1
    assert restored.labelled_episodes == 2 and len(restored.monitor) == 1
    for a, b in zip(learner.seed_actor.parameters(), restored.seed_actor.parameters()):
        torch.testing.assert_close(a, b)


def test_v3_checkpoint_migrates_with_a_fresh_gate(learner_config, tmp_path):
    learner = rl_learner(learner_config)
    path = tmp_path / "v3.pt"
    learner.save(path, {})
    payload = torch.load(path, weights_only=False)
    payload["format"] = "facet_online_adaptation_v3"
    for key in ("seed_actor", "gate", "monitor", "labelled_episodes", "rl_active_critic_updates",
                "td3_actor_updates", "held_actor_updates"):
        payload.pop(key)
    for key in ("critic_gate", "monitor_every", "bc_beta_end", "actor_q_weight"):
        payload["config"].pop(key)
    torch.save(payload, path)
    restored = Learner(learner.config, mode="rl")
    restored.load(path, {})
    assert not restored.gate.active and restored.seed_actor is None and restored.labelled_episodes == 2
    with pytest.raises(ValueError, match="critic_target"):
        Learner(replace(learner.config, critic_target="td"), mode="rl").load(path, {})


def test_gate_metrics_are_strict_json_before_the_monitor_exists(learner_config):
    import json

    learner = rl_learner(learner_config, gate_eval_every=1)
    metrics = learner.update_td3()
    assert metrics["gate/success_terminals"] == 0 and metrics["gate/terminal_brier"] is None
    assert metrics["gate_fixed_q_delta_rms"] is None
    json.dumps(metrics, allow_nan=False)
