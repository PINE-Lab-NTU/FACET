import numpy as np
import pytest
import torch

from facet_adaptation.learner import Learner, actor_loss_components, critic_targets, discounted_returns


def episode(learner, count=6, human=True):
    x = learner.state_vector(np.linspace(-1, 1, learner.config.embedding_dim), np.zeros(13))
    rows = []
    for i in range(count):
        reference = np.array([0.1 + 0.001 * i, 0.2, 0.3, 0, 0, 0, 1], np.float32)
        a = reference.copy()
        a[2] -= 0.001
        rows.append(
            dict(
                x=x.copy(),
                reference=reference,
                action=a,
                next_x=x.copy(),
                next_reference=reference.copy(),
                human=human,
            )
        )
    return rows


def test_mc_returns_and_unknown_td_fallback():
    np.testing.assert_allclose(discounted_returns([0, 0, 1], 0.9), [0.81, 0.9, 1])
    td, mc = torch.tensor([[0.2], [0.4], [0.6]]), torch.tensor([[0.8], [float("nan")], [0.1]])
    torch.testing.assert_close(critic_targets(td, mc, "mc"), torch.tensor([[0.8], [0.4], [0.1]]))
    torch.testing.assert_close(critic_targets(td, mc, "td_mc_max"), torch.tensor([[0.8], [0.4], [0.6]]))


def test_demo_only_bc_has_no_failed_online_actor_gradient():
    proposed = torch.zeros(4, 7, requires_grad=True)
    target = torch.ones(4, 7)
    weights = torch.tensor([0, 0, 1, 1])
    loss, _, bc = actor_loss_components(torch.zeros(4, 1), proposed, target, 1, weights)
    loss.backward()
    assert bc == 7
    assert torch.count_nonzero(proposed.grad[:2]) == 0
    assert torch.count_nonzero(proposed.grad[2:]) == 14


def test_failure_kept_for_critic_excluded_from_demo(learner_config):
    learner = Learner(learner_config)
    learner.finish_episode(episode(learner), success=False)
    assert learner.replay.size == 6 and learner.demo.size == 0
    assert len(learner.interventions) == 0
    np.testing.assert_array_equal(learner.replay._mc_returns[:6], 0)
    learner.finish_episode(episode(learner, human=False), success=True)
    assert learner.demo.size == 0
    np.testing.assert_allclose(
        learner.replay._mc_returns[6:12, 0], learner_config.gamma ** np.arange(5, -1, -1)
    )


def test_c1_td3_updates_and_full_resume(learner_config, tmp_path):
    learner = Learner(learner_config, mode="rl")
    learner.finish_episode(episode(learner), success=True)
    # rl mode: confirmed labels reach the Actor through the annealed TD3+BC term, not extra BC steps,
    # and the Actor is held at its BC seed until the Critic gate opens.
    assert learner.updates == 6 and learner.bc_updates == 0
    assert learner.td3_actor_updates == 0 and learner.held_actor_updates == 3
    assert learner.last_metrics["critic_loss"] >= 0
    path = tmp_path / "model.pt"
    contract = {"policy": "test", "frame": "native"}
    learner.save(path, contract)
    restored = Learner(learner_config, mode="eval")
    restored.load(path, contract)
    x, ref = episode(learner)[0]["x"], episode(learner)[0]["reference"]
    np.testing.assert_array_equal(learner.act(x, ref), restored.act(x, ref))
    assert restored.replay.size == learner.replay.size
    assert restored.updates == learner.updates
    with pytest.raises(ValueError, match="contract"):
        restored.load(path, {"policy": "different"})


def test_eval_is_frozen(learner_config):
    learner = Learner(learner_config, mode="eval")
    before = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    learner.finish_episode(episode(learner), success=True)
    assert learner.replay.size == 0 and learner.updates == learner.bc_updates == 0
    for key, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        learner.update_bc()


def test_training_and_execution_have_identical_action_boxes(learner_config):
    learner = Learner(learner_config)
    ref = np.array([0.1, -0.2, 0.3, 2.9, -0.3, 0.1, 1], np.float32)
    proposal = np.ones(7, np.float32) * 100
    a = learner.project(proposal, ref)
    b = learner.project(torch.tensor(proposal), torch.tensor(ref)).numpy()
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(a, learner_config.action_max)


def test_actor_has_four_modalities_and_two_heads(learner_config):
    learner = Learner(learner_config)
    row = episode(learner)[0]
    a, w = learner.actor.outputs(learner.tensor(row["x"])[None], learner.tensor(row["reference"])[None])
    assert a.shape == (1, 7) and w.shape == (1, 6)
    assert learner.actor.encoder.output_dim == 4 * learner_config.projection_dim
    assert all(not p.requires_grad for p in learner.target_actor.parameters())


def test_invalid_episode_is_atomic(learner_config):
    learner = Learner(learner_config)
    rows = episode(learner)
    rows[-1]["action"][0] = np.nan
    with pytest.raises(ValueError):
        learner.finish_episode(rows, success=True)
    assert learner.replay.size == learner.demo.size == learner.episodes == 0


def test_fuse_blocks_updates_and_actor_execution(learner_config):
    learner = Learner(learner_config)
    learner.fused = True
    row = episode(learner)[0]
    np.testing.assert_array_equal(learner.act(row["x"], row["reference"]), row["reference"])
    with pytest.raises(RuntimeError):
        learner.finish_episode([row], success=True)


def test_unknown_legacy_checkpoints_rejected(learner_config, tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": "rlt_robot_online_full_v4"}, path)
    with pytest.raises(ValueError, match="legacy"):
        Learner(learner_config).load(path, {})


def test_absolute_action_box_is_independent_of_reference(learner_config):
    learner = Learner(learner_config)
    row = episode(learner)[0]
    a = (
        learner.actor(learner.tensor(row["x"])[None], torch.full((1, 7), 100.0), reference_dropout=False)[0]
        .detach()
        .numpy()
    )
    assert np.all(a >= learner_config.action_min) and np.all(a <= learner_config.action_max)
    assert not np.allclose(a[3:], row["reference"][3:])
