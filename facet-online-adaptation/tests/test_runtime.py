import hashlib
import json

import numpy as np
import pytest

from facet_adaptation.cameras import MockCameras
from facet_adaptation.policy import MockPolicy
from facet_adaptation.robots.mock import MockRobot
from facet_adaptation.runtime import Session


def make(config, **kwargs):
    robot = MockRobot(config.safety)
    return Session(config, robot, MockPolicy(config.learner.embedding_dim), MockCameras(), **kwargs)


def test_no_motion_before_explicit_episode_start_and_no_reset_after_label(config):
    s = make(config)
    s.tick()
    assert s.robot.commands == []
    s.command("resume")
    for _ in range(6):
        s.tick()
    before = len(s.robot.commands)
    terminal_pose = s.robot.read().pose.copy()
    s.command("success")
    assert s.paused and len(s.robot.commands) == before
    np.testing.assert_array_equal(s.robot.read().pose, terminal_pose)
    assert s.learner.replay.size == 6
    s.close()


def test_incomplete_episode_never_gets_implicit_failure_reward(config):
    s = make(config)
    s.begin()
    s.tick()
    s.command("pause")
    assert s.learner.replay.size == 0
    result = json.loads((s.run / "episodes/episode_000001/outcome.json").read_text())
    assert result["success"] is None and result["steps"] == 1
    s.close()


def test_restart_has_new_run_and_preserves_original_checkpoint(config):
    s = make(config)
    s.begin()
    s.tick()
    s.finish(True)
    path = s.run / "checkpoint_ep000001.pt"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    next_session = make(config, resume=str(path))
    assert next_session.run != s.run
    next_session.begin()
    next_session.tick()
    next_session.finish(False)
    next_session.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    s.close()


def test_executed_actions_not_proposals_are_stored(config):
    s = make(config)
    s.begin()
    s.learner.act = lambda x, ref: ref + np.array([100, 0, 0, 0, 0, 0, 0])
    s.tick()
    np.testing.assert_array_equal(s.pending["action"], s.robot.commands[-1])
    assert s.pending["action"][0] < 0.01
    s.close()


def test_base_rollout_does_not_require_token(config):
    config.runtime.mode = "base"
    s = make(config)
    s.begin()
    s.tick()
    s.finish(True)
    assert s.learner.updates == s.learner.replay.size == 0
    s.close()


def test_eval_requires_actor_checkpoint(config):
    config.runtime.mode = "eval"
    with pytest.raises(ValueError, match="resume"):
        make(config)


def test_history_and_chunk_cache_reset_between_episodes(config):
    s = make(config)
    s.begin()
    s.tick()
    s.finish(False)
    s.begin()
    assert not s.history.rows and s.actions is None and s.pending is None
    s.close()


def test_terminal_state_fault_does_not_create_reward(config):
    s = make(config)
    s.begin()
    s.tick()
    s.robot.read = lambda: (_ for _ in ()).throw(RuntimeError("disconnected"))
    with pytest.raises(RuntimeError, match="disconnected"):
        s.finish(True)
    assert s.learner.replay.size == 0
    outcome = json.loads((s.run / "episodes/episode_000001/outcome.json").read_text())
    assert outcome["success"] is None
    s.close()
