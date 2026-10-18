from types import SimpleNamespace

import numpy as np
import pytest

from facet_adaptation.integrations.serve import FrozenPolicyServer, checkpoint_id


def test_wire_episode_reset_clears_policy_cache():
    calls = []
    server = object.__new__(FrozenPolicyServer)
    server.policy = SimpleNamespace(reset=lambda: calls.append(True))
    assert server.infer({"facet_control": "reset"}) == {"reset": True}
    assert calls == [True]


def test_checkpoint_identity_covers_normalization_and_config(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "params/_METADATA").write_text("metadata")
    stats = tmp_path / "assets/norm_stats.json"
    stats.write_text("first")
    first = checkpoint_id(tmp_path, "openpi", "cartesian")
    stats.write_text("second")
    assert first != checkpoint_id(tmp_path, "openpi", "cartesian")
    assert first != checkpoint_id(tmp_path, "openpi", "other_config")


def test_facet_wire_gripper_and_predicted_wrench_boundary():
    p = object.__new__(FrozenPolicyServer)
    p.backend, p.emit_prefix, p.token = "facet0", False, None
    captured = []

    def infer(obs):
        captured.append(obs)
        actions = np.ones((50, 7), np.float32)
        actions[:3, 6] = [1, 0.6, 0]
        return {"actions": actions, "wrench": np.ones((50, 6))}

    p.policy = SimpleNamespace(infer=infer)
    obs = {
        "observation/state": np.r_[np.zeros(6), 0.2, np.zeros(6)].astype(np.float32),
        "observation/wrench_history": np.zeros((10, 6)),
        "observation/right_wrist_image": np.zeros((32, 32, 3), np.uint8),
    }
    result = p.infer(obs)
    assert captured[0]["observation/state"][6] == pytest.approx(0.8)
    assert obs["observation/state"][6] == pytest.approx(0.2)
    np.testing.assert_allclose(result["actions"][:3, 6], [0, 0.4, 1], atol=1e-7)
    assert result["actions"].shape == (50, 7) and "predicted_wrench" in result


def test_invalid_wrench_history_never_reaches_policy():
    p = object.__new__(FrozenPolicyServer)
    p.backend = "facet0"
    with pytest.raises(ValueError, match="wrench"):
        p._model_obs({"observation/state": np.zeros(13), "observation/wrench_history": np.zeros((1, 6))})
