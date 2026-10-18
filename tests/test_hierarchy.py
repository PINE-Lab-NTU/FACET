import numpy as np
import pytest

from facet0.policy.hierarchy import JointPolicyHierarchy


class Expert:
    def __init__(self, value):
        self.calls = 0
        self.value = value

    def infer(self, obs):
        self.calls += 1
        return {"actions": np.full((50, 7), self.value), "wrench": np.full((50, 6), self.value + 1)}

    def contact_features(self, obs):
        return np.ones((2, 64)), np.ones(2, bool)


def test_coarse_5hz_fine_20hz_and_predictive_wrench():
    coarse, fine = Expert(1), Expert(2)
    policy = JointPolicyHierarchy(coarse, fine, wrench_envelope=[1] * 6)
    obs = {"observation/state": np.zeros(13), "contact_phase": False}
    for t in np.arange(0, 0.4, 0.05):
        result = policy.infer(obs, now=t)
        assert result["expert"] == "coarse" and result["wrench"].shape == (50, 6)
    assert coarse.calls == 2 and fine.calls == 0
    obs["contact_phase"] = True
    for t in np.arange(0.4, 0.8, 0.05):
        result = policy.infer(obs, now=t)
        assert result["expert"] == "refinement"
        np.testing.assert_array_equal(result["actions"], 2)
    assert coarse.calls == 4 and fine.calls == 8
    policy.reset()
    assert policy._coarse is None


def test_measured_wrench_trigger_and_bad_time():
    policy = JointPolicyHierarchy(Expert(1), Expert(2), wrench_envelope=[2] * 6)
    obs = {"observation/state": np.r_[np.zeros(7), np.ones(6)]}
    assert policy.infer(obs, now=1)["expert"] == "refinement"
    with pytest.raises(ValueError, match="timestamps"):
        policy.infer(obs, now=0)
