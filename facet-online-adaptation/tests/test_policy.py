from dataclasses import replace
import time

import numpy as np
import pytest

from facet_adaptation.config import PolicyConfig
from facet_adaptation.policy import FrameCodec, RemotePolicy, WrenchHistory
from facet_adaptation.safety import RobotState


class Client:
    def __init__(self, token=True):
        self.token = token

    def get_server_metadata(self):
        return {
            "facet_adaptation": {
                "schema": 1,
                "frozen": True,
                "gripper": "1=open",
                "checkpoint_id": "vla",
                "token_id": "token",
            }
        }

    def infer(self, obs):
        out = {"actions": np.zeros((50, 7), np.float32)}
        if self.token:
            out["rl_token"] = np.ones(16)
        return out


def test_frozen_policy_contract_and_real_token_required():
    cfg = PolicyConfig(checkpoint_id="vla", token_id="token")
    RemotePolicy(cfg, client=Client()).infer({})
    with pytest.raises(ValueError, match="token"):
        RemotePolicy(cfg, client=Client(False)).infer({})
    with pytest.raises(ValueError, match="contract"):
        RemotePolicy(replace(cfg, token_id="other"), client=Client())
    RemotePolicy(cfg, client=Client(False), adaptation=False).infer({})


def test_explicit_frame_sign_is_invertible_without_moving_orientation():
    codec = FrameCodec(PolicyConfig(position_sign=[-1, 1, 1]))
    pose = np.array([0.4, 0.1, 0.2, 2.9, 0.1, 0])
    state = RobotState(pose, 0.37, np.arange(6), time.monotonic())
    encoded = codec.state(state)
    assert encoded[0] < 0 and encoded[6] == pytest.approx(0.37)
    decoded = codec.native_action(encoded[:7])
    np.testing.assert_allclose(decoded[:6], pose)


def test_history_is_causal_and_initially_padded():
    history = WrenchHistory()
    history.append(np.ones(6))
    np.testing.assert_array_equal(history.window(), np.ones((10, 6)))
    history.append(np.full(6, 2))
    assert history.window()[-1, 0] == 2 and history.window()[0, 0] == 1
