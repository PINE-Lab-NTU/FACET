"""Frozen-policy wire contract and explicit policy-frame encoding."""

from collections import deque
import time

import numpy as np

from .geometry import canonical_rotvec


class FrameCodec:
    def __init__(self, config):
        self.signs = np.asarray(config.position_sign, float)
        self.reference = config.reference_rotvec

    def state(self, native):
        p = native.vector()
        p[:3] *= self.signs
        p[3:6] = canonical_rotvec(p[3:6], self.reference)
        return p

    def native_action(self, policy_action):
        a = np.asarray(policy_action, np.float32).copy()
        a[:3] *= self.signs
        return a

    def policy_action(self, native_action):
        return self.native_action(native_action)


class WrenchHistory:
    def __init__(self, size=10, hz=15.0):
        self.hz = hz
        self.timestamps = deque(maxlen=256)
        self.size = size
        self.rows = deque(maxlen=256)

    def reset(self):
        self.rows.clear()
        self.timestamps.clear()

    def append(self, value, timestamp=None):
        value = np.asarray(value, np.float32)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError("invalid measured wrench history")
        timestamp = time.monotonic() if timestamp is None else timestamp
        if self.timestamps and timestamp < self.timestamps[-1]:
            raise ValueError("wrench timestamps must be monotonic")
        self.rows.append(value.copy())
        self.timestamps.append(timestamp)

    def window(self):
        if not self.rows:
            raise ValueError("no wrench samples")
        times = np.asarray(self.timestamps)
        grid = times[-1] - np.arange(self.size - 1, -1, -1) / self.hz
        ids = np.maximum(np.searchsorted(times, grid, side="right") - 1, 0)
        return np.asarray(self.rows)[ids]


class RemotePolicy:
    def __init__(self, config, *, adaptation=True, client=None):
        self.config, self.adaptation = config, adaptation
        if client is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            client = WebsocketClientPolicy(config.host, config.port)
        self.client = client
        meta = client.get_server_metadata().get("facet_adaptation", {})
        expected = {"schema": 1, "frozen": True, "gripper": "1=open", "checkpoint_id": config.checkpoint_id}
        if adaptation:
            expected["token_id"] = config.token_id
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("frozen policy/token wire-contract mismatch")
        self.contract = meta
        history_hz = meta.get("prefix_contract", {}).get("history_hz")
        if history_hz is not None and history_hz != config.history_hz:
            raise ValueError("policy wrench-history cadence mismatch")

    def infer(self, obs):
        started = time.monotonic()
        out = self.client.infer(obs)
        if time.monotonic() - started > self.config.max_latency_s:
            raise TimeoutError("policy inference exceeded latency limit")
        actions = np.asarray(out["actions"], np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < self.config.execute_steps:
            raise ValueError("policy must return a sufficiently long Hx7 action chunk")
        if not np.isfinite(actions).all() or np.any((actions[:, 6] < 0) | (actions[:, 6] > 1)):
            raise ValueError("invalid base-policy command")
        token = out.get("rl_token")
        if self.adaptation and (token is None or not np.isfinite(token).all() or not np.any(token)):
            raise ValueError("policy must supply a real frozen RL token")
        return actions, None if token is None else np.asarray(token, np.float32)

    def reset(self):
        if self.contract.get("supports_reset"):
            if self.client.infer({"facet_control": "reset"}).get("reset") is not True:
                raise ValueError("policy server did not acknowledge episode reset")
            return
        reset = getattr(self.client, "reset", None)
        if reset:
            reset()


class MockPolicy:
    def __init__(self, embedding_dim=2048):
        self.embedding_dim = embedding_dim
        self.contract = {
            "schema": 1,
            "frozen": True,
            "gripper": "1=open",
            "checkpoint_id": "mock-policy",
            "token_id": "mock-token",
        }

    def infer(self, obs):
        action = np.asarray(obs["observation/state"][:7], np.float32).copy()
        action[0] += 0.001
        return np.repeat(action[None], 50, 0), np.linspace(-1, 1, self.embedding_dim, dtype=np.float32)
