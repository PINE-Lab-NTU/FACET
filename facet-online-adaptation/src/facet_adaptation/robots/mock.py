"""Deterministic in-memory backend for hardware-free workflow tests."""

import time

import numpy as np

from ..safety import RobotState, SafetyGuard


class MockRobot:
    def __init__(self, safety):
        self.state = np.zeros(13, np.float32)
        self.state[6] = 1
        self.guard = SafetyGuard(safety)
        self.commands, self.holds = [], 0

    def read(self):
        return RobotState(
            self.state[:6].copy(), float(self.state[6]), self.state[7:].copy(), time.monotonic()
        )

    def command(self, action, dt):
        result = self.guard.project(action, self.read(), dt)
        self.state[:7] = result
        self.commands.append(result.copy())
        return result

    def hold(self):
        self.holds += 1

    def close(self):
        self.hold()
