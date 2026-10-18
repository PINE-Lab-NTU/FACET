"""Measured-state safety bounds, independent of task completion or target geometry."""

from dataclasses import dataclass
import time

import numpy as np

from .geometry import interpolate_rotation, rotation_distance


class SafetyError(RuntimeError):
    pass


@dataclass
class RobotState:
    pose: np.ndarray
    gripper: float
    wrench: np.ndarray
    timestamp: float

    def vector(self):
        return np.r_[self.pose, self.gripper, self.wrench].astype(np.float32)


class SafetyGuard:
    def __init__(self, config, max_age_s=0.5):
        self.config, self.max_age_s = config, max_age_s

    def check_state(self, state):
        if state.pose.shape != (6,) or state.wrench.shape != (6,) or not np.isfinite(state.vector()).all():
            raise SafetyError("invalid measured robot state")
        age = time.monotonic() - state.timestamp
        if age < -0.1 or age > self.max_age_s:
            raise SafetyError("stale robot state")
        if not 0 <= state.gripper <= 1:
            raise SafetyError("measured gripper outside [0,1]")
        if np.linalg.norm(state.wrench[:3]) > self.config.max_force_n:
            raise SafetyError("measured contact force exceeds limit")
        if self.config.workspace_min is not None:
            if np.any(state.pose[:3] < self.config.workspace_min) or np.any(
                state.pose[:3] > self.config.workspace_max
            ):
                raise SafetyError("measured robot is outside configured workspace")

    def project(self, action, state, dt):
        self.check_state(state)
        a = np.asarray(action, dtype=float).copy()
        if a.shape != (7,) or not np.isfinite(a).all() or dt <= 0:
            raise SafetyError("invalid command")
        delta = a[:3] - state.pose[:3]
        norm = np.linalg.norm(delta)
        bound = self.config.max_linear_speed_m_s * dt
        if norm > bound:
            a[:3] = state.pose[:3] + delta * (bound / norm)
        angle = rotation_distance(state.pose[3:6], a[3:6])
        fraction = min(1.0, self.config.max_angular_speed_rad_s * dt / max(angle, 1e-12))
        a[3:6] = interpolate_rotation(state.pose[3:6], a[3:6], fraction)
        if self.config.workspace_min is not None:
            a[:3] = np.clip(a[:3], self.config.workspace_min, self.config.workspace_max)
        a[6] = np.clip(a[6], 0, 1)
        return a.astype(np.float32)
