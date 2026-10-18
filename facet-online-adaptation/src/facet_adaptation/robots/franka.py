"""FR3/Robotiq HTTP client. Requires an existing external compliant controller."""

from __future__ import annotations

import json
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from ..geometry import quat_to_rotvec, rotvec_to_quat
from ..safety import RobotState, SafetyError


class FrankaHTTP:
    def __init__(self, config):
        parsed = urlparse(config.address)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Franka address must be an HTTP robot-server URL")
        self.config = config
        self._last_gripper = None

    def _post(self, endpoint, payload=None, *, expect_json=False):
        req = Request(
            self.config.address.rstrip("/") + endpoint,
            data=json.dumps(payload or {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=self.config.http_timeout_s) as response:
            raw = response.read().decode()
        return json.loads(raw) if expect_json else raw

    def read(self):
        d = self._post("/getstate", expect_json=True)
        age = float(d.get("state_age_sec", 0))
        if (
            not d.get("state_fresh", False)
            or not np.isfinite(age)
            or age < 0
            or age > self.config.state_max_age_s
        ):
            raise SafetyError("Franka server reports stale state")
        pose = np.asarray(d["pose"], float)
        if pose.shape != (7,):
            raise SafetyError("invalid Franka pose")
        return RobotState(
            np.r_[pose[:3], quat_to_rotvec(pose[3:])],
            float(d["gripper_pos"]),
            np.r_[d["force"], d["torque"]],
            time.monotonic() - age,
        )

    def send(self, action):
        self._post("/pose", {"arr": np.r_[action[:3], rotvec_to_quat(action[3:6])].tolist()})
        if self.config.gripper == "robotiq":
            value = int(round(float(action[6]) * 255))
            if value != self._last_gripper:
                self._post("/move_gripper", {"gripper_pos": value})
                self._last_gripper = value

    def hold(self):

        state = self.read()
        self._post("/pose", {"arr": np.r_[state.pose[:3], rotvec_to_quat(state.pose[3:])].tolist()})

    def close(self):
        pass
