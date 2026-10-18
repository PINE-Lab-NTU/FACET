"""UR RTDE Cartesian position-control backend with optional Robotiq register socket."""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from ..safety import RobotState, SafetyError


class RobotiqSocket:
    def __init__(self, host, port):
        self.sock = socket.create_connection((host, port), timeout=0.2)
        self.sock.settimeout(0.2)
        self.file = self.sock.makefile("rwb", buffering=0)
        self.lock = threading.Lock()
        if self.get("STA") != 3:
            self.close()
            raise RuntimeError("activate/calibrate Robotiq with its driver before starting")
        self.last = None

    def get(self, register):
        with self.lock:
            self.file.write(f"GET {register}\n".encode())
            parts = self.file.readline(128).decode().split()
        if len(parts) != 2 or parts[0] != register:
            raise RuntimeError("invalid Robotiq register response")
        return int(parts[1])

    def command(self, opening):
        raw = int(round((1 - float(opening)) * 255))
        with self.lock:
            if raw != self.last:
                self.file.write(f"SET POS {raw} SPE 100 FOR 50 GTO 1\n".encode())
                if self.file.readline(128).strip().lower() != b"ack":
                    raise RuntimeError("Robotiq command rejected")
                self.last = raw

    def close(self):
        self.file.close()
        self.sock.close()


class URRTDE:
    def __init__(self, config, *, execute=False):
        import rtde_receive

        self.config = config
        self.rx = rtde_receive.RTDEReceiveInterface(config.address)
        self.tx = self.gripper = None
        self.last_timestamp = self.last_advance = None
        self._gripper_opening, self._gripper_read_at = 1.0, 0.0
        try:
            if config.gripper == "robotiq":
                self.gripper = RobotiqSocket(config.address, config.gripper_port)
            if execute:
                import rtde_control

                self.tx = rtde_control.RTDEControlInterface(config.address, config.ur_control_hz)
        except Exception:
            self.close()
            raise

    def read(self):
        timestamp = self.rx.getTimestamp()
        now = time.monotonic()
        if timestamp != self.last_timestamp:
            self.last_timestamp, self.last_advance = timestamp, now
        if self.last_advance is None or now - self.last_advance > self.config.state_max_age_s:
            raise SafetyError("UR RTDE state timestamp stopped advancing")
        if self.rx.getSafetyMode() != 1:
            raise SafetyError("UR is not in normal safety mode")
        if self.gripper and now - self._gripper_read_at >= 0.05:
            self._gripper_opening = 1 - self.gripper.get("POS") / 255
            self._gripper_read_at = now
        return RobotState(
            np.asarray(self.rx.getActualTCPPose(), float),
            self._gripper_opening,
            np.asarray(self.rx.getActualTCPForce(), float),
            self.last_advance,
        )

    def send(self, action):
        if self.tx is None:
            raise RuntimeError("read-only UR connection")
        ok = self.tx.servoL(action[:6].tolist(), 0.1, 0.1, 1 / self.config.ur_control_hz, 0.1, 300)
        if ok is False:
            raise RuntimeError("UR servo command rejected")
        if self.gripper:
            self.gripper.command(action[6])

    def hold(self):
        if self.tx is not None:
            self.tx.servoStop()

    def close(self):
        if self.gripper:
            self.gripper.close()
            self.gripper = None
        if self.tx:
            try:
                self.tx.servoStop()
            finally:
                self.tx.disconnect()
                self.tx = None
        if self.rx:
            self.rx.disconnect()
