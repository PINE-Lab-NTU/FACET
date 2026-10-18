"""Teleop."""

import queue
import sys
import threading
import time

import numpy as np


class ConsoleCommands:
    def __init__(self):
        self.queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            self.queue.put(line.strip().lower())
        self.queue.put("stop")

    def poll(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None


class SpaceMouse:
    def __init__(self, speed, signs, angular_speed=0.3):
        import pyspacemouse

        self.device, self.speed, self.signs = pyspacemouse, speed, np.asarray(signs)
        self.angular_speed = angular_speed
        if not self.device.open():
            raise RuntimeError("SpaceMouse unavailable")
        self._latest, self._timestamp = None, 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while not self._stop.is_set():
            value = self.device.read()
            if value is not None:
                self._latest, self._timestamp = value, time.monotonic()
            self._stop.wait(0.005)

    def action(self, state, dt):
        a = np.r_[state.pose, state.gripper].astype(np.float32)
        value = self._latest
        if value is None or time.monotonic() - self._timestamp > 0.2:
            return a
        motion = np.array([value.x, value.y, value.z])
        motion[np.abs(motion) < 0.08] = 0
        a[:3] += motion * self.signs * self.speed * dt
        from .geometry import rotvec_to_quat, quat_to_rotvec

        rotation = np.asarray([value.roll, value.pitch, value.yaw], float)
        rotation[np.abs(rotation) < 0.08] = 0
        q, r = rotvec_to_quat(a[3:6]), rotvec_to_quat(rotation * self.angular_speed * dt)
        vector = q[3] * r[:3] + r[3] * q[:3] + np.cross(q[:3], r[:3])
        scalar = q[3] * r[3] - np.dot(q[:3], r[:3])
        a[3:6] = quat_to_rotvec(np.r_[vector, scalar])

        if len(value.buttons) > 0 and value.buttons[0]:
            a[6] = 1
        if len(value.buttons) > 1 and value.buttons[1]:
            a[6] = 0
        return a

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)
        self.device.close()
