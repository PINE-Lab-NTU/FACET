"""Small native-frame robot protocol and guarded continuous target streaming."""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np

from ..safety import RobotState, SafetyError, SafetyGuard


class Robot(Protocol):
    def read(self) -> RobotState: ...
    def command(self, action, dt): ...
    def hold(self): ...
    def close(self): ...


class StreamingRobot:
    def __init__(self, driver, safety, *, hz=50, max_age_s=0.5, command_timeout_s=0.5):
        self.driver = driver
        self.guard = SafetyGuard(safety, max_age_s)
        self.period = 1 / hz
        self.execution_dt = self.period
        self.command_timeout_s = command_timeout_s
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._target = None
        self._deadline = 0.0
        self._fault = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def read(self):
        if self._fault:
            raise SafetyError(self._fault)
        return self.driver.read()

    def command(self, action, dt):
        with self._lock:
            if self._fault:
                raise SafetyError(self._fault)
            target = self.guard.project(action, self.driver.read(), min(dt, self.execution_dt))
            try:
                self.driver.send(target)
            except Exception as exc:
                self._latch_fault(exc)
                raise SafetyError(self._fault) from exc
            self._target = target
            self._deadline = time.monotonic() + self.command_timeout_s
            return target.copy()

    def _latch_fault(self, exc):
        self._fault = f"{type(exc).__name__}: {exc}"
        self._target = None
        try:
            self.driver.hold()
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            started = time.monotonic()
            with self._lock:
                if self._target is not None and not self._fault:
                    try:
                        if started > self._deadline:
                            raise SafetyError("command heartbeat expired")
                        state = self.driver.read()
                        self.guard.check_state(state)

                        projected = self.guard.project(self._target, state, self.period)
                        if not np.allclose(projected, self._target, atol=1e-7, rtol=0):
                            raise SafetyError("measured-state drift would change the recorded command")
                        # Repeat the already sent endpoint; never introduce an unrecorded action.
                        self.driver.send(self._target.copy())
                    except Exception as exc:
                        self._latch_fault(exc)
            self._stop.wait(max(0, self.period - (time.monotonic() - started)))

    def hold(self):
        with self._lock:
            self._target = None
            self.driver.hold()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        try:
            self.hold()
        finally:
            self.driver.close()
