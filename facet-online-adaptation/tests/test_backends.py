import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from facet_adaptation.config import RobotConfig, SafetyConfig
from facet_adaptation.robots.base import StreamingRobot
from facet_adaptation.robots.franka import FrankaHTTP
from facet_adaptation.robots.ur import URRTDE
from facet_adaptation.safety import SafetyError


@pytest.fixture
def http_robot():
    state = {
        "pose": [0.4, 0, 0.3, 0, 0, 0, 1],
        "gripper_pos": 0.3,
        "force": [0, 0, 0],
        "torque": [0, 0, 0],
        "state_fresh": True,
        "state_age_sec": 0.001,
    }
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            self.send_response(200)
            self.end_headers()
            if self.path == "/getstate":
                self.wfile.write(json.dumps(state).encode())
            elif self.path == "/pose":
                state["pose"] = body["arr"]
                self.wfile.write(b"OK")
            elif self.path == "/move_gripper":
                state["gripper_pos"] = body["gripper_pos"] / 255
                self.wfile.write(b"OK")

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", state, calls
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_franka_read_only_probe_and_gripper_polarity(http_robot):
    url, state, calls = http_robot
    robot = FrankaHTTP(RobotConfig(address=url, gripper="robotiq"))
    assert robot.read().gripper == 0.3
    robot.close()
    assert all(path == "/getstate" for path, _ in calls)
    robot.send(np.array([0.4, 0, 0.3, 0, 0, 0, 1]))
    assert state["gripper_pos"] == 1
    robot.send(np.array([0.4, 0, 0.3, 0, 0, 0, 0]))
    assert state["gripper_pos"] == 0
    count = sum(path == "/move_gripper" for path, _ in calls)
    robot.hold()
    assert sum(path == "/move_gripper" for path, _ in calls) == count


def test_franka_stale_and_force_faults_latch(http_robot):
    url, state, calls = http_robot
    driver = FrankaHTTP(RobotConfig(address=url))
    robot = StreamingRobot(driver, SafetyConfig(), hz=50)
    assert not any(path == "/pose" for path, _ in calls)
    robot.command([0.401, 0, 0.3, 0, 0, 0, 0.3], 0.1)
    state["force"] = [100, 0, 0]
    deadline = time.monotonic() + 1
    while robot._fault is None and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(SafetyError):
        robot.command([0.4, 0, 0.3, 0, 0, 0, 1], 0.1)
    robot.close()
    state["state_fresh"] = False
    with pytest.raises(SafetyError):
        driver.read()


def test_ur_probe_never_constructs_control_interface(monkeypatch):
    control_calls = []
    rx = SimpleNamespace(
        getTimestamp=lambda: time.monotonic(),
        getSafetyMode=lambda: 1,
        getActualTCPPose=lambda: [0] * 6,
        getActualTCPForce=lambda: [0] * 6,
        disconnect=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=lambda _: rx))
    monkeypatch.setitem(
        sys.modules,
        "rtde_control",
        SimpleNamespace(RTDEControlInterface=lambda *args: control_calls.append(args)),
    )
    probe = URRTDE(RobotConfig(address="mock-driver"), execute=False)
    np.testing.assert_array_equal(probe.read().pose, 0)
    probe.close()
    assert control_calls == []


def test_ur_commands_require_explicit_control_and_stop_does_not_reset(monkeypatch):
    commands, stops = [], []
    rx = SimpleNamespace(
        getTimestamp=lambda: time.monotonic(),
        getSafetyMode=lambda: 1,
        getActualTCPPose=lambda: [0] * 6,
        getActualTCPForce=lambda: [0] * 6,
        disconnect=lambda: None,
    )
    tx = SimpleNamespace(
        servoL=lambda *args: commands.append(args) or True,
        servoStop=lambda: stops.append(True),
        disconnect=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "rtde_receive", SimpleNamespace(RTDEReceiveInterface=lambda _: rx))
    monkeypatch.setitem(sys.modules, "rtde_control", SimpleNamespace(RTDEControlInterface=lambda *args: tx))
    robot = URRTDE(RobotConfig(address="mock-driver"), execute=True)
    assert not commands
    robot.send(np.array([0.001, 0, 0, 0, 0, 0, 1]))
    assert len(commands) == 1 and commands[0][0][0] == 0.001
    robot.hold()
    robot.close()
    assert len(commands) == 1 and stops


def test_command_heartbeat_expiry_stops_streaming(http_robot):
    url, _, calls = http_robot
    driver = FrankaHTTP(RobotConfig(address=url))
    robot = StreamingRobot(driver, SafetyConfig(), hz=100, command_timeout_s=0.03)
    try:
        robot.command([0.401, 0, 0.3, 0, 0, 0, 1], 0.1)
        deadline = time.monotonic() + 1
        while not robot._fault and time.monotonic() < deadline:
            time.sleep(0.005)
        assert "heartbeat" in robot._fault
        before = len(calls)
        time.sleep(0.03)
        assert len(calls) == before
    finally:
        robot.close()
