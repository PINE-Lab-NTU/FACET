import threading
import time

import numpy as np
import pytest

from facet_adaptation.config import SafetyConfig
from facet_adaptation.robots.base import StreamingRobot
from facet_adaptation.safety import RobotState


class Driver:
    def __init__(self):
        self.pose = np.zeros(6)
        self.sent = []
        self.event = threading.Event()

    def read(self):
        return RobotState(self.pose.copy(), 0.5, np.zeros(6), time.monotonic())

    def send(self, action):
        self.sent.append(action.copy())
        if len(self.sent) >= 2:
            self.event.set()

    def hold(self):
        pass

    def close(self):
        pass


def test_returned_command_is_sent_and_repeated_exactly():
    driver = Driver()
    robot = StreamingRobot(driver, SafetyConfig(), hz=50)
    try:
        receipt = robot.command([0.1, 0, 0, 0, 0, 0, 0.5], 0.1)
        assert receipt[0] == pytest.approx(0.001)
        np.testing.assert_array_equal(driver.sent[0], receipt)
        assert driver.event.wait(1)
        for sent in driver.sent:
            np.testing.assert_array_equal(sent, receipt)
    finally:
        robot.close()


def test_drift_stops_instead_of_sending_a_different_action():
    driver = Driver()
    robot = StreamingRobot(driver, SafetyConfig(), hz=50)
    try:
        receipt = robot.command([0.1, 0, 0, 0, 0, 0, 0.5], 0.1)
        driver.pose[0] = -0.1
        deadline = time.monotonic() + 1
        while robot._fault is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert "drift" in robot._fault
        for sent in driver.sent:
            np.testing.assert_array_equal(sent, receipt)
    finally:
        robot.close()
