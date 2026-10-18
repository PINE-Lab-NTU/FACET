import time

import numpy as np
import pytest

from facet_adaptation.config import SafetyConfig, load_config
from facet_adaptation.geometry import rotation_distance
from facet_adaptation.safety import RobotState, SafetyError, SafetyGuard


def state():
    return RobotState(np.zeros(6), 1, np.zeros(6), time.monotonic())


def test_speed_limits_and_rotation_pi_equivalence():
    guard = SafetyGuard(SafetyConfig())
    a = guard.project([10, 10, 10, 0, 0, np.pi, 1], state(), 0.1)
    assert np.linalg.norm(a[:3]) <= 0.005001
    assert rotation_distance(np.zeros(3), a[3:6]) <= 0.030001
    assert rotation_distance([0, 0, np.pi], [0, 0, -np.pi]) < 1e-7


@pytest.mark.parametrize("kind", ["stale", "force", "nan", "workspace"])
def test_bad_state_rejected(kind):
    s = state()
    cfg = SafetyConfig(workspace_min=[-0.1] * 3, workspace_max=[0.1] * 3)
    if kind == "stale":
        s.timestamp -= 2
    elif kind == "force":
        s.wrench[2] = 100
    elif kind == "nan":
        s.pose[0] = np.nan
    else:
        s.pose[0] = 1
    with pytest.raises(SafetyError):
        SafetyGuard(cfg).project(np.ones(7), s, 0.1)


def test_unknown_task_specific_configuration_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("robot:\n  backend: mock\n  fixed_grasp_pose: [1,2,3]\n")
    with pytest.raises(TypeError):
        load_config(p)


def test_hardware_example_has_no_machine_workspace():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("ur.example.yaml", "franka.example.yaml"):
        cfg = load_config(root / "configs" / name)
        with pytest.raises(ValueError):
            cfg.validate(execute=True)
