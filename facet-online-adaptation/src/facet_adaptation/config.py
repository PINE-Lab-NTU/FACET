"""Training and runtime configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math

import yaml


@dataclass
class LearnerConfig:
    embedding_dim: int = 1024
    hidden_dim: int = 256
    projection_dim: int = 256
    hidden_layers: int = 2
    actor_mode: str = "absolute"
    reference_dropout: float = 0.5
    action_min: list[float] = field(default_factory=lambda: [-1.0, -1.0, 0.0, -3.14, -3.14, -3.14, 0.0])
    action_max: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 1.0])
    action_mask: list[bool] = field(default_factory=lambda: [True] * 7)
    proprio_mean: list[float] = field(default_factory=lambda: [0.0] * 13)
    proprio_std: list[float] = field(default_factory=lambda: [1.0] * 13)
    wrench_prediction_weight: float = 0.1
    actor_lr: float = 1e-6
    bc_lr: float = 5e-4
    critic_lr: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    bc_beta: float = 0.5 / 7
    critic_target: str = "td"
    td3_bc_target: str = "demo"
    demo_success_only: bool = True
    demo_seed_success_episodes: int = 0
    batch_size: int = 256
    capacity: int = 10000
    intervention_capacity: int = 10000
    updates_per_transition: int = 5
    critic_updates_per_actor: int = 20
    intervention_updates_per_episode: int = 200
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5
    q_abort_threshold: float = 2.0
    q_abort_consecutive: int = 20
    device: str = "cpu"
    seed: int = 42

    def validate(self):
        for key in (
            "embedding_dim",
            "hidden_dim",
            "projection_dim",
            "hidden_layers",
            "capacity",
            "intervention_capacity",
            "batch_size",
            "critic_updates_per_actor",
            "q_abort_consecutive",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"learner.{key} must be a positive integer")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch_size must be even for the 50/50 online/demo batch")
        if self.capacity < self.batch_size:
            raise ValueError("capacity must be at least batch_size")
        for name in (
            "intervention_updates_per_episode",
            "updates_per_transition",
            "demo_seed_success_episodes",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"learner.{name} must be a nonnegative integer")
        if self.actor_mode != "absolute":
            raise ValueError("actor_mode must be absolute")
        if self.critic_target not in {"td", "mc", "td_mc_max"}:
            raise ValueError("invalid critic_target")
        if self.td3_bc_target not in {"reference", "demo"}:
            raise ValueError("invalid td3_bc_target")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1 or not 0 <= self.reference_dropout < 1:
            raise ValueError("invalid gamma, tau or reference_dropout")
        if (
            len(self.action_min) != 7
            or len(self.action_max) != 7
            or len(self.action_mask) != 7
            or any(a >= b for a, b in zip(self.action_min, self.action_max))
            or not any(self.action_mask)
        ):
            raise ValueError("invalid seven-dimensional action box or mask")
        if len(self.proprio_mean) != 13 or len(self.proprio_std) != 13 or min(self.proprio_std) <= 0:
            raise ValueError("invalid proprioception normalization")
        if self.wrench_prediction_weight < 0:
            raise ValueError("invalid wrench prediction configuration")
        for name in ("actor_lr", "bc_lr", "critic_lr", "bc_beta"):
            if getattr(self, name) <= 0:
                raise ValueError(f"learner.{name} must be positive")
        if min(self.target_noise_std, self.target_noise_clip) < 0 or self.q_abort_threshold <= 1:
            raise ValueError("invalid target-noise or Q guard settings")
        _finite(asdict(self))


@dataclass
class RobotConfig:
    backend: str = "mock"
    address: str = ""
    http_timeout_s: float = 0.2
    state_max_age_s: float = 0.5
    gripper: str = "none"
    gripper_port: int = 63352
    gripper_max_open_m: float = 0.085
    ur_control_hz: int = 500


@dataclass
class SafetyConfig:
    workspace_min: list[float] | None = None
    workspace_max: list[float] | None = None
    max_linear_speed_m_s: float = 0.05
    max_angular_speed_rad_s: float = 0.3
    max_force_n: float = 40.0


@dataclass
class PolicyConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = ""
    checkpoint_id: str = ""
    token_id: str = ""
    execute_steps: int = 1
    max_latency_s: float = 1.0

    position_sign: list[int] = field(default_factory=lambda: [1, 1, 1])
    reference_rotvec: list[float] | None = None
    history_hz: float = 15.0


@dataclass
class CameraConfig:
    serials: dict[str, str] = field(default_factory=dict)
    width: int = 640
    height: int = 480
    fps: int = 30

    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    mode: str = "bc"
    hz: float = 10.0
    max_episode_steps: int = 1000
    output_dir: str = "runs"
    checkpoint_every: int = 20
    teleop: str = "none"
    teleop_speed_m_s: float = 0.03
    teleop_axis_sign: list[int] = field(default_factory=lambda: [1, 1, 1])
    record_images: bool = True


@dataclass
class Config:
    learner: LearnerConfig = field(default_factory=LearnerConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self, *, execute=False):
        self.learner.validate()
        if self.robot.backend not in {"mock", "ur", "franka"}:
            raise ValueError("robot.backend must be mock, ur or franka")
        if self.runtime.mode not in {"base", "bc", "rl", "eval"}:
            raise ValueError("runtime.mode must be base, bc, rl or eval")
        if self.runtime.teleop not in {"none", "spacemouse"}:
            raise ValueError("runtime.teleop must be none or spacemouse")
        if not 0 < self.runtime.hz <= 100 or self.runtime.max_episode_steps < 1:
            raise ValueError("invalid runtime cadence or episode length")
        if self.runtime.checkpoint_every < 1 or self.policy.execute_steps < 1:
            raise ValueError("invalid checkpoint interval or execute_steps")
        if len(self.policy.position_sign) != 3 or any(v not in {-1, 1} for v in self.policy.position_sign):
            raise ValueError("policy.position_sign must be three signs")
        if len(self.runtime.teleop_axis_sign) != 3 or any(
            v not in {-1, 1} for v in self.runtime.teleop_axis_sign
        ):
            raise ValueError("runtime.teleop_axis_sign must be three signs")
        if self.policy.reference_rotvec is not None and len(self.policy.reference_rotvec) != 3:
            raise ValueError("reference_rotvec must be three numbers")
        if self.robot.gripper not in {"none", "robotiq"}:
            raise ValueError("supported grippers: none, robotiq")
        for obj, names in [
            (self.safety, ("max_linear_speed_m_s", "max_angular_speed_rad_s", "max_force_n")),
            (self.policy, ("max_latency_s",)),
            (self.robot, ("state_max_age_s", "http_timeout_s", "gripper_max_open_m")),
            (self.runtime, ("teleop_speed_m_s",)),
        ]:
            if any(getattr(obj, n) <= 0 for n in names):
                raise ValueError(f"positive limits required: {names}")
        lo, hi = self.safety.workspace_min, self.safety.workspace_max
        if (lo is None) != (hi is None):
            raise ValueError("set both workspace bounds")
        if lo is not None and (len(lo) != 3 or len(hi) != 3 or any(a >= b for a, b in zip(lo, hi))):
            raise ValueError("workspace bounds must be ordered 3-vectors")
        if execute and self.robot.backend != "mock":
            if not self.robot.address or not self.policy.prompt or not self.policy.checkpoint_id:
                raise ValueError("hardware requires robot address, task prompt and frozen checkpoint_id")
            if self.runtime.mode != "base" and not self.policy.token_id:
                raise ValueError("adaptation requires a frozen token_id")
            if lo is None:
                raise ValueError("set native-frame workspace bounds before hardware execution")
        _finite(asdict(self))

    def to_dict(self):
        return asdict(self)


def _finite(value):
    if isinstance(value, dict):
        for v in value.values():
            _finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _finite(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("configuration contains non-finite number")


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text()) or {}
    allowed = {f.name for f in fields(Config)}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("unknown top-level config fields")
    mapping = {
        "learner": LearnerConfig,
        "robot": RobotConfig,
        "safety": SafetyConfig,
        "policy": PolicyConfig,
        "cameras": CameraConfig,
        "runtime": RuntimeConfig,
    }
    config = Config(**{k: mapping[k](**v) for k, v in data.items()})
    config.validate()
    return config
