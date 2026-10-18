"""Hardware-free demo, read-only probes and explicit physical execution."""

from __future__ import annotations

import argparse
import json
import time

from .cameras import MockCameras, RealSenseCameras
from .config import Config, load_config
from .policy import MockPolicy, RemotePolicy
from .robots.base import StreamingRobot
from .robots.mock import MockRobot
from .runtime import Session
from .teleop import ConsoleCommands, SpaceMouse


def make_robot(config, *, execute=False):
    if config.robot.backend == "mock":
        return MockRobot(config.safety)
    if config.robot.backend == "ur":
        from .robots.ur import URRTDE

        driver = URRTDE(config.robot, execute=execute)
        hz = config.robot.ur_control_hz
    else:
        from .robots.franka import FrankaHTTP

        driver, hz = FrankaHTTP(config.robot), 50
    if not execute:
        return driver
    return StreamingRobot(
        driver,
        config.safety,
        hz=hz,
        max_age_s=config.robot.state_max_age_s,
        command_timeout_s=max(0.5, 2 / config.runtime.hz),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["demo", "validate", "probe", "run"])
    ap.add_argument("--config")
    ap.add_argument("--resume", default="")
    ap.add_argument("--execute", action="store_true", help="enable physical commands for a non-mock backend")
    ap.add_argument("--output-dir")
    ap.add_argument("--episodes", type=int, default=3, help="mock demo only")
    ap.add_argument("--steps", type=int, default=12, help="mock demo only")
    args = ap.parse_args()
    config = load_config(args.config) if args.config else Config()
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.command == "validate":
        config.validate()
        print(json.dumps(config.to_dict(), indent=2))
        return
    if args.command == "demo":
        config.robot.backend = "mock"
        config.safety.workspace_min = config.safety.workspace_max = None
        config.learner.embedding_dim = 16
        config.learner.hidden_dim = config.learner.projection_dim = 16
        config.learner.batch_size = 8
        config.learner.capacity = 128
        config.learner.intervention_updates_per_episode = 2
        config.learner.updates_per_transition = 1
        config.runtime.mode, config.runtime.checkpoint_every = "rl", 1
        if args.episodes < 1 or args.steps < 1:
            ap.error("demo episodes and steps must be positive")
    if args.command == "probe":
        robot = make_robot(config)
        try:
            print(
                json.dumps(
                    {
                        "backend": config.robot.backend,
                        "native_state": robot.read().vector().tolist(),
                        "robot_commands_sent": 0,
                    },
                    indent=2,
                )
            )
        finally:
            robot.close()
        return
    if config.robot.backend != "mock" and not args.execute:
        ap.error("run with hardware requires --execute; use probe for read-only inspection")
    config.validate(execute=True)
    robot = cameras = teleop = session = None
    try:
        mock = config.robot.backend == "mock"
        policy = (
            MockPolicy(config.learner.embedding_dim)
            if mock
            else RemotePolicy(config.policy, adaptation=config.runtime.mode != "base")
        )
        cameras = MockCameras() if mock else RealSenseCameras(config.cameras)
        robot = make_robot(config, execute=True)
        if config.runtime.teleop == "spacemouse":
            teleop = SpaceMouse(config.runtime.teleop_speed_m_s, config.runtime.teleop_axis_sign)
        session = Session(config, robot, policy, cameras, resume=args.resume, teleop=teleop)
        print(f"Run directory: {session.run}", flush=True)
        if args.command == "demo":
            for i in range(args.episodes):
                session.begin()
                for _ in range(args.steps):
                    session.tick()
                session.finish(i % 2 == 0)
            print(json.dumps(session.learner.last_metrics, indent=2))
            return
        commands = ConsoleCommands()
        print(
            "Paused. Place the robot manually, then type resume and Enter.\n"
            "Commands: resume, success, failure, pause, teleop, policy, stop",
            flush=True,
        )
        while True:
            started = time.monotonic()
            command = commands.poll()
            if command:
                if not session.command(command):
                    break
                print("Paused" if session.paused else "Running", flush=True)
            session.tick()
            time.sleep(max(0, 1 / config.runtime.hz - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("Stopping; incomplete episode remains unlabelled.", flush=True)
    finally:
        if session is not None:
            session.close()
        else:
            if robot:
                robot.close()
            if cameras:
                cameras.close()
            if teleop:
                teleop.close()


if __name__ == "__main__":
    main()
