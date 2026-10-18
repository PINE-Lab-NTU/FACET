"""One robot-agnostic control loop; operator-managed episode starts and placement."""

from __future__ import annotations


from .learner import Learner
from .policy import FrameCodec, WrenchHistory
from .recording import EpisodeRecorder, append_metrics, create_run
from .safety import SafetyGuard


class Session:
    def __init__(self, config, robot, policy, cameras, *, resume="", teleop=None):
        self.config, self.robot, self.policy, self.cameras, self.teleop = (
            config,
            robot,
            policy,
            cameras,
            teleop,
        )
        self.codec = FrameCodec(config.policy)
        self.history = WrenchHistory(hz=config.policy.history_hz)
        self.guard = SafetyGuard(config.safety, config.robot.state_max_age_s)
        self.execution_dt = min(1 / config.runtime.hz, getattr(robot, "execution_dt", 1 / config.runtime.hz))
        self.learner = Learner(
            config.learner,
            mode=config.runtime.mode,
            safety=config.safety,
            hz=1 / self.execution_dt,
            position_sign=config.policy.position_sign,
        )
        self.contract = {
            "policy": policy.contract,
            "robot_backend": config.robot.backend,
            "position_sign": config.policy.position_sign,
            "reference_rotvec": config.policy.reference_rotvec,
            "state": "xyz_rotvec_gripper_open_wrench13",
            "control_hz": config.runtime.hz,
            "execution_dt": self.execution_dt,
            "action_semantics": "synchronously_sent_absolute_endpoint_v1",
            "teleop_axis_sign": config.runtime.teleop_axis_sign,
        }
        from dataclasses import asdict

        self.contract["safety"] = asdict(config.safety)
        self.contract["history_hz"] = config.policy.history_hz
        if resume:
            self.learner.load(resume, self.contract)
        if config.runtime.mode == "eval" and not resume:
            raise ValueError("frozen Actor evaluation requires --resume")
        self.run = create_run(config.runtime.output_dir, config, self.contract, resume)
        self.index = 0
        self.recorder = None
        self.steps, self.pending = [], None
        self.actions = self.token = None
        self.chunk_step, self.tick_count = 0, 0
        self.paused, self.manual = True, False

    def begin(self):
        if not self.paused:
            raise RuntimeError("episode already running")
        self.guard.check_state(self.robot.read())
        reset = getattr(self.policy, "reset", None)
        if reset:
            reset()
        self.index += 1
        self.recorder = EpisodeRecorder(self.run, self.index, record_images=self.config.runtime.record_images)
        self.history.reset()
        self.actions = self.token = None
        self.steps, self.pending, self.chunk_step, self.tick_count = [], None, 0, 0
        self.paused = False

    def _observation(self):
        state = self.robot.read()
        self.guard.check_state(state)
        self.history.append(state.wrench, state.timestamp)
        images = self.cameras.capture()
        proprio = self.codec.state(state)
        obs = {
            "observation/state": proprio,
            "observation/wrench_history": self.history.window(),
            "observation/image": images["view1"],
            "observation/wrist_image": images["hand"],
            "observation/right_wrist_image": images["view2"],
            "prompt": self.config.policy.prompt,
        }
        return state, proprio, images, obs

    def tick(self):
        if self.paused:
            return
        state, proprio, images, obs = self._observation()
        if self.actions is None or self.chunk_step >= self.config.policy.execute_steps:
            self.robot.hold()
            self.actions, self.token = self.policy.infer(obs)
            self.chunk_step = 0
        reference = self.actions[self.chunk_step].copy()
        state = self.robot.read()
        self.guard.check_state(state)
        proprio = self.codec.state(state)
        if self.config.runtime.mode == "base":
            x = proprio.copy()
        else:
            x = self.learner.state_vector(self.token, proprio)
        if self.pending is not None:
            self.pending.update(next_x=x.copy(), next_reference=reference.copy())
            self.steps.append(self.pending)
        if self.manual:
            if self.teleop is None:
                raise RuntimeError("teleop requested but no teleop source configured")
            proposal = self.codec.policy_action(self.teleop.action(state, 1 / self.config.runtime.hz))
        else:
            proposal = self.learner.act(x, reference)

        current = self.robot.read()
        self.guard.check_state(current)
        native = self.guard.project(self.codec.native_action(proposal), current, self.execution_dt)
        executed = self.robot.command(native, self.execution_dt)
        action = self.codec.policy_action(executed)
        self.pending = dict(x=x.copy(), reference=reference.copy(), action=action.copy(), human=self.manual)
        self.recorder.append(
            state=proprio,
            reference=reference,
            action=action,
            requested_command=proposal,
            x=x,
            human=self.manual,
            images=images,
            wrench_history=self.history.window(),
            execution_dt=self.execution_dt,
        )
        self.chunk_step += 1
        self.tick_count += 1
        if self.tick_count >= self.config.runtime.max_episode_steps:
            self.finish(None, "episode time limit; operator outcome not supplied")

    def finish(self, success, reason="operator"):
        if self.paused:
            return
        self.robot.hold()
        self.paused = True
        if self.pending is not None:
            try:
                terminal_state = self.robot.read()
                self.guard.check_state(terminal_state)
                terminal = self.codec.state(terminal_state)
            except Exception:
                self.recorder.finalize(None, "terminal state unavailable; no reward assigned")
                self.pending = None
                raise
            x = (
                terminal
                if self.config.runtime.mode == "base"
                else self.learner.state_vector(self.token, terminal)
            )
            self.pending.update(next_x=x, next_reference=self.pending["reference"].copy())
            self.steps.append(self.pending)
            self.pending = None
        self.recorder.finalize(success, reason)
        if success is not None:
            metrics = self.learner.finish_episode(self.steps, success=bool(success))
            append_metrics(self.run, metrics)
        if (
            self.config.runtime.mode in {"bc", "rl"}
            and self.index % self.config.runtime.checkpoint_every == 0
        ):
            self.save()
        self.actions = self.token = None
        self.manual = False

    def save(self):
        if self.config.runtime.mode in {"bc", "rl"}:
            self.learner.save(self.run / f"checkpoint_ep{self.index:06d}.pt", self.contract)

    def command(self, command):
        if command == "resume":
            self.begin()
        elif command in {"success", "failure"}:
            self.finish(command == "success")
        elif command == "pause":
            self.finish(None, "operator pause; unlabelled")
        elif command == "teleop":
            if self.config.runtime.mode == "eval":
                raise ValueError("teleoperation is disabled during actor evaluation")
            if self.teleop is None:
                raise ValueError("no teleop device configured")
            self.manual = True
        elif command == "confirm":
            for row in reversed(self.steps + ([self.pending] if self.pending else [])):
                if not row["human"]:
                    break
                row["bc_confirmed"] = True
        elif command == "policy":
            self.manual = False
        elif command != "stop":
            raise ValueError("commands: resume, success, failure, pause, teleop, policy, stop")
        return command != "stop"

    def close(self):
        try:
            if not self.paused:
                try:
                    self.finish(None, "session closed; unlabelled")
                except Exception as exc:
                    self.paused = True
                    if self.recorder:
                        self.recorder.finalize(None, f"aborted: {type(exc).__name__}: {exc}")
            self.save()
        finally:
            errors = []
            for resource in (self.robot, self.cameras, self.teleop):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        errors.append(exc)
            if errors:
                raise errors[0]
