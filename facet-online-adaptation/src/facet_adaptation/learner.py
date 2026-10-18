"""Train and apply bounded local policies."""

from __future__ import annotations

from dataclasses import asdict
import copy
import time
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from .config import LearnerConfig, SafetyConfig
from .safety import SafetyGuard, RobotState
from .models import Actor, TwinQCritic
from .replay import ReplayBuffer


def discounted_returns(rewards, gamma):
    result = np.empty(len(rewards), np.float32)
    value = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        value = float(rewards[i]) + gamma * value
        result[i] = value
    return result


def critic_targets(td, mc, mode):

    if mode == "td":
        return td
    if mode not in {"mc", "td_mc_max"}:
        raise ValueError("unknown critic target")
    known = torch.isfinite(mc)
    safe = torch.where(known, mc, torch.zeros_like(mc))
    target = safe if mode == "mc" else torch.maximum(td, safe)
    return torch.where(known, target, td)


def actor_loss_components(q, proposed, target, beta, weights=None):

    per_row = (proposed - target).square().sum(-1)
    bc = per_row.mean() if weights is None else (per_row * weights).sum() / weights.sum().clamp_min(1)
    policy = -q.mean()
    return policy + beta * bc, policy, bc


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class Learner:
    def __init__(self, config: LearnerConfig, *, mode="bc", safety=None, hz=10.0, position_sign=None):
        config.validate()
        if mode not in {"base", "bc", "rl", "eval"}:
            raise ValueError("invalid learner mode")
        self.config, self.mode = config, mode
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.actor = Actor(config).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.safety = safety or SafetyConfig()
        self.control_dt = 1.0 / hz
        self.position_sign = np.asarray(position_sign or [1, 1, 1])
        self.last_wrench = None
        self.critic = TwinQCritic(config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.capacity, config.embedding_dim + 13, 7, 1)
        self.demo = ReplayBuffer(config.capacity, config.embedding_dim + 13, 7, 1)
        self.interventions = []
        self.updates = self.bc_updates = self.episodes = self.demo_successes = 0
        self.mean = (np.asarray(config.action_min, np.float32) + config.action_max) / 2
        self.std = (np.asarray(config.action_max, np.float32) - config.action_min) / 2
        self.fused, self.guard_count = False, 0
        self.last_metrics = {}

    @property
    def actor_lr(self):
        return self.config.bc_lr if self.mode == "bc" else self.config.actor_lr

    def state_vector(self, token, proprio):
        token = np.asarray(token, np.float32)
        proprio = np.asarray(proprio, np.float32)
        if token.shape != (self.config.embedding_dim,) or proprio.shape != (13,):
            raise ValueError("invalid RL token or proprio dimension")
        if not np.isfinite(token).all() or not np.any(token) or not np.isfinite(proprio).all():
            raise ValueError("RL state requires a finite nonzero token and finite proprioception")
        return np.concatenate([token, proprio])

    def project(self, proposed, reference):
        if isinstance(proposed, torch.Tensor):
            lo, hi = self.tensor(self.config.action_min), self.tensor(self.config.action_max)
            bounded = torch.maximum(torch.minimum(proposed, hi), lo)
            return torch.where(torch.tensor(self.config.action_mask, device=self.device), bounded, reference)
        if not np.isfinite(proposed).all() or not np.isfinite(reference).all():
            raise FloatingPointError("non-finite actor command")
        return np.where(
            self.config.action_mask,
            np.clip(proposed, self.config.action_min, self.config.action_max),
            reference,
        ).astype(np.float32)

    def applied(self, proposed, x):
        tensor = isinstance(proposed, torch.Tensor)
        values = proposed.detach().cpu().numpy() if tensor else np.asarray(proposed)
        states = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        one = values.ndim == 1
        values, states = np.atleast_2d(values), np.atleast_2d(states)
        out = []
        guard = SafetyGuard(self.safety)
        for command, state in zip(values, states):
            p = state[-13:]
            pose = p[:6].copy()
            pose[:3] *= self.position_sign
            native = command.copy()
            native[:3] *= self.position_sign
            measurement = RobotState(pose, float(p[6]), np.zeros(6), time.monotonic())
            result = guard.project(native, measurement, self.control_dt)
            result[:3] *= self.position_sign
            out.append(result)
        result = np.stack(out)
        if one:
            result = result[0]
        if tensor:
            return self.tensor(result) + (proposed - proposed.detach())
        return result

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, x, reference):
        reference = np.asarray(reference, np.float32)
        if self.mode == "base" or self.fused:
            return reference.copy()
        self.actor.eval()
        result, _ = self.actor.outputs(
            self.tensor(x)[None], self.tensor(reference)[None], reference_dropout=False
        )
        applied = self.applied(self.project(result[0].cpu().numpy(), reference), x)
        _, wrench = self.actor.outputs(
            self.tensor(x)[None], self.tensor(reference)[None], reference_dropout=False,
            executed_action=self.tensor(applied)[None],
        )
        self.last_wrench = (
            wrench[0].cpu().numpy() * np.asarray(self.config.proprio_std)[7:]
            + np.asarray(self.config.proprio_mean)[7:]
        ).astype(np.float32)
        return applied

    def normalize(self, value):
        return (value - self.tensor(self.mean)) / self.tensor(self.std)

    def denormalize(self, value):
        return value * self.tensor(self.std) + self.tensor(self.mean)

    def finish_episode(self, steps, *, success: bool):
        if self.mode in {"eval", "base"}:
            return {"success": float(success), "steps": len(steps), "updates": 0}
        if self.fused:
            raise RuntimeError("learner fuse is latched; inspect the run before resuming")
        if not steps:
            return {"steps": 0, "updates": 0}
        rewards = np.zeros(len(steps), np.float32)
        if any("reward" in row for row in steps):
            if not all("reward" in row for row in steps):
                raise ValueError("reward coverage must include every transition")
            rewards = np.asarray([row["reward"] for row in steps], np.float32)
            if not np.isfinite(rewards).all():
                raise ValueError("nonfinite transition reward")
        else:
            rewards[-1] = float(success)
        returns = discounted_returns(rewards, self.config.gamma)
        seed_demo = success and self.demo_successes < self.config.demo_seed_success_episodes

        for row in steps:
            for key, width in (
                ("x", self.config.embedding_dim + 13),
                ("next_x", self.config.embedding_dim + 13),
                ("reference", 7),
                ("next_reference", 7),
                ("action", 7),
            ):
                value = np.asarray(row[key])
                if value.shape != (width,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid executed transition {key}")
        for i, row in enumerate(steps):
            human = bool(row["human"])
            reference = row["reference"]
            item = dict(
                x=row["x"],
                a=row["action"],
                a_tilde=reference,
                next_a_tilde=row["next_reference"],
                rewards=np.array([rewards[i]]),
                next_x=row["next_x"],
                done=float(i == len(steps) - 1),
                mc_return=returns[i],
            )
            self.replay.add(**item)
            active = np.asarray(self.config.action_mask)
            inside = np.all((np.asarray(row["action"]) >= self.config.action_min) | ~active) and np.all(
                (np.asarray(row["action"]) <= self.config.action_max) | ~active
            )
            if human and bool(row.get("bc_confirmed", success)) and inside:
                self.demo.add(**item)
            if human and bool(row.get("bc_confirmed", success)) and inside:
                self.interventions.append((row["x"].copy(), row["reference"].copy(), row["action"].copy()))
        self.interventions = self.interventions[-self.config.intervention_capacity :]
        self.demo_successes += int(seed_demo)
        self.episodes += 1
        metrics = {}
        if self.interventions:
            for _ in range(self.config.intervention_updates_per_episode):
                metrics.update(self.update_bc())
        if self.mode == "rl" and self.replay.size >= self.config.batch_size:
            for _ in range(self.config.updates_per_transition * len(steps)):
                metrics.update(self.update_td3())
                if self.fused:
                    break
        self.last_metrics = {
            **metrics,
            "episodes": self.episodes,
            "replay_size": self.replay.size,
            "demo_size": self.demo.size,
            "updates": self.updates,
            "bc_updates": self.bc_updates,
            "success": float(success),
            "fused": self.fused,
        }
        return self.last_metrics

    def _optimizer_step(self, optimizer, model, loss):
        if not torch.isfinite(loss).all():
            self.fused = True
            raise FloatingPointError("non-finite learning loss; actor disabled")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=False)
        if not torch.isfinite(norm):
            self.fused = True
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite gradient; actor disabled")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            self.fused = True
            raise FloatingPointError("non-finite parameters; actor disabled")

    def update_bc(self):
        if self.mode not in {"bc", "rl"} or self.fused:
            raise RuntimeError("BC update forbidden in this mode/state")
        ids = np.random.randint(len(self.interventions), size=self.config.batch_size)
        x, ref, target = (self.tensor(np.stack([self.interventions[i][j] for i in ids])) for j in range(3))
        self.actor.train()
        proposed = self.project(self.actor(x, ref), ref)
        mask = self.tensor(self.config.action_mask)
        loss = ((self.normalize(proposed) - self.normalize(target)).square() * mask).sum(
            -1
        ).mean() / mask.sum().clamp_min(1)
        self._optimizer_step(self.actor_optimizer, self.actor, loss)
        self._update_actor_target()
        self.bc_updates += 1
        return {"intervention_bc_loss": float(loss.detach())}

    def update_td3(self):
        if self.mode != "rl" or self.fused:
            raise RuntimeError("TD3 update forbidden in this mode/state")
        half = self.config.batch_size // 2
        online = self.replay.sample(half, str(self.device), include_mc_returns=True)
        has_demo = self.demo.size > 0
        demo = (self.demo if has_demo else self.replay).sample(
            half, str(self.device), include_mc_returns=True
        )
        b = {k: torch.cat([online[k], demo[k]]) for k in online}
        ref, x = b["a_tilde"], b["x"]
        self.actor.eval()
        with torch.no_grad():
            nxt = self.project(
                self.target_actor(b["next_x"], b["next_a_tilde"], reference_dropout=False), b["next_a_tilde"]
            )
            noise = (torch.randn_like(nxt) * self.config.target_noise_std).clamp(
                -self.config.target_noise_clip, self.config.target_noise_clip
            )
            noise *= self.tensor(self.config.action_mask)
            nxt = self.project(self.denormalize(self.normalize(nxt) + noise), b["next_a_tilde"])
            nxt = self.applied(nxt, b["next_x"])
            q = self.critic.target_q_min(b["next_x"], self.normalize(nxt), b["next_a_tilde"])
            td = b["rewards"] + self.config.gamma * (1 - b["dones"]) * q
            target = critic_targets(td, b["mc_returns"], self.config.critic_target)
        q1, q2 = self.critic(x, self.normalize(b["a"]), ref)
        loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        if not all(torch.isfinite(t).all() for t in (q1, q2, target)):
            self.fused = True
            raise FloatingPointError("non-finite Q or target; actor disabled")
        magnitude = max(float(q1.detach().abs().max()), float(q2.detach().abs().max()))
        self.guard_count = self.guard_count + 1 if magnitude > self.config.q_abort_threshold else 0
        if self.guard_count >= self.config.q_abort_consecutive:
            self.fused = True
            return {"fused": True, "q_abs_max": magnitude}
        self._optimizer_step(self.critic_optimizer, self.critic, loss)
        metrics = {
            "critic_loss": float(loss.detach()),
            "target_q": float(target.mean()),
            "q_abs_max": magnitude,
        }
        if self.updates % self.config.critic_updates_per_actor == 0:
            self.actor.train()
            self.critic.requires_grad_(False)
            try:
                proposal, _ = self.actor.outputs(x, ref, reference_dropout=False)
                action = self.normalize(self.project(proposal, ref))
                q = self.critic.q1(x[:half], action[:half], ref[:half])
                weights = None
                bc_target = self.normalize(ref)
                if self.config.td3_bc_target == "demo":
                    weights = torch.cat(
                        [torch.zeros(half), torch.ones(half) if has_demo else torch.zeros(half)]
                    ).to(self.device)
                    bc_target = self.normalize(b["a"])
                mask = self.tensor(self.config.action_mask)
                a_loss, policy, bc = actor_loss_components(
                    q, action * mask, bc_target * mask, self.config.bc_beta, weights
                )
                _, wrench = self.actor.outputs(x, ref, reference_dropout=False, executed_action=b["a"])
                target_wrench = (
                    b["next_x"][:, -6:] - self.tensor(self.config.proprio_mean[7:])
                ) / self.tensor(self.config.proprio_std[7:])
                prediction = (wrench - target_wrench).square().mean()
                a_loss = a_loss + self.config.wrench_prediction_weight * prediction
                metrics.update(
                    wrench_loss=float(prediction.detach()), wrench_valid_rows=float(len(x)), wrench_coverage=1.0
                )
                self._optimizer_step(self.actor_optimizer, self.actor, a_loss)
                metrics.update(
                    actor_loss=float(a_loss.detach()),
                    actor_policy_loss=float(policy.detach()),
                    actor_bc_loss=float(bc.detach()),
                )
            finally:
                self.critic.q1.requires_grad_(True)
                self.critic.q2.requires_grad_(True)
        self._update_actor_target()
        self.critic.update_targets(self.config.tau)
        self.updates += 1
        return metrics

    @torch.no_grad()
    def _update_actor_target(self):
        for source, target in zip(self.actor.parameters(), self.target_actor.parameters(), strict=True):
            target.lerp_(source, self.config.tau)

    def save(self, path, contract):
        atomic_torch_save(
            {
                "format": "facet_online_adaptation_v3",
                "config": asdict(self.config),
                "contract": contract,
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "replay": self.replay.state_dict(),
                "demo": self.demo.state_dict(),
                "interventions": self.interventions,
                "mean": self.mean,
                "std": self.std,
                "episodes": self.episodes,
                "demo_successes": self.demo_successes,
                "updates": self.updates,
                "bc_updates": self.bc_updates,
                "fused": self.fused,
                "guard_count": self.guard_count,
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            path,
        )

    def load(self, path, contract):

        p = torch.load(path, map_location=self.device, weights_only=False)
        if p.get("format") != "facet_online_adaptation_v3":
            raise ValueError("legacy deployment checkpoints require explicit migration/retraining")
        if p["contract"] != contract:
            raise ValueError("checkpoint policy/token/robot-frame contract mismatch")
        mutable = {
            "device",
            "actor_lr",
            "bc_lr",
            "critic_lr",
            "updates_per_transition",
            "intervention_updates_per_episode",
            "critic_updates_per_actor",
        }
        if any(p["config"][k] != v for k, v in asdict(self.config).items() if k not in mutable):
            raise ValueError("checkpoint architecture or action-space configuration mismatch")
        self.actor.load_state_dict(p["actor"])
        self.target_actor.load_state_dict(p["target_actor"])
        self.critic.load_state_dict(p["critic"])
        self.actor_optimizer.load_state_dict(p["actor_optimizer"])
        self.critic_optimizer.load_state_dict(p["critic_optimizer"])
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.config.critic_lr
        self.replay.load_state_dict(p["replay"])
        self.demo.load_state_dict(p["demo"])
        self.interventions, self.mean, self.std = p["interventions"], p["mean"], p["std"]
        for name in ("episodes", "demo_successes", "updates", "bc_updates", "fused", "guard_count"):
            setattr(self, name, p[name])
        np.random.set_state(p["numpy_rng"])
        torch.set_rng_state(p["torch_rng"].cpu())
        if p.get("cuda_rng") is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in p["cuda_rng"]])
