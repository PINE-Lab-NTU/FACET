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
from .gate import CriticReadinessGate, GateLimits, annealed_beta
from .safety import SafetyGuard, RobotState
from .models import Actor, TwinQCritic
from .replay import ReplayBuffer

CHECKPOINT_FORMAT = "facet_online_adaptation_v4"
# v3 checkpoints share the network/replay layout; they resume with a fresh gate, monitor and seed.
MIGRATABLE_FORMATS = {"facet_online_adaptation_v3"}
# Schedule and gate settings may change between sessions; architecture, action space and the
# Critic regression rule may not.
MUTABLE_CONFIG = {
    "device",
    "actor_lr",
    "bc_lr",
    "critic_lr",
    "updates_per_transition",
    "intervention_updates_per_episode",
    "rl_intervention_updates_per_episode",
    "critic_updates_per_actor",
    "bc_beta",
    "bc_beta_end",
    "bc_anneal_critic_updates",
    "actor_q_weight",
    "hold_actor_until_critic_ready",
    "monitor_capacity",
    "gate_eval_every",
    "gate_min_updates",
    "gate_consecutive_passes",
    "gate_min_terminals",
    "gate_terminal_brier_max",
    "gate_success_q_min",
    "gate_failure_q_abs_max",
    "gate_disagreement_max",
    "gate_fixed_q_delta_max",
    "gate_target_q_delta_max",
    "gate_seed_drift_max_m",
    "gate_bc_degradation_max",
    "critic_lr_schedule",
    "critic_lr_final",
    "critic_lr_decay_updates",
}


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


def actor_loss_components(q, proposed, target, beta, weights=None, q_weight=1.0):
    """TD3+BC Actor loss ``q_weight * (-Q) + beta * BC``.

    ``q=None`` (or ``q_weight=0``) gives the matched BC control: same rows, same BC weighting,
    no Critic term in the graph.
    """

    per_row = (proposed - target).square().sum(-1)
    bc = per_row.mean() if weights is None else (per_row * weights).sum() / weights.sum().clamp_min(1)
    policy = torch.zeros((), device=proposed.device) if q is None else -q.mean()
    return q_weight * policy + beta * bc, policy, bc


def critic_lr_at(config, updates):
    """Critic LR on the Critic-update clock: hold through gate_min_updates, then optional cosine."""
    if config.critic_lr_schedule == "constant" or not config.critic_lr_decay_updates:
        return config.critic_lr
    progress = min(1.0, max(0, updates - config.gate_min_updates) / config.critic_lr_decay_updates)
    return config.critic_lr_final + 0.5 * (config.critic_lr - config.critic_lr_final) * (
        1 + np.cos(np.pi * progress)
    )


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
        # Critic-gated TD3+BC state.
        self.gate = CriticReadinessGate(GateLimits.from_config(config))
        self.seed_actor = None  # BC seed the gate measures drift/degradation against
        self.monitor = []  # held-out labelled episodes, never trained on
        self.labelled_episodes = 0
        self.rl_active_critic_updates = 0
        self.td3_actor_updates = self.held_actor_updates = 0
        self.last_gate_metrics = {}

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
        held_out = self.config.monitor_every > 0 and (self.labelled_episodes + 1) % self.config.monitor_every == 0
        seed_demo = (
            success and not held_out and self.demo_successes < self.config.demo_seed_success_episodes
        )

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
        self.labelled_episodes += 1
        if held_out:
            self._add_monitor_episode(steps, success)
        new_labels = 0
        active = np.asarray(self.config.action_mask)
        for i, row in enumerate(steps):
            if held_out:
                break
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
            inside = np.all((np.asarray(row["action"]) >= self.config.action_min) | ~active) and np.all(
                (np.asarray(row["action"]) <= self.config.action_max) | ~active
            )
            confirmed = bool(row.get("bc_confirmed", success))
            if inside and (seed_demo or (human and (confirmed or not self.config.demo_success_only))):
                self.demo.add(**item)
            if human and confirmed and inside:
                self.interventions.append((row["x"].copy(), row["reference"].copy(), row["action"].copy()))
                new_labels += 1
        self.interventions = self.interventions[-self.config.intervention_capacity :]
        self.demo_successes += int(seed_demo)
        self.episodes += 1
        metrics = {}
        bc_before, td3_before = self.bc_updates, self.td3_actor_updates
        # Intervention BC only consumes NEW confirmed labels; replaying the whole label set after every
        # autonomous episode would let BC dominate the Actor.
        bc_steps = self.config.intervention_updates_per_episode if self.mode == "bc" else (
            self.config.rl_intervention_updates_per_episode
        )
        if new_labels and self.interventions:
            for _ in range(bc_steps):
                metrics.update(self.update_bc())
        if self.mode == "rl" and self.replay.size >= self.config.batch_size:
            for _ in range(self.config.updates_per_transition * len(steps)):
                metrics.update(self.update_td3())
                if self.fused:
                    break
        bc_now, td3_now = self.bc_updates - bc_before, self.td3_actor_updates - td3_before
        self.last_metrics = {
            **metrics,
            "episodes": self.episodes,
            "replay_size": self.replay.size,
            "demo_size": self.demo.size,
            "monitor_episodes": len(self.monitor),
            "held_out_for_monitor": float(held_out),
            "new_confirmed_labels": new_labels,
            "updates": self.updates,
            "bc_updates": self.bc_updates,
            "td3_actor_updates": self.td3_actor_updates,
            "episode_bc_updates": bc_now,
            "episode_td3_actor_updates": td3_now,
            "episode_q_gradient_share": td3_now / max(1, bc_now + td3_now),
            "gate_active": float(self.gate.active),
            "rl_active_critic_updates": self.rl_active_critic_updates,
            "success": float(success),
            "fused": self.fused,
        }
        return self.last_metrics

    def _add_monitor_episode(self, steps, success):
        self.monitor.append(
            dict(
                x=np.stack([np.asarray(r["x"], np.float32) for r in steps]),
                action=np.stack([np.asarray(r["action"], np.float32) for r in steps]),
                reference=np.stack([np.asarray(r["reference"], np.float32) for r in steps]),
                success=bool(success),
            )
        )
        while sum(len(e["x"]) for e in self.monitor) > self.config.monitor_capacity and len(self.monitor) > 1:
            self.monitor.pop(0)

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
        if self.seed_actor is None:
            # The BC seed is the Actor as RL training starts; the gate measures drift against it.
            self.seed_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_lr_at(self.config, self.updates)
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
        step = self.updates + 1  # Critic-update clock, 1-based like the offline schedule
        metrics = {
            "critic_loss": float(loss.detach()),
            "target_q": float(target.mean()),
            "q_abs_max": magnitude,
            "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
        }
        if self.config.critic_gate and step % self.config.gate_eval_every == 0:
            metrics.update(self.evaluate_gate(step))
        rl_active = (not self.config.critic_gate) or self.gate.active
        if rl_active:
            self.rl_active_critic_updates += 1
        beta = annealed_beta(
            self.config.bc_beta, self.config.bc_beta_end, self.config.bc_anneal_critic_updates,
            self.rl_active_critic_updates if rl_active else 0,
        )
        metrics.update(
            rl_active=float(rl_active), bc_beta_effective=beta,
            rl_active_critic_updates=float(self.rl_active_critic_updates),
        )
        if step % self.config.critic_updates_per_actor == 0:
            if rl_active:
                metrics.update(self._actor_step(b, half, has_demo, beta, self.config.actor_q_weight))
                self.td3_actor_updates += int(self.config.actor_q_weight > 0)
            elif not self.config.hold_actor_until_critic_ready:
                # Gate closed and no hold: keep BC-accumulating on the same cadence, without Q.
                metrics.update(self._actor_step(b, half, has_demo, self.config.bc_beta, 0.0))
            else:
                self.held_actor_updates += 1
                metrics["actor_held_for_critic_gate"] = 1.0
        self._update_actor_target()
        self.critic.update_targets(self.config.tau)
        self.updates = step
        return metrics

    def _actor_step(self, b, half, has_demo, beta, q_weight):
        x, ref = b["x"], b["a_tilde"]
        self.actor.train()
        self.critic.requires_grad_(False)
        try:
            proposal, _ = self.actor.outputs(x, ref, reference_dropout=False)
            action = self.normalize(self.project(proposal, ref))
            q = self.critic.q1(x[:half], action[:half], ref[:half]) if q_weight > 0 else None
            weights = None
            bc_target = self.normalize(ref)
            if self.config.td3_bc_target == "demo":
                weights = torch.cat(
                    [torch.zeros(half), torch.ones(half) if has_demo else torch.zeros(half)]
                ).to(self.device)
                bc_target = self.normalize(b["a"])
            mask = self.tensor(self.config.action_mask)
            a_loss, policy, bc = actor_loss_components(
                q, action * mask, bc_target * mask, beta, weights, q_weight=q_weight
            )
            _, wrench = self.actor.outputs(x, ref, reference_dropout=False, executed_action=b["a"])
            target_wrench = (
                b["next_x"][:, -6:] - self.tensor(self.config.proprio_mean[7:])
            ) / self.tensor(self.config.proprio_std[7:])
            prediction = (wrench - target_wrench).square().mean()
            a_loss = a_loss + self.config.wrench_prediction_weight * prediction
            self._optimizer_step(self.actor_optimizer, self.actor, a_loss)
            return dict(
                wrench_loss=float(prediction.detach()),
                wrench_valid_rows=float(len(x)),
                wrench_coverage=1.0,
                actor_loss=float(a_loss.detach()),
                actor_policy_loss=float(policy.detach()),
                actor_bc_loss=float(bc.detach()),
                actor_uses_q_gradient=float(q_weight > 0),
            )
        finally:
            self.critic.q1.requires_grad_(True)
            self.critic.q2.requires_grad_(True)

    @torch.no_grad()
    def gate_metrics(self):
        """Critic and Actor checks on held-out monitor episodes (recorded states and actions)."""
        nan = float("nan")
        empty = dict(success_terminals=0, failure_terminals=0, terminal_brier=nan, success_terminal_q=nan,
                     failure_terminal_q=nan, q_disagreement=nan, seed_drift_p95_m=nan, bc_degradation=nan)
        if not self.monitor or self.seed_actor is None:
            return empty, (np.zeros(0), np.zeros(0))
        x = self.tensor(np.concatenate([e["x"] for e in self.monitor]))
        a = self.tensor(np.concatenate([e["action"] for e in self.monitor]))
        ref = self.tensor(np.concatenate([e["reference"] for e in self.monitor]))
        success = np.concatenate([np.full(len(e["x"]), e["success"]) for e in self.monitor])
        terminal = np.zeros(len(success), bool)
        terminal[np.cumsum([len(e["x"]) for e in self.monitor]) - 1] = True
        self.critic.eval()
        q1, q2 = self.critic(x, self.normalize(a), ref)
        q = torch.minimum(q1, q2).cpu().numpy().reshape(-1)
        target_q = self.critic.target_q_min(x, self.normalize(a), ref).cpu().numpy().reshape(-1)
        self.critic.train()
        terminal_q, labels = q[terminal], success[terminal]
        self.actor.eval()
        proposal = self.project(self.actor(x, ref, reference_dropout=False), ref)
        seed = self.project(self.seed_actor(x, ref, reference_dropout=False), ref)
        drift = torch.linalg.vector_norm(proposal[:, :3] - seed[:, :3], dim=1).cpu().numpy()
        pose = self.tensor(np.r_[np.asarray(self.config.action_mask[:6], np.float32), 0.0])
        rows = torch.as_tensor(success, device=self.device)

        def bc_mse(actions):
            if not rows.any():
                return nan
            err = (self.normalize(actions) - self.normalize(a))[rows] * pose
            return float(err.square().sum(-1).mean() / pose.sum().clamp_min(1))

        current, initial = bc_mse(proposal), bc_mse(seed)
        metrics = dict(
            success_terminals=int(labels.sum()),
            failure_terminals=int((~labels).sum()),
            terminal_brier=float(np.mean((terminal_q - labels) ** 2)) if len(labels) else nan,
            success_terminal_q=float(terminal_q[labels].mean()) if labels.any() else nan,
            failure_terminal_q=float(terminal_q[~labels].mean()) if (~labels).any() else nan,
            q_disagreement=float((q1 - q2).abs().mean()),
            seed_drift_p95_m=float(np.percentile(drift, 95)),
            bc_degradation=current / initial if initial > 0 else (1.0 if current == 0 else nan),
        )
        return metrics, (q, target_q)

    def evaluate_gate(self, step):
        metrics, probe = self.gate_metrics()
        result = {f"gate/{k}": float(v) for k, v in metrics.items()}
        result.update(self.gate.observe(step, metrics, probe))
        # Unavailable checks (empty monitor, first probe) are logged as null; run metrics are strict JSON.
        result = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in result.items()}
        self.last_gate_metrics = result
        return result

    @torch.no_grad()
    def _update_actor_target(self):
        for source, target in zip(self.actor.parameters(), self.target_actor.parameters(), strict=True):
            target.lerp_(source, self.config.tau)

    def save(self, path, contract):
        atomic_torch_save(
            {
                "format": CHECKPOINT_FORMAT,
                "seed_actor": None if self.seed_actor is None else self.seed_actor.state_dict(),
                "gate": self.gate.state_dict(),
                "monitor": self.monitor,
                "labelled_episodes": self.labelled_episodes,
                "rl_active_critic_updates": self.rl_active_critic_updates,
                "td3_actor_updates": self.td3_actor_updates,
                "held_actor_updates": self.held_actor_updates,
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
        fmt = p.get("format")
        if fmt != CHECKPOINT_FORMAT and fmt not in MIGRATABLE_FORMATS:
            raise ValueError("legacy deployment checkpoints require explicit migration/retraining")
        if p["contract"] != contract:
            raise ValueError("checkpoint policy/token/robot-frame contract mismatch")
        # Fields added after the saved format keep their configured values; every saved field that
        # is not a schedule/gate setting must match exactly.
        saved = p["config"]
        if any(saved[k] != v for k, v in asdict(self.config).items() if k in saved and k not in MUTABLE_CONFIG):
            raise ValueError(
                "checkpoint architecture or action-space configuration mismatch "
                "(the Critic target rule is fixed per run; resume with the saved critic_target)"
            )
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
        if fmt == CHECKPOINT_FORMAT:
            if p["seed_actor"] is not None:
                self.seed_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
                self.seed_actor.load_state_dict(p["seed_actor"])
            self.gate.load_state_dict(p["gate"])
            self.monitor = p["monitor"]
            for name in ("labelled_episodes", "rl_active_critic_updates", "td3_actor_updates", "held_actor_updates"):
                setattr(self, name, p[name])
        else:
            # Migrated run: the gate starts closed, the monitor empty, and the seed is the loaded Actor
            # at the first RL update.
            self.labelled_episodes = self.episodes
        np.random.set_state(p["numpy_rng"])
        torch.set_rng_state(p["torch_rng"].cpu())
        if p.get("cuda_rng") is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in p["cuda_rng"]])
