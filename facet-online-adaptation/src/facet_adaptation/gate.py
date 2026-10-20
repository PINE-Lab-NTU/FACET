"""Critic readiness gate and RL-active BC annealing for TD3+BC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

import numpy as np

ACTOR_CHECKS = frozenset({"actor_trust_region", "successful_behavior"})


@dataclass(frozen=True)
class GateLimits:
    min_updates: int = 750
    consecutive_passes: int = 3
    min_terminals: int = 3
    terminal_brier_max: float = 0.1
    success_q_min: float = 0.7
    failure_q_abs_max: float = 0.3
    disagreement_max: float = 0.1
    fixed_q_delta_max: float = 0.02
    target_q_delta_max: float = 0.02
    trust_region_max_m: float = 0.001
    bc_degradation_max: float = 1.1

    @classmethod
    def from_config(cls, config):
        return cls(
            min_updates=config.gate_min_updates,
            consecutive_passes=config.gate_consecutive_passes,
            min_terminals=config.gate_min_terminals,
            terminal_brier_max=config.gate_terminal_brier_max,
            success_q_min=config.gate_success_q_min,
            failure_q_abs_max=config.gate_failure_q_abs_max,
            disagreement_max=config.gate_disagreement_max,
            fixed_q_delta_max=config.gate_fixed_q_delta_max,
            target_q_delta_max=config.gate_target_q_delta_max,
            trust_region_max_m=config.gate_trust_region_max_m,
            bc_degradation_max=config.gate_bc_degradation_max,
        )


def _passes(value, predicate):
    return math.isfinite(float(value)) and bool(predicate(float(value)))


class CriticReadinessGate:
    def __init__(self, limits: GateLimits):
        self.limits = limits
        self.active = False
        self.pass_streak = 0
        self.events: list[dict] = []
        self.last_failed: list[str] = []
        self.previous_probe: tuple[np.ndarray, np.ndarray] | None = None
        self.evaluations = 0

    def critic_ready(self):
        return self.evaluations > 0 and not set(self.last_failed) - ACTOR_CHECKS

    def _probe_deltas(self, probe):
        q, target_q = (np.asarray(v, np.float64).reshape(-1) for v in probe)
        previous, self.previous_probe = self.previous_probe, (q.copy(), target_q.copy())
        if previous is None or previous[0].shape != q.shape or not len(q):
            return math.inf, math.inf
        return (
            float(np.sqrt(np.mean((q - previous[0]) ** 2))),
            float(np.sqrt(np.mean((target_q - previous[1]) ** 2))),
        )

    def observe(self, updates: int, metrics: dict, probe) -> dict:
        fixed_delta, target_delta = self._probe_deltas(probe)
        lim = self.limits
        checks = {
            "minimum_critic_updates": updates >= lim.min_updates,
            "monitor_coverage": metrics["success_terminals"] >= lim.min_terminals
            and metrics["failure_terminals"] >= lim.min_terminals,
            "terminal_calibration": _passes(metrics["terminal_brier"], lambda v: v <= lim.terminal_brier_max),
            "success_terminal_q": _passes(metrics["success_terminal_q"], lambda v: v >= lim.success_q_min),
            "failure_terminal_q": _passes(metrics["failure_terminal_q"], lambda v: abs(v) <= lim.failure_q_abs_max),
            "twin_q_disagreement": _passes(metrics["q_disagreement"], lambda v: v <= lim.disagreement_max),
            "fixed_q_stability": fixed_delta <= lim.fixed_q_delta_max,
            "target_q_stability": target_delta <= lim.target_q_delta_max,
            "actor_trust_region": _passes(metrics["anchor_drift_p95_m"], lambda v: v <= lim.trust_region_max_m),
            "successful_behavior": _passes(metrics["bc_degradation"], lambda v: v <= lim.bc_degradation_max),
        }
        self.last_failed = [name for name, passed in checks.items() if not passed]
        self.evaluations += 1
        self.pass_streak = 0 if self.last_failed else self.pass_streak + 1
        was_active = self.active
        self.active = self.pass_streak >= lim.consecutive_passes
        if self.active != was_active:
            self.events.append(
                {"update": int(updates), "event": "opened" if self.active else "paused",
                 "failed_checks": list(self.last_failed)}
            )
        return {
            "gate_active": float(self.active),
            "gate_check_passed": float(not self.last_failed),
            "gate_pass_streak": float(self.pass_streak),
            "gate_fixed_q_delta_rms": fixed_delta,
            "gate_target_q_delta_rms": target_delta,
            "gate_failed_checks": ",".join(self.last_failed),
        }

    def state_dict(self):
        return {
            "limits": asdict(self.limits),
            "active": self.active,
            "pass_streak": self.pass_streak,
            "events": copy.deepcopy(self.events),
            "last_failed": list(self.last_failed),
            "evaluations": self.evaluations,
            "previous_probe": None if self.previous_probe is None else [v.copy() for v in self.previous_probe],
        }

    def load_state_dict(self, state):
        self.active = bool(state["active"])
        self.pass_streak = int(state["pass_streak"])
        self.events = copy.deepcopy(state["events"])
        self.last_failed = list(state["last_failed"])
        self.evaluations = int(state.get("evaluations", 0))
        probe = state.get("previous_probe")
        self.previous_probe = None if probe is None else tuple(np.asarray(v) for v in probe)


def annealed_beta(start: float, end: float, anneal_updates: int, rl_active_updates: int) -> float:
    progress = min(1.0, rl_active_updates / anneal_updates) if anneal_updates else 0.0
    return float(start + progress * (end - start))
