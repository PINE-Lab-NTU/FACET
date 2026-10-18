"""Contact."""

from __future__ import annotations

import dataclasses

import numpy as np

from facet0.data import lerobot


@dataclasses.dataclass(frozen=True)
class ContactSpec:
    filter_window: int = 5
    lag: int = 8
    episode_quantile: float = 0.8
    corpus_quantile: float = 0.5
    regime_threshold: float = 1.0


def causal_filter(x: np.ndarray, window: int) -> np.ndarray:

    if window <= 1:
        return x
    csum = np.cumsum(x, axis=0)
    out = np.empty_like(x)
    for t in range(len(x)):
        lo = max(0, t - window + 1)
        total = csum[t] - (csum[lo - 1] if lo > 0 else 0.0)
        out[t] = total / (t - lo + 1)
    return out


def wrench_change(wrench: np.ndarray, spec: ContactSpec) -> np.ndarray:

    w_bar = causal_filter(np.asarray(wrench, dtype=np.float32), spec.filter_window)
    prev = np.maximum(np.arange(len(w_bar)) - spec.lag, 0)
    return np.linalg.norm(w_bar - w_bar[prev], axis=-1)


@dataclasses.dataclass(frozen=True)
class EpisodeContact:
    change: np.ndarray
    scale: float
    intensity: np.ndarray
    regime: np.ndarray


def episode_contact(states_by_episode: dict[int, np.ndarray], spec: ContactSpec) -> dict[int, EpisodeContact]:
    changes = {e: wrench_change(lerobot.wrench(s), spec) for e, s in states_by_episode.items() if len(s)}
    floor = float(np.quantile(np.concatenate(list(changes.values())), spec.corpus_quantile)) if changes else 0.0
    out = {}
    for e, d in changes.items():
        kappa = max(float(np.quantile(d, spec.episode_quantile)), floor)
        intensity = np.clip(d / kappa, 0.0, 1.0) if kappa > 1e-6 else np.zeros_like(d)
        out[e] = EpisodeContact(change=d, scale=kappa, intensity=intensity, regime=intensity >= spec.regime_threshold)
    return out
