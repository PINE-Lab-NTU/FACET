"""Joint action-wrench targets Y^data, validity masks M^a / M^w and the causal wrench history W."""

from __future__ import annotations

import numpy as np

from facet0 import contact, conventions

FILTER_WINDOW = 5


def filtered_wrench(states: np.ndarray, window: int = FILTER_WINDOW) -> np.ndarray:
    return contact.causal_filter(np.asarray(states, dtype=np.float32)[:, conventions.WRENCH_SLICE], window)


def wrench_history(states: np.ndarray, frame: int, window: int = conventions.WRENCH_WINDOW) -> np.ndarray:

    rows = np.clip(np.arange(frame - window + 1, frame + 1), 0, len(states) - 1)
    return np.asarray(states, dtype=np.float32)[rows][:, conventions.WRENCH_SLICE]


def wrench_targets(w_bar: np.ndarray, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:

    rows = frame + 1 + np.arange(horizon)
    valid = rows < len(w_bar)
    targets = w_bar[np.clip(rows, 0, len(w_bar) - 1)]
    valid &= np.isfinite(targets).all(axis=-1)
    return np.nan_to_num(targets).astype(np.float32), valid


def action_valid(frame: int, horizon: int, episode_length: int) -> np.ndarray:

    return frame + np.arange(horizon) < episode_length
