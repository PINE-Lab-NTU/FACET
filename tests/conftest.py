"""Synthetic RL-ready corpora for CPU tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def write_corpus(root: Path, episodes: list[dict]) -> Path:

    (root / "meta").mkdir(parents=True)
    tasks = sorted({e.get("task", 0) for e in episodes})
    (root / "meta" / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": t, "task": f"task {t}"}) + "\n" for t in tasks)
    )
    rows = []
    for i, ep in enumerate(episodes):
        states = np.asarray(ep["states"], dtype=np.float32)
        n = len(states)
        rows.append(json.dumps({"episode_index": i, "length": n, "rollout_success": bool(ep["success"])}) + "\n")
        df = pd.DataFrame(
            {
                "episode_index": np.full(n, i, dtype=np.int64),
                "frame_index": np.arange(n, dtype=np.int64),
                "task_index": np.full(n, ep.get("task", 0), dtype=np.int64),
                "observation.state": list(states),
                "action": list(np.concatenate([states[:, :6], states[:, 6:7]], axis=1)),
                "is_success": np.full(n, int(ep["success"]), dtype=np.int64),
            }
        )
        path = root / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    (root / "meta" / "episodes.jsonl").write_text("".join(rows))
    return root


def contact_episode(n: int, onset: int, success: bool, seed: int) -> dict:

    rng = np.random.default_rng(seed)
    states = np.zeros((n, 13), dtype=np.float32)
    states[:, :3] = np.linspace(0, 0.1, n)[:, None]
    states[:, 6] = 1.0
    states[:, 7:13] = rng.normal(0, 0.05, (n, 6))
    states[onset:, 9] += np.linspace(0, 20, n - onset)
    return {"states": states, "success": success}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    eps = [contact_episode(60, 30 + i, success=i % 2 == 0, seed=i) for i in range(6)]
    return write_corpus(tmp_path / "corpus", eps)
