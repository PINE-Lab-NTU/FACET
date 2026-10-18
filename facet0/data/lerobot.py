"""Read the low-dimensional columns of a LeRobot v2.1 corpus without decoding video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from facet0 import conventions


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_tasks(root: Path) -> dict[int, str]:
    return {int(r["task_index"]): str(r["task"]) for r in read_jsonl(Path(root) / "meta" / "tasks.jsonl")}


def episode_indices(root: Path) -> list[int]:

    return sorted(int(r["episode_index"]) for r in read_jsonl(Path(root) / "meta" / "episodes.jsonl"))


def data_files(root: Path) -> list[Path]:
    files = sorted((Path(root) / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquet files under {Path(root) / 'data'}")
    return files


def _stack(column) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in column], axis=0)


def load_frames(root: Path, columns: list[str]) -> pd.DataFrame:

    parts = [pd.read_parquet(f, columns=columns) for f in data_files(root)]
    return pd.concat(parts, ignore_index=True)


def load_episode_states(root: Path, key: str = "observation.state") -> dict[int, np.ndarray]:

    out: dict[int, np.ndarray] = {}
    for f in data_files(root):
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", key])
        for ep, group in df.groupby("episode_index", sort=True):
            group = group.sort_values("frame_index")
            frames = group["frame_index"].to_numpy(dtype=np.int64)
            states = _stack(group[key].to_list())
            if frames[0] == 0 and np.array_equal(frames, np.arange(len(frames))):
                out[int(ep)] = states
            else:
                raise ValueError(f"episode {ep}: frame indices must be contiguous from zero")
    return out


def wrench(states: np.ndarray) -> np.ndarray:

    return np.asarray(states, dtype=np.float32)[:, conventions.WRENCH_SLICE]


def episode_success(root: Path) -> dict[int, bool]:

    df = load_frames(root, ["episode_index", "frame_index", "is_success"])
    df = df.sort_values(["episode_index", "frame_index"])
    last = df.groupby("episode_index", sort=True).tail(1)
    return {int(e): bool(s) for e, s in zip(last["episode_index"], last["is_success"])}
