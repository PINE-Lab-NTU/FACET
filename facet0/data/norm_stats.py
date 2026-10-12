"""Per-axis normalization statistics of the policy state and the joint action-wrench target."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpi.shared.normalize as normalize
import pyarrow.parquet as pq

from facet0 import action_wrench, conventions
from facet0.data.lerobot import data_files


def stats_dir(root: Path) -> Path:
    return Path(root) / "facet0"


def _rows(column) -> np.ndarray:
    arr = np.asarray(column.to_numpy())
    if arr.dtype == object:
        arr = np.stack(arr.tolist(), axis=0)
    return arr.reshape(len(arr), -1).astype(np.float32)


def _concat(parts: list[normalize.NormStats]) -> normalize.NormStats:
    return normalize.NormStats(
        **{f: np.concatenate([getattr(p, f) for p in parts]) for f in ("mean", "std", "q01", "q99")}
    )


def compute(
    root: Path, horizon: int = conventions.ACTION_HORIZON, window: int = conventions.WRENCH_WINDOW
) -> dict[str, normalize.NormStats]:
    kin, wrench = normalize.RunningStats(), normalize.RunningStats()
    act, fut = normalize.RunningStats(), normalize.RunningStats()
    for f in data_files(root):
        table = pq.read_table(f, columns=["observation.state", "action"])
        state, action = _rows(table["observation.state"]), _rows(table["action"])
        n = len(state)
        kin.update(state[:, conventions.KIN_SLICE])
        wrench.update(state[:, conventions.WRENCH_SLICE])
        rows = np.arange(n)[:, None] + np.arange(horizon)[None, :]
        valid = rows < n
        chunks = action[np.minimum(rows, n - 1)][..., : conventions.ACTION_DIM].copy()
        chunks[..., : conventions.DELTA_ACTION_DIMS] -= state[:, None, : conventions.DELTA_ACTION_DIMS]
        act.update(chunks[valid])
        w_bar = action_wrench.filtered_wrench(state)
        next_rows = rows + 1
        fut.update(w_bar[np.minimum(next_rows, n - 1)][next_rows < n])
    w = wrench.get_statistics()
    return {
        "state": _concat([kin.get_statistics(), *([w] * window)]),
        "actions": _concat([act.get_statistics(), fut.get_statistics()]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute <dataset>/facet0/norm_stats.json.")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None, help="directory for norm_stats.json")
    args = ap.parse_args()
    out_dir = args.out or stats_dir(args.dataset)
    normalize.save(out_dir, compute(args.dataset))
    print(f"wrote {Path(out_dir) / 'norm_stats.json'}")


if __name__ == "__main__":
    main()
