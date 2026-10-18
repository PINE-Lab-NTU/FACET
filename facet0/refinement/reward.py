"""Reward."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from facet0 import conventions
from facet0.data import lerobot

JUDGE_COLUMNS = ("subgoal_index", "subgoal_complete", "phase")


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    gamma: float = 0.99
    alpha: float = 0.0
    beta: float = 0.0
    time_budget: float | None = None
    envelopes: dict[str, tuple[float, ...]] | None = None
    sparse: bool = False

    def validate(self):
        if not 0 < self.gamma <= 1 or self.alpha < 0 or self.beta < 0:
            raise ValueError("invalid reward coefficients")
        if not self.sparse and (not self.envelopes or self.alpha <= 0 or self.time_budget is None):
            raise ValueError("interaction rewards require envelopes, alpha > 0 and a time budget")
        if self.beta > 0 and (self.time_budget is None or self.time_budget <= 0):
            raise ValueError("speed bonus requires a positive time budget")
        for values in (self.envelopes or {}).values():
            if len(values) != 6 or not np.isfinite(values).all() or min(values) <= 0:
                raise ValueError("wrench envelopes must contain six positive finite values")


def interaction_reward(
    complete: np.ndarray,
    elapsed: np.ndarray,
    wrench: np.ndarray,
    envelope: np.ndarray | None,
    cfg: RewardConfig,
) -> np.ndarray:

    r = complete.astype(np.float64)
    if cfg.beta > 0 and cfg.time_budget:
        r = r * (1.0 + cfg.beta * np.maximum(1.0 - elapsed / cfg.time_budget, 0.0))
    if cfg.alpha > 0 and envelope is not None:
        normalized = np.max(np.abs(wrench) / envelope, axis=-1)
        r = r - cfg.alpha * np.maximum(normalized - 1.0, 0.0)
    return r.astype(np.float32)


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    g = np.zeros(len(rewards), dtype=np.float32)
    g[-1] = rewards[-1]
    for t in range(len(rewards) - 2, -1, -1):
        g[t] = rewards[t] + gamma * g[t + 1]
    return g


def _episode_reward(df: pd.DataFrame, cfg: RewardConfig) -> np.ndarray:
    n = len(df)
    states = np.stack([np.asarray(s, np.float32).reshape(-1) for s in df["observation.state"]])
    wrench = states[:, conventions.WRENCH_SLICE]
    present = [c in df.columns for c in JUDGE_COLUMNS]
    if any(present) and not all(present):
        raise ValueError("incomplete judge columns")
    if all(present):
        subgoal = df["subgoal_index"].to_numpy()
        complete = df["subgoal_complete"].to_numpy().astype(bool)
        phase = [str(p) for p in df["phase"]]
    else:
        if not cfg.sparse:
            raise ValueError("judge columns required; use sparse mode for outcome-only data")
        subgoal = np.zeros(n, dtype=np.int64)
        complete = np.zeros(n, dtype=bool)
        complete[-1] = bool(np.asarray(df["is_success"].iloc[-1]).reshape(-1)[0])
        phase = None
    starts = np.r_[0, np.flatnonzero(np.diff(subgoal)) + 1]
    elapsed = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n])) + 1
    envelope = None
    if cfg.envelopes:
        default = cfg.envelopes.get("*")
        rows = [cfg.envelopes.get(p, default) for p in phase] if phase else [default] * n
        if any(r is None for r in rows):
            raise ValueError("missing phase wrench envelope")
        envelope = np.asarray(rows, dtype=np.float64)
    return interaction_reward(complete, elapsed.astype(np.float64), wrench, envelope, cfg)


def compute_returns(root: Path, cfg: RewardConfig) -> pd.DataFrame:
    cfg.validate()
    parts = []
    for f in lerobot.data_files(root):
        names = set(pq.ParquetFile(f).schema_arrow.names)
        cols = ["episode_index", "frame_index", "observation.state", "is_success"]
        cols += [c for c in JUDGE_COLUMNS if c in names]
        df = pd.read_parquet(f, columns=cols).sort_values("frame_index")
        if not len(df):
            continue
        rewards = _episode_reward(df, cfg)
        parts.append(
            pd.DataFrame(
                {
                    "episode_index": df["episode_index"].to_numpy(np.int64),
                    "frame_index": df["frame_index"].to_numpy(np.int64),
                    "reward": rewards,
                    "return": discounted_returns(rewards, cfg.gamma),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def write_returns(root: Path, tag: str, cfg: RewardConfig) -> Path:
    out = Path(root) / "meta" / f"returns_{tag}.parquet"
    compute_returns(root, cfg).to_parquet(out, index=False)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Interaction reward (Eq. 4) and returns sidecar.")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gamma", type=float, default=RewardConfig.gamma)
    ap.add_argument("--alpha", type=float, default=RewardConfig.alpha)
    ap.add_argument("--beta", type=float, default=RewardConfig.beta)
    ap.add_argument("--time-budget", type=float, default=None, help="T_max in frames")
    ap.add_argument("--envelopes", type=Path, default=None, help='JSON {phase: [6 limits], "*": [...]}')
    ap.add_argument("--sparse", action="store_true")
    args = ap.parse_args()
    envelopes = {k: tuple(v) for k, v in json.loads(args.envelopes.read_text()).items()} if args.envelopes else None
    cfg = RewardConfig(args.gamma, args.alpha, args.beta, args.time_budget, envelopes, args.sparse)
    print(f"wrote {write_returns(args.dataset, args.tag, cfg)}")


if __name__ == "__main__":
    main()
