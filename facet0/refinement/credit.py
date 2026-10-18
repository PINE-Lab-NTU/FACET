"""Credit."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from facet0 import contact
from facet0.data import lerobot


@dataclasses.dataclass(frozen=True)
class CreditConfig:
    horizon: int = 20
    positive_fraction: float = 0.2
    contact: contact.ContactSpec = contact.ContactSpec()


def episode_credit(values: np.ndarray, rewards: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    t_len = len(values)
    csum = np.r_[0.0, np.cumsum(rewards, dtype=np.float64)]
    ends = np.minimum(np.arange(t_len) + horizon, t_len)
    reward_sum = csum[ends] - csum[np.arange(t_len)]
    v_next = np.zeros(t_len, dtype=np.float64)
    inside = np.arange(t_len) + horizon < t_len
    v_next[inside] = values[np.arange(t_len)[inside] + horizon]
    return reward_sum + v_next - values, v_next, reward_sum


def compute_credit(root: Path, scores: pd.DataFrame, returns: pd.DataFrame, cfg: CreditConfig) -> pd.DataFrame:
    if not 0 < cfg.positive_fraction < 1 or not 0 < cfg.horizon < 50:
        raise ValueError("credit requires 0 < positive_fraction < 1 and 0 < horizon < 50")
    frames = scores[["episode_index", "frame_index", "value"]].merge(
        returns[["episode_index", "frame_index", "reward"]], on=["episode_index", "frame_index"], how="left"
    )
    if frames["reward"].isna().any():
        raise ValueError("the reward sidecar does not cover every scored frame")
    frames = frames.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)
    contact_by_episode = contact.episode_contact(lerobot.load_episode_states(root), cfg.contact)

    parts = []
    for ep, group in frames.groupby("episode_index", sort=True):
        fr = group["frame_index"].to_numpy()
        if not np.array_equal(fr, np.arange(len(fr))):
            raise ValueError(f"episode {ep}: frames must be contiguous from 0 to form N-step windows")
        delta, v_next, reward_sum = episode_credit(
            group["value"].to_numpy(np.float64), group["reward"].to_numpy(np.float64), cfg.horizon
        )
        c = contact_by_episode[int(ep)]
        if len(fr) != len(c.intensity):
            raise ValueError(f"episode {ep}: incomplete feature coverage")
        parts.append(
            pd.DataFrame(
                {
                    "episode_index": int(ep),
                    "frame_index": fr,
                    "credit": delta,
                    "contact_intensity": c.intensity[fr].astype(np.float64),
                    "contact_regime": c.regime[fr],
                    "value": group["value"].to_numpy(np.float64),
                    "value_next": v_next,
                    "reward_sum": reward_sum,
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    thresholds = {}
    label = np.zeros(len(out), dtype=bool)
    for name, regime in (("contact", out["contact_regime"].to_numpy()), ("free", ~out["contact_regime"].to_numpy())):
        if regime.any():
            q = float(np.percentile(out["credit"].to_numpy()[regime], (1 - cfg.positive_fraction) * 100))
            rows = np.flatnonzero(regime)
            count = max(1, int(np.ceil(len(rows) * cfg.positive_fraction)))
            order = np.argsort(-out["credit"].to_numpy()[rows], kind="stable")
            label[rows[order[:count]]] = True
            thresholds[name] = q
    out.insert(3, "label", label)
    out.attrs["thresholds"] = thresholds
    return out


def write_credit(root: Path, tag: str, scores_path: Path, returns_tag: str, cfg: CreditConfig) -> Path:
    root = Path(root)
    table = compute_credit(
        root, pd.read_parquet(scores_path), pd.read_parquet(root / "meta" / f"returns_{returns_tag}.parquet"), cfg
    )
    out = root / "meta" / f"credit_{tag}.parquet"
    table.to_parquet(out, index=False)
    regime = table["contact_regime"]
    summary = {
        "tag": tag,
        "scores": str(scores_path),
        "returns_tag": returns_tag,
        "config": dataclasses.asdict(cfg),
        "thresholds": table.attrs["thresholds"],
        "frames": len(table),
        "positive_rate": float(table["label"].mean()),
        "positive_rate_contact": float(table.loc[regime, "label"].mean()) if regime.any() else None,
        "positive_rate_free": float(table.loc[~regime, "label"].mean()) if (~regime).any() else None,
        "contact_rate": float(regime.mean()),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Contact-selective credit (Eq. 6) and labels b_t.")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--tag", required=True, help="writes meta/credit_{tag}.parquet")
    ap.add_argument("--scores", required=True, type=Path, help="parquet with per-frame V_psi(h_t^c)")
    ap.add_argument("--returns-tag", required=True)
    ap.add_argument("--horizon", type=int, default=CreditConfig.horizon)
    ap.add_argument("--positive-fraction", type=float, default=CreditConfig.positive_fraction)
    args = ap.parse_args()
    cfg = CreditConfig(horizon=args.horizon, positive_fraction=args.positive_fraction)
    out = write_credit(args.dataset, args.tag, args.scores, args.returns_tag, cfg)
    print(f"wrote {out}\n{out.with_suffix('.json').read_text()}")


if __name__ == "__main__":
    main()
