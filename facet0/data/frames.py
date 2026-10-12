"""Corpus frames with wrench history, wrench targets, validity masks and optional credit labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpi.transforms as _transforms
import pandas as pd
from openpi.training import data_loader as _data_loader

from facet0 import action_wrench, conventions
from facet0.data import lerobot


def lerobot_dataset(root: Path, horizon: int):
    """LeRobot (>= 0.3) dataset over the real, possibly sparse, episode index set."""
    import lerobot.datasets.lerobot_dataset as lerobot_ds

    root = Path(root)
    fps = float(json.loads((root / "meta" / "info.json").read_text())["fps"])
    ds = lerobot_ds.LeRobotDataset(
        repo_id=f"facet0/{root.name}",
        root=str(root),
        episodes=lerobot.episode_indices(root),
        delta_timestamps={"action": [t / fps for t in range(horizon)]},
    )
    return _data_loader.TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(lerobot.read_tasks(root))])


def _hf_column(dataset, key: str) -> np.ndarray:
    while not hasattr(dataset, "hf_dataset"):
        dataset = dataset._dataset  # noqa: SLF001
    return np.asarray([int(v) for v in dataset.hf_dataset[key]], dtype=np.int64)


class CorpusFrames:
    def __init__(
        self,
        root: Path,
        horizon: int = conventions.ACTION_HORIZON,
        window: int = conventions.WRENCH_WINDOW,
        credit: Path | pd.DataFrame | None = None,
    ):
        self.root = Path(root)
        self.horizon, self.window = horizon, window
        self._base = lerobot_dataset(self.root, horizon)
        self.episode = _hf_column(self._base, "episode_index")
        self.frame = _hf_column(self._base, "frame_index")
        self._states = lerobot.load_episode_states(self.root)
        self._w_bar = {e: action_wrench.filtered_wrench(s) for e, s in self._states.items()}

        self._labels = None
        if credit is not None:
            table = credit if isinstance(credit, pd.DataFrame) else pd.read_parquet(credit)
            lookup = {
                (int(e), int(f)): (bool(b), float(c))
                for e, f, b, c in zip(
                    table["episode_index"], table["frame_index"], table["label"], table["contact_intensity"]
                )
            }
            missing = [(int(e), int(f)) for e, f in zip(self.episode, self.frame) if (int(e), int(f)) not in lookup]
            if missing:
                raise ValueError(f"{len(missing)} frames have no credit label (first: {missing[:5]})")
            self._labels = [lookup[(int(e), int(f))] for e, f in zip(self.episode, self.frame)]

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, i: int) -> dict:
        sample = dict(self._base[i])
        e, f = int(self.episode[i]), int(self.frame[i])
        states = self._states[e]
        targets, wrench_valid = action_wrench.wrench_targets(self._w_bar[e], f, self.horizon)
        sample["observation.wrench_history"] = action_wrench.wrench_history(states, f, self.window)
        sample["wrench_target"] = targets
        sample["wrench_valid"] = wrench_valid
        sample["action_valid"] = action_wrench.action_valid(f, self.horizon, len(states))
        if self._labels is not None:
            label, intensity = self._labels[i]
            sample["label"] = np.asarray(label)
            sample["contact_intensity"] = np.asarray(intensity, dtype=np.float32)
        return sample
