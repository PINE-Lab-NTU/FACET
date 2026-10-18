"""New-run-only manifests, complete episode arrays, and auditable outcome labels."""

from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np


def create_run(output_dir, config, contract, resume=""):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "facet_adaptation_run_v1",
                "config": config.to_dict(),
                "contract": contract,
                "resume": str(resume),
            },
            indent=2,
        )
        + "\n"
    )
    return path


class EpisodeRecorder:
    def __init__(self, run, index, *, record_images=True):
        self.path = Path(run) / "episodes" / f"episode_{index:06d}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows, self.record_images = [], record_images

    def append(self, *, state, reference, action, x, human, images, wrench_history=None,
               requested_command=None, execution_dt=None):
        index = len(self.rows)
        self.rows.append(
            {
                "state": np.asarray(state).copy(),
                "reference": np.asarray(reference).copy(),
                "action": np.asarray(action).copy(),
                "x": np.asarray(x).copy(),
                "human": bool(human),
                "monotonic_s": time.monotonic(),
            }
        )
        if wrench_history is not None:
            self.rows[-1]["wrench_history"] = np.asarray(wrench_history).copy()
        self.rows[-1]["requested_command"] = np.asarray(action if requested_command is None else requested_command).copy()
        self.rows[-1]["execution_dt"] = np.nan if execution_dt is None else execution_dt
        if self.record_images:
            directory = self.path / "images"
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory / f"{index:06d}.npz", **images)

    def finalize(self, outcome, reason=""):
        if self.rows:
            np.savez_compressed(
                self.path / "steps.npz", **{k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
            )
        (self.path / "outcome.json").write_text(
            json.dumps(
                {
                    "format": "facet_adaptation_episode_v1",
                    "success": outcome,
                    "steps": len(self.rows),
                    "reason": reason,
                    "human_steps": sum(r["human"] for r in self.rows),
                },
                indent=2,
            )
            + "\n"
        )


def append_metrics(run, metrics):
    with (Path(run) / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(metrics, allow_nan=False) + "\n")
