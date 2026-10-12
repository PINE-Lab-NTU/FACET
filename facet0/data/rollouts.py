"""Convert a deployment-rollout LeRobot export into an RL-ready corpus."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from facet0 import conventions
from facet0.data.lerobot import read_jsonl, read_tasks


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _set_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, array)
    return table.append_column(name, array)


def _canonical_gripper(column: pa.ChunkedArray, flip: bool) -> pa.Array:
    rows = []
    for row in column.to_pylist():
        row = [float(x) for x in row]
        g = min(max(row[conventions.GRIPPER_INDEX], 0.0), 1.0)
        row[conventions.GRIPPER_INDEX] = 1.0 - g if flip else g
        rows.append(row)
    return pa.array(rows, type=pa.list_(pa.float32()))


def _convert_episode(src: Path, dst: Path, success: dict[int, bool], tasks: dict[int, str], flip: bool) -> dict:
    table = pq.read_table(src)
    episodes = set(table.column("episode_index").to_pylist())
    if len(episodes) != 1:
        raise ValueError(f"{src}: expected one episode per file, found {sorted(episodes)}")
    ep = int(episodes.pop())
    if ep not in success:
        raise ValueError(f"{src}: episode {ep} has no rollout_success in meta/episodes.jsonl")
    n = table.num_rows
    prompts = [tasks.get(int(i), "perform the task") for i in table.column("task_index").to_pylist()]
    for key in ("action", "observation.state"):
        if key in table.column_names:
            table = _set_column(table, key, _canonical_gripper(table.column(key), flip))
    table = _set_column(table, "is_success", pa.array([int(success[ep])] * n, type=pa.int64()))
    table = _set_column(table, "done", pa.array([int(i == n - 1) for i in range(n)], type=pa.int64()))
    table = _set_column(table, "reward", pa.array(np.zeros(n, dtype=np.float32)))
    table = _set_column(table, "return", pa.array(np.zeros(n, dtype=np.float32)))
    table = _set_column(table, "prompt", pa.array(prompts, type=pa.string()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return {"episode_index": ep, "frames": n, "success": success[ep], "prompt": prompts[0]}


def prepare(src: Path, dst: Path, *, flip_gripper: bool = True, overwrite: bool = False) -> dict:
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    success = {int(r["episode_index"]): bool(r["rollout_success"]) for r in read_jsonl(src / "meta" / "episodes.jsonl")}
    tasks = read_tasks(src)

    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name in {"data", "videos"}:
            continue
        if item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)
    for f in (src / "videos").rglob("*") if (src / "videos").exists() else ():
        if f.is_file():
            _link_or_copy(f, dst / f.relative_to(src))

    reports = [
        _convert_episode(f, dst / f.relative_to(src), success, tasks, flip_gripper)
        for f in sorted((src / "data").rglob("*.parquet"))
    ]

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    for key in ("action", "observation.state"):
        if key in features:
            features[key]["dtype"] = "float32"
    features["is_success"] = {"dtype": "int64", "shape": [1], "names": None}
    features["done"] = {"dtype": "int64", "shape": [1], "names": None}
    features["reward"] = {"dtype": "float32", "shape": [1], "names": None}
    features["return"] = {"dtype": "float32", "shape": [1], "names": None}
    features["prompt"] = {"dtype": "string", "shape": [1], "names": None}
    info["gripper_encoding"] = "1_closed_0_open"
    info["proprio_mode"] = "state13_pose6_gripper1_wrench6"
    info["rl_ready_prepared_from"] = str(src)
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    report = {
        "source": str(src),
        "destination": str(dst),
        "episodes": len(reports),
        "frames": sum(r["frames"] for r in reports),
        "success_episodes": sum(r["success"] for r in reports),
        "failure_episodes": sum(not r["success"] for r in reports),
        "episodes_per_task": dict(Counter(r["prompt"] for r in reports)),
        "successes_per_task": dict(Counter(r["prompt"] for r in reports if r["success"])),
        "gripper_flipped": flip_gripper,
    }
    (dst / "meta" / "rl_ready_audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Deployment rollouts -> RL-ready post-training corpus.")
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--no-flip-gripper", action="store_true", help="keep the gripper channel of the source as is"
    )
    args = ap.parse_args()
    report = prepare(args.src, args.dst, flip_gripper=not args.no_flip_gripper, overwrite=args.overwrite)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
