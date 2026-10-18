"""Source-demonstration train and validation groups."""

import argparse
import hashlib
import json
from pathlib import Path


def build(root, fraction=0.05, seed=42):
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    rows = [json.loads(line) for line in (Path(root) / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if any(not r.get("source_demo_id") for r in rows):
        raise ValueError("every episode requires source_demo_id")
    groups = sorted(
        {str(r["source_demo_id"]) for r in rows}, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()
    )
    if len(groups) < 2:
        raise ValueError("at least two source demonstration groups required")
    valid = set(groups[: max(1, min(len(groups) - 1, round(len(groups) * fraction)))])
    return {
        "schema": 2,
        "seed": seed,
        "episodes": [
            {
                "episode_index": int(r["episode_index"]),
                "source_demo_id": str(r["source_demo_id"]),
                "split": "validation" if str(r["source_demo_id"]) in valid else "train",
                "tier": r.get("tier", "base"),
            }
            for r in rows
        ],
    }


def load(path):
    data = json.loads(Path(path).read_text())
    seen, ids = {}, set()
    for row in data["episodes"]:
        key, split = str(row["source_demo_id"]), row["split"]
        if row.get("tier", "base") not in {"base", "precision", "recovery"}:
            raise ValueError("unknown sampling tier")
        if split not in {"train", "validation"} or row["episode_index"] in ids:
            raise ValueError("invalid or duplicated split episode")
        ids.add(row["episode_index"])
        if key in seen and seen[key] != split:
            raise ValueError("source demonstration crosses splits")
        seen[key] = split
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.write_text(json.dumps(build(args.dataset, seed=args.seed), indent=2) + "\n")


if __name__ == "__main__":
    main()
