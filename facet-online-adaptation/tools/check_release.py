#!/usr/bin/env python3
"""Audit a source tree for generated artifacts and private deployment settings."""

import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".mp4", ".avi", ".log", ".pyc", ".jsonl"}
PRIVATE = [
    r"/home/(?:pinerl|marvel|pine)/",
    r"/Users/",
    r"193\.193\.193\.\d+",
    r"10\.42\.0\.\d+",
    r"\b(?:262622275490|332522071841)\b",
]
TASK_FIELDS = {
    "fixed_grasp_pose",
    "fixed_grasp_lock",
    "startup_handoff_receipt",
    "actor_stage",
    "absolute_action_box_target",
    "reset_pose_file",
    "insertion_start_pose",
}


def check(root=ROOT):
    errors = []
    for p in root.rglob("*"):
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"} for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.name == ".DS_Store":
            continue
        if p.suffix in ARTIFACT_SUFFIXES or any(
            part in {"runs", "run_logs", "checkpoints", "data", "build", "dist"} or part.endswith(".egg-info")
            for part in p.relative_to(root).parts
        ):
            errors.append(f"generated artifact: {p.relative_to(root)}")
        if p.suffix in {".py", ".md", ".yaml", ".toml", ".json"} and p.name != "check_release.py":
            text = p.read_text()
            for pattern in PRIVATE:
                if re.search(pattern, text):
                    errors.append(f"private deployment setting: {p.relative_to(root)}")
            if p.suffix == ".py":
                tree = ast.parse(text, filename=str(p))
                if "src" in p.relative_to(root).parts:
                    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
                    if names & TASK_FIELDS:
                        errors.append(f"task-specific control field: {p.relative_to(root)}")
    if errors:
        raise ValueError("\n".join(errors))
    print("PASS: source-only tree, no private deployment paths/addresses/serials, no task-control fields")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT)
