"""Checkpoint identities and model interface metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 2
FEATURE_SCHEMA = "contact_tokens_tau1_untagged_v1"


def implementation_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in (
        "model.py",
        "conventions.py",
        "action_wrench.py",
        "contact.py",
        "policy/guidance.py",
        "policy/transforms.py",
    ):
        digest.update(name.encode())
        digest.update(bytes.fromhex(file_digest(root / name)))
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"empty parameter directory: {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


def model_metadata(config):
    return {
        key: getattr(config, key)
        for key in (
            "paligemma_variant",
            "action_expert_variant",
            "dtype",
            "action_dim",
            "action_horizon",
            "wrench_window",
            "max_token_len",
            "pi05",
            "discrete_state_input",
            "vision_variant",
        )
    }


def write_manifest(directory, config, *, stage, parent="", extra=None):
    directory = Path(directory)
    payload = {
        "schema": SCHEMA,
        "stage": stage,
        "parent": parent,
        "model": model_metadata(config),
        "feature_schema": FEATURE_SCHEMA,
        "implementation_sha256": implementation_digest(),
        "history_hz": 15.0,
        "target_hz": 15.0,
        "wrench_target_offset": 1,
        "wrench_filter_window": 5,
        "internal_gripper": "1=closed",
        "wire_gripper": "1=open",
        "state_dim": 67,
        "params_sha256": tree_digest(directory / "params"),
        "norm_sha256": file_digest(directory / "assets/norm_stats.json"),
        "tokenizer_sha256": (
            file_digest(directory / "assets/policy_tokenizer.model")
            if (directory / "assets/policy_tokenizer.model").exists() else None
        ),
        "extra": extra or {},
    }
    (directory / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def read_manifest(directory, *, verify=True):
    directory = Path(directory)
    payload = json.loads((directory / "manifest.json").read_text())
    if payload.get("schema") != SCHEMA or payload.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("unsupported checkpoint interface")
    if payload.get("implementation_sha256") != implementation_digest():
        raise ValueError("checkpoint model implementation mismatch")
    if payload.get("wrench_target_offset") != 1 or payload.get("wire_gripper") != "1=open":
        raise ValueError("checkpoint state/action convention mismatch")
    if file_digest(directory / "assets/norm_stats.json") != payload["norm_sha256"]:
        raise ValueError("checkpoint normalization checksum mismatch")
    tokenizer = directory / "assets/policy_tokenizer.model"
    if (file_digest(tokenizer) if tokenizer.exists() else None) != payload.get("tokenizer_sha256"):
        raise ValueError("checkpoint tokenizer checksum mismatch")
    if verify and tree_digest(directory / "params") != payload["params_sha256"]:
        raise ValueError("checkpoint parameter checksum mismatch")
    return payload


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_same_normalization(paths):
    ids = {file_digest(Path(path) / "norm_stats.json") for path in paths}
    if len(ids) != 1:
        raise ValueError("corpora must use identical normalization statistics")
