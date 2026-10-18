"""Evaluate held-out action-wrench and VQA losses."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.model import Observation

from facet0.contracts import read_manifest
from facet0.model import Facet0Config
from facet0.policy.checkpoint import restore_params, load_norm_stats
from facet0.pretraining.core import aw_loss, vqa_loss
from facet0.pretraining.data import RobotFrames, VQAFrames
from facet0.pretraining.vqa import VQADataConfig, VQATokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    if args.batches < 1 or args.batch_size < 1:
        raise ValueError("positive evaluation sizes required")
    cfg = json.loads(args.config.read_text())
    manifest = read_manifest(args.checkpoint)
    model_cfg = Facet0Config(**manifest["model"])
    model = model_cfg.load(restore_params(args.checkpoint / "params"))
    model.eval()
    norm = load_norm_stats(args.checkpoint / "assets")
    robot = RobotFrames(
        Path(cfg["dataset"]),
        model_cfg,
        norm,
        cfg["split"],
        cfg.get("tier_weights", {"base": 1.0, "precision": 2.0, "recovery": 2.0}),
        split="validation",
        tokenizer_path=str(args.checkpoint / "assets/policy_tokenizer.model") if manifest.get("tokenizer_sha256") else None,
    )
    vqa = VQAFrames(
        cfg["vqa_records"],
        cfg["split"],
        VQATokenizer(VQADataConfig(**cfg.get("vqa", {}))),
        cfg["vqa_images"],
        split="validation",
    )
    rng = np.random.default_rng(42)
    losses = []
    for i in range(args.batches):
        batch = jax.tree.map(jnp.asarray, robot.batch(rng, args.batch_size))
        aw = aw_loss(
            model,
            jax.random.key(i),
            Observation.from_dict(batch),
            batch["actions"],
            batch,
            microbatch_size=args.batch_size,
            train=False,
        )
        semantic = vqa_loss(model, jax.tree.map(jnp.asarray, vqa.batch(rng, args.batch_size)))
        losses.append([float(aw), float(semantic)])
    print(json.dumps(dict(zip(("action_wrench_loss", "vqa_loss"), np.mean(losses, axis=0))), indent=2))


if __name__ == "__main__":
    main()
