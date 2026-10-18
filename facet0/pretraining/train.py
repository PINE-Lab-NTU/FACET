"""Train and resume alternating action-wrench and VQA updates."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from openpi.models.model import Observation

from facet0.contracts import file_digest, identity, model_metadata, read_manifest, write_manifest
from facet0.model import Facet0Config
from facet0.policy.checkpoint import load_norm_stats, restore_params, save_norm_stats
from facet0.pretraining.core import aw_loss, update, vqa_filter, vqa_loss
from facet0.pretraining.data import RobotFrames, VQAFrames
from facet0.pretraining.vqa import VQADataConfig, VQATokenizer


def save(directory, model, model_config, norm, state, contract, parent):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(directory / "params", {"params": nnx.state(model).to_pure_dict()})
    save_norm_stats(directory / "assets", norm)
    tokenizer_path = contract.get("config", {}).get("policy_tokenizer")
    if tokenizer_path:
        shutil.copy2(tokenizer_path, directory / "assets/policy_tokenizer.model")
    with (directory / "training.pkl").open("wb") as stream:
        pickle.dump({"state": jax.device_get(state), "contract": contract}, stream)
    write_manifest(
        directory,
        model_config,
        stage="pretraining",
        parent=parent,
        extra={"step": state["step"], "aw_steps": state["aw_steps"], "contract": contract},
    )


def run(cfg):
    model_cfg = Facet0Config(**cfg.get("model", {}))
    norm_path = Path(cfg["norm_stats"])
    norm = load_norm_stats(norm_path)
    contract = {
        "config": {k: v for k, v in cfg.items() if k not in {"steps", "save_every", "output", "resume"}},
        "norm": file_digest(norm_path / "norm_stats.json"),
        "split": file_digest(cfg["split"]),
        "vqa": file_digest(cfg["vqa_records"]),
        "tokenizer": file_digest(cfg["policy_tokenizer"]) if cfg.get("policy_tokenizer") else None,
    }
    rng = np.random.default_rng(cfg.get("seed", 42))
    parent = ""
    initial = cfg.get("resume") or cfg.get("initial")
    if initial:
        manifest = read_manifest(initial)
        if manifest["norm_sha256"] != contract["norm"]:
            raise ValueError("initial checkpoint normalization differs from training normalization")
        if manifest["model"] != model_metadata(model_cfg):
            raise ValueError("initial model configuration differs")
        parent = identity(manifest)
        model = model_cfg.load(restore_params(Path(initial) / "params"))
    else:
        if model_cfg.paligemma_variant != "dummy":
            raise ValueError("full-size training requires an initial checkpoint")
        model = model_cfg.create(jax.random.key(cfg.get("seed", 42)))
    aw_lr = optax.linear_schedule(0.0, cfg.get("aw_lr", 5e-5), cfg.get("aw_warmup", 2000))
    vqa_lr = optax.linear_schedule(0.0, cfg.get("vqa_lr", 1e-5), cfg.get("vqa_warmup", 500))
    aw_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(aw_lr))
    vqa_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(vqa_lr))
    state = {
        "step": 0,
        "aw_steps": 0,
        "vqa_steps": 0,
        "aw_opt": aw_tx.init(nnx.state(model, nnx.Param)),
        "vqa_opt": vqa_tx.init(nnx.state(model, vqa_filter())),
        "rng": rng.bit_generator.state,
    }
    if cfg.get("resume"):
        with (Path(cfg["resume"]) / "training.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        if payload["contract"] != contract:
            raise ValueError("resume training contract changed")
        state = payload["state"]
        state["aw_opt"] = jax.tree.map(jnp.asarray, state["aw_opt"])
        state["vqa_opt"] = jax.tree.map(jnp.asarray, state["vqa_opt"])
        rng.bit_generator.state = state["rng"]
    robot = RobotFrames(
        Path(cfg["dataset"]),
        model_cfg,
        norm,
        cfg["split"],
        cfg.get("tier_weights", {"base": 1.0, "precision": 2.0, "recovery": 2.0}),
        tokenizer_path=cfg.get("policy_tokenizer"),
    )
    tokenizer = VQATokenizer(VQADataConfig(**cfg.get("vqa", {})))
    vqa = VQAFrames(cfg["vqa_records"], cfg["split"], tokenizer, cfg["vqa_images"])
    batch_size, micro = cfg.get("aw_batch_size", 32), cfg.get("microbatch_size", 4)
    if batch_size % micro:
        raise ValueError("aw batch size must be divisible by microbatch size")
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    for index in range(int(state["step"]), cfg["steps"]):
        model.train()
        before = nnx.state(model, nnx.Param) if cfg.get("verify_updates", False) else None
        if index % 5 < 4:
            batch = jax.tree.map(jnp.asarray, robot.batch(rng, batch_size))
            obs = Observation.from_dict(batch)
            key = jax.random.fold_in(jax.random.key(cfg.get("seed", 42)), int(state["aw_steps"]))
            loss, state["aw_opt"] = update(
                model,
                aw_tx,
                state["aw_opt"],
                lambda m: aw_loss(
                    m,
                    key,
                    obs,
                    batch["actions"],
                    batch,
                    microbatch_size=micro,
                    wrench_weight=cfg.get("wrench_weight", 0.1),
                ),
                nnx.Param,
            )
            state["aw_steps"] += 1
            branch = "action_wrench"
        else:
            batch = jax.tree.map(jnp.asarray, vqa.batch(rng, cfg.get("vqa_batch_size", 16)))
            loss, state["vqa_opt"] = update(model, vqa_tx, state["vqa_opt"], lambda m: vqa_loss(m, batch), vqa_filter())
            state["vqa_steps"] += 1
            branch = "vqa"
        if not np.isfinite(float(loss)):
            raise FloatingPointError("nonfinite pretraining loss")
        state["step"], state["rng"] = index + 1, rng.bit_generator.state
        metrics = {"step": index + 1, "branch": branch, "loss": float(loss)}
        if before is not None:
            delta = jax.tree.map(lambda a, b: a - b, nnx.state(model, nnx.Param), before)
            metrics["parameter_delta_norm"] = float(optax.global_norm(delta))
        print(json.dumps(metrics), flush=True)
        if (index + 1) % cfg.get("save_every", 1000) == 0 or index + 1 == cfg["steps"]:
            save(output / str(index + 1), model, model_cfg, norm, state, contract, parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    run(json.loads(ap.parse_args().config.read_text()))


if __name__ == "__main__":
    main()
