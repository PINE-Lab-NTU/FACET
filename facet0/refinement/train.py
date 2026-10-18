"""Refinement trainer for pi_ref (Alg. 2, line 10; Eq. 7)."""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import pathlib
import platform
import shutil
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax
import numpy as np
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import tyro
import wandb
from flax.training import common_utils
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import sharding
from openpi.training import utils as training_utils

from facet0.policy import checkpoint as _ckpt
from facet0.policy import policy as _policy
from facet0.refinement import data as _data
from facet0.refinement import objective as _objective
from facet0.contracts import read_manifest, file_digest, identity, write_manifest


@dataclasses.dataclass
class Args:
    dataset: pathlib.Path
    credit_tag: str
    base_params: str
    exp_name: str
    norm_stats: pathlib.Path | None = None
    checkpoint_base_dir: str = "./checkpoints"
    project_name: str = "facet0_refinement"

    contact_weight: float = 1.0
    wrench_weight: float = 0.1

    batch_size: int = 32
    num_train_steps: int = 30_000
    warmup_steps: int = 2_000
    peak_lr: float = 1e-5
    ema_decay: float = 0.999
    fsdp_devices: int = 1
    num_workers: int = 8
    seed: int = 42
    log_interval: int = 10
    save_interval: int = 1_000
    keep_period: int = 5_000
    resume: bool = False
    overwrite: bool = False
    wandb_enabled: bool = False
    round_manifest: pathlib.Path | None = None


def build_config(args: Args):
    parent = pathlib.Path(args.base_params).parent
    manifest = read_manifest(parent)
    model_config = _policy.model_config(**manifest["model"])
    tokenizer = parent / "assets/policy_tokenizer.model"
    config = _config.TrainConfig(
        name="facet0_refine",
        project_name=args.project_name,
        exp_name=args.exp_name,
        model=model_config,
        weight_loader=_ckpt.ParamsLoader(args.base_params),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps, peak_lr=args.peak_lr, decay_steps=100_000, decay_lr=args.peak_lr
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=args.ema_decay,
        freeze_filter=model_config.vl_backbone_filter(),
        data=_policy.data_config_factory(
            repo_id=str(args.dataset), training=True, tokenizer_path=str(tokenizer) if tokenizer.exists() else None,
        ),
        checkpoint_base_dir=args.checkpoint_base_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
        fsdp_devices=args.fsdp_devices,
    )
    credit_path = args.dataset / "meta" / f"credit_{args.credit_tag}.parquet"
    return config, credit_path, _objective.RefinementObjective(args.contact_weight, args.wrench_weight)


def init_logging() -> None:
    level = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class Formatter(logging.Formatter):
        def format(self, record):
            record.levelname = level.get(record.levelname, record.levelname)
            return super().format(record)

    logging.basicConfig(level=logging.INFO, force=True)
    logging.getLogger().handlers[0].setFormatter(
        Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)", "%H:%M:%S"
        )
    )


def init_wandb(config: _config.TrainConfig, extra: dict[str, Any], *, resuming: bool) -> None:
    if not config.wandb_enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if resuming:
        wandb.init(id=(ckpt_dir / "wandb_id.txt").read_text().strip(), resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config) | extra, project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def init_train_state(config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(shape, mesh, log=True)
    if resume:
        return shape, state_sharding
    params_shape = shape.params.to_pure_dict()
    loaded = config.weight_loader.load(params_shape)
    loaded = jax.tree.map(
        lambda value, expected: (
            np.asarray(value, dtype=expected.dtype) if not isinstance(value, jax.ShapeDtypeStruct) else value
        ),
        loaded,
        params_shape,
    )
    at.check_pytree_equality(expected=params_shape, got=loaded, check_shapes=True, check_dtypes=True)
    partial = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(init, donate_argnums=(1,), in_shardings=replicated, out_shardings=state_sharding)(init_rng, partial)
    return state, state_sharding


def checkpoint_manager(config: _config.TrainConfig) -> tuple[ocp.CheckpointManager, bool]:

    ckpt_dir = epath.Path(config.checkpoint_dir).resolve()
    resuming = False
    if ckpt_dir.exists():
        if config.overwrite:
            ckpt_dir.rmtree()
        elif config.resume:
            resuming = True
        else:
            raise FileExistsError(f"{ckpt_dir} exists; pass --overwrite or --resume")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mngr = ocp.CheckpointManager(
        ckpt_dir,
        item_handlers={
            "assets": _checkpoints.CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1, keep_period=config.keep_period, create=False, enable_async_checkpointing=False
        ),
    )
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        resuming = False
    return mngr, resuming


def save(
    mngr: ocp.CheckpointManager, state: training_utils.TrainState, norm_stats, step: int, model_config, parent,
    tokenizer_path=None,
) -> None:
    with at.disable_typechecking():
        train_state, params = _checkpoints._split_params(state)  # noqa: SLF001
    items = {
        "assets": lambda directory: _ckpt.save_norm_stats(directory, norm_stats),
        "train_state": train_state,
        "params": {"params": params},
    }
    mngr.save(step, items)
    mngr.wait_until_finished()
    if tokenizer_path:
        shutil.copy2(tokenizer_path, pathlib.Path(mngr.directory) / str(step) / "assets/policy_tokenizer.model")
    write_manifest(pathlib.Path(mngr.directory) / str(step), model_config, stage="post_training", parent=parent)


def train_step(config, objective, rng, state: training_utils.TrainState, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    observation, actions, info = batch

    def loss_fn(model, rng, observation, actions, info):
        return _objective.refinement_loss(model, rng, observation, actions, info, objective)

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (_, metrics), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions, info
    )
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )
    return new_state, {**metrics, "grad_norm": optax.global_norm(grads)}


def main(args: Args) -> None:
    init_logging()
    logging.info("Facet-0 value-guided refinement on %s", platform.node())
    config, credit_path, objective = build_config(args)
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch size {config.batch_size} must be divisible by {jax.device_count()} devices")
    import json

    parent_manifest = read_manifest(pathlib.Path(args.base_params).parent)
    stats_dir = args.norm_stats or pathlib.Path(args.base_params).parent / "assets"
    if file_digest(stats_dir / "norm_stats.json") != parent_manifest["norm_sha256"]:
        raise ValueError("post-training normalization must match the parent policy")
    norm_stats = _ckpt.load_norm_stats(stats_dir)
    logging.info("Loaded norm stats from %s", stats_dir / "norm_stats.json")

    jax.config.update("jax_compilation_cache_dir", os.environ.get(
        "JAX_COMPILATION_CACHE_DIR", str(pathlib.Path(args.checkpoint_base_dir).resolve() / ".jax_cache")
    ))
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    mngr, resuming = checkpoint_manager(config)
    init_wandb(config, {"credit": str(credit_path), "objective": dataclasses.asdict(objective)}, resuming=resuming)

    corpora = json.loads(args.round_manifest.read_text())["corpora"] if args.round_manifest else None
    loader = _data.create_data_loader(
        config, credit_path, norm_stats, sharding=data_sharding, shuffle=True, corpora=corpora
    )
    batches = iter(loader)
    batch = next(batches)
    logging.info("Batch:\n%s", training_utils.array_tree_to_info(batch))

    state, state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(state)
    if resuming:
        state = _checkpoints.restore_state(mngr, state, loader)

    step_fn = jax.jit(
        functools.partial(train_step, config, objective),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    start = int(state.step)
    pbar = tqdm.tqdm(
        range(start, config.num_train_steps), initial=start, total=config.num_train_steps, dynamic_ncols=True
    )
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, batch)
        infos.append(info)
        if step % config.log_interval == 0 or step + 1 == config.num_train_steps:
            reduced = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
            record = {"step": step + 1, **{key: float(value) for key, value in reduced.items()}}
            if not all(np.isfinite(value) for value in record.values()):
                raise FloatingPointError("nonfinite refinement optimization metrics")
            with (pathlib.Path(config.checkpoint_dir) / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            wandb.log(reduced, step=step)
            infos = []
        batch = next(batches)
        completed = step + 1
        if completed % config.save_interval == 0 or completed == config.num_train_steps:
            save(mngr, state, norm_stats, completed, config.model, identity(parent_manifest), config.data.tokenizer_path)
    mngr.wait_until_finished()


if __name__ == "__main__":
    main(tyro.cli(Args))
