"""Checkpoint parameter and normalization-statistics I/O."""

from __future__ import annotations

import dataclasses
import pathlib

import flax.traverse_util as traverse_util
import jax
import numpy as np
import openpi.shared.array_typing as at
import openpi.shared.normalize as _normalize
import orbax.checkpoint as ocp
from openpi.training import weight_loaders as _weight_loaders


def restore_params(params_path: pathlib.Path | str, *, restore_type=jax.Array, dtype=None, sharding=None) -> at.Params:
    """``openpi.models.model.restore_params`` that also reads orbax >= 0.11 metadata."""
    params_path = pathlib.Path(params_path).resolve()
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        tree = getattr(metadata, "item_metadata", metadata)  # StepMetadata on orbax >= 0.11
        item = {"params": tree["params"]}
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]
    flat = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat):  # saved from an nnx.State during training
        flat = {kp[:-1]: v for kp, v in flat.items()}
    return traverse_util.unflatten_dict(flat)


@dataclasses.dataclass(frozen=True)
class ParamsLoader(_weight_loaders.WeightLoader):
    """Initializes post-training from a policy checkpoint's ``params/`` directory."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = restore_params(self.params_path, restore_type=np.ndarray)
        return _weight_loaders._merge_params(loaded, params, missing_regex=".*lora.*")  # noqa: SLF001


def load_norm_stats(directory: pathlib.Path | str) -> dict[str, _normalize.NormStats]:
    """Read ``<directory>/norm_stats.json`` directly; resolving via ``asset_id`` can pick up another corpus's."""
    return _normalize.load(directory)


def save_norm_stats(directory: pathlib.Path | str, norm_stats: dict[str, _normalize.NormStats]) -> None:
    _normalize.save(directory, norm_stats)
