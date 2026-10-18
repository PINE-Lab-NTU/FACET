from types import SimpleNamespace
import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import optax
from openpi.shared.normalize import NormStats
from openpi.training.utils import TrainState
from facet0.model import Facet0Config
from facet0.refinement.train import checkpoint_manager, save
from facet0.contracts import read_manifest
from facet0.policy.checkpoint import restore_params


def test_refinement_checkpoint_manager_writes_manifest(tmp_path):
    cfg = SimpleNamespace(checkpoint_dir=tmp_path / "run", overwrite=False, resume=False, keep_period=5)
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    graph, params = nnx.split(model)
    tx = optax.adam(0.01)
    state = TrainState(
        step=jnp.array(1),
        params=params,
        model_def=graph,
        opt_state=tx.init(params),
        tx=tx,
        ema_decay=None,
        ema_params=None,
    )
    norm = {
        key: NormStats(mean=np.zeros(n), std=np.ones(n), q01=-np.ones(n), q99=np.ones(n))
        for key, n in (("state", 67), ("actions", 13))
    }
    manager, resumed = checkpoint_manager(cfg)
    try:
        save(manager, state, norm, 1, Facet0Config(), "parent-id")
    finally:
        manager.close()
    manifest = read_manifest(tmp_path / "run/1")
    assert manifest["parent"] == "parent-id" and manifest["stage"] == "post_training"
    restored = restore_params(tmp_path / "run/1/params")
    np.testing.assert_allclose(restored["kernel"], params.to_pure_dict()["kernel"])
