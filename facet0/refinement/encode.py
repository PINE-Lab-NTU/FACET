"""Encode stored rollouts with pi_ref into h^c and joint proposals for the critic."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
import pandas as pd
import torch
from openpi.training import data_loader as _data_loader

from facet0.data import frames as _frames
from facet0.data import norm_stats as _norm_stats
from facet0.model import Facet0Config, Facet0Model
from facet0.policy import checkpoint, guidance
from facet0.policy import policy as _policy

logger = logging.getLogger("facet0.encode")


def _collate(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items)


def model_input_dataset(root: Path, model_config: Facet0Config, norm_stats: dict):
    data_config = _policy.data_config_factory().create(Path("assets"), model_config)
    return _data_loader.TransformedDataset(
        _frames.CorpusFrames(root, model_config.action_horizon, model_config.wrench_window),
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *guidance.swap_tokenizer(list(data_config.model_transforms.inputs)),
        ],
    )


def encode(
    model: Facet0Model,
    root: Path,
    norm_stats_dir: Path,
    out: Path,
    *,
    num_proposals: int = 1,
    guidance_scale: float = 1.0,
    num_steps: int = 10,
    batch_size: int = 32,
    num_workers: int = 8,
    limit: int | None = None,
    seed: int = 0,
    params_path: str = "",
) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    norm_stats = checkpoint.load_norm_stats(norm_stats_dir)
    config = Facet0Config(action_horizon=model.action_horizon, wrench_window=model.wrench_window)
    dataset = model_input_dataset(root, config, norm_stats)
    frames = dataset._dataset  # noqa: SLF001
    n = len(dataset) if limit is None else min(limit, len(dataset))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, range(n)), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_collate,
    )
    graphdef, state = nnx.split(model)

    @jax.jit
    def run(state, rng, observation, guided, guided_mask, noise):
        return guidance.sample_with_guidance(
            nnx.merge(graphdef, state), rng, observation, guided, guided_mask,
            num_steps=num_steps, guidance_scale=guidance_scale, noise=noise,
        )

    rep_dim = 2 * model.action_in_proj.out_features
    hc = np.lib.format.open_memmap(out / "hc.npy", mode="w+", dtype=np.float16, shape=(n, rep_dim))
    props = np.lib.format.open_memmap(
        out / "proposals.npy", mode="w+", dtype=np.float16,
        shape=(n, num_proposals, model.action_horizon, model.action_dim),
    )
    rng = jax.random.key(seed)
    row = 0
    for batch in loader:
        b = len(batch["state"])
        guided, guided_mask = jnp.asarray(batch.pop("guided_prompt")), jnp.asarray(batch.pop("guided_prompt_mask"))
        observation = _model.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
        for m in range(num_proposals):
            rng, key = jax.random.split(rng)
            noise = jax.random.normal(key, (b, model.action_horizon, model.action_dim))
            y, h = run(state, key, observation, guided, guided_mask, noise)
            props[row : row + b, m] = np.asarray(y, np.float16)
            if m == 0:
                hc[row : row + b] = np.asarray(h, np.float16)
        row += b
        logger.info("encoded %d / %d frames", row, n)
    hc.flush()
    props.flush()
    pd.DataFrame({"episode_index": frames.episode[:n], "frame_index": frames.frame[:n]}).to_parquet(
        out / "index.parquet", index=False
    )
    meta = {
        "dataset": str(root), "params": params_path, "norm_stats": str(norm_stats_dir), "frames": n,
        "num_proposals": num_proposals, "guidance_scale": guidance_scale, "num_steps": num_steps,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="Encode rollouts with pi_ref: h^c and proposals per frame.")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--params", required=True, help="params/ directory of pi_ref")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--norm-stats", type=Path, default=None, help="default: <dataset>/facet0")
    ap.add_argument("--num-proposals", type=int, default=1)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--paligemma-variant", default="gemma_2b")
    ap.add_argument("--action-expert-variant", default="gemma_300m")
    args = ap.parse_args()
    config = Facet0Config(paligemma_variant=args.paligemma_variant, action_expert_variant=args.action_expert_variant)
    model = config.load(checkpoint.restore_params(args.params, dtype=jnp.bfloat16))
    encode(
        model, args.dataset, args.norm_stats or _norm_stats.stats_dir(args.dataset), args.out,
        num_proposals=args.num_proposals, guidance_scale=args.guidance_scale, num_steps=args.num_steps,
        batch_size=args.batch_size, num_workers=args.num_workers, limit=args.limit, params_path=args.params,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
