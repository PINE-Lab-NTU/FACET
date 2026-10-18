"""Initialize a joint policy from a base or force-aware parameter checkpoint."""

import argparse
import json
from pathlib import Path

import flax.nnx as nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import numpy as np
import orbax.checkpoint as ocp

from facet0.contracts import write_manifest
from facet0.model import Facet0Config
from facet0.policy.checkpoint import restore_params, load_norm_stats, save_norm_stats


def convert(reference, loaded):
    ref, old = flatten_dict(reference), flatten_dict(loaded)
    result, report = {}, {"copied": [], "initialized": [], "converted": []}
    for key, value in ref.items():
        name = "/".join(key)
        source = old.get(key)
        if key[0] in {"action_in_proj", "action_out_proj"} and source is not None and source.shape != value.shape:
            out = np.array(value, copy=True)
            if key[0] == "action_in_proj" and key[-1] == "kernel":
                out[:7] = source[:7]
            elif key[0] == "action_out_proj" and key[-1] == "kernel":
                out[..., :7] = source[..., :7]
            elif key[0] == "action_out_proj" and key[-1] == "bias":
                out[:7] = source[:7]
            else:
                raise ValueError(f"unsupported projection conversion: {name}")
            result[key] = out
            report["converted"].append(name)
        elif source is not None and source.shape == value.shape:
            result[key] = source
            report["copied"].append(name)
        elif key[0] in {"state_encoder", "wrench_encoder_in", "wrench_encoder_out"}:
            result[key] = value
            report["initialized"].append(name)
        else:
            raise ValueError(f"missing or incompatible parameter: {name}")
    return unflatten_dict(result), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--norm-stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = Facet0Config()
    model = cfg.create(jax.random.key(42))
    params, report = convert(nnx.state(model).to_pure_dict(), restore_params(args.params, restore_type=np.ndarray))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(args.out / "params", {"params": params})
    save_norm_stats(args.out / "assets", load_norm_stats(args.norm_stats))
    (args.out / "initialization.json").write_text(json.dumps(report, indent=2) + "\n")
    write_manifest(args.out, cfg, stage="initialization", extra={"source": str(args.params), "conversion": report})


if __name__ == "__main__":
    main()
