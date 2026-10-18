"""Extract frozen prefix shards from recorded episodes and train the Stage-1 bottleneck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .learner import atomic_torch_save
from .token_model import RLTokenModel


def validate_embeddings(z, mask):
    if z.ndim != 3 or mask.shape != z.shape[:2] or not np.isfinite(z).all() or not mask.any(axis=1).all():
        raise ValueError("invalid frozen embedding shard")


def extract(args, *, client=None):
    if client is None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        client = WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata().get("facet_adaptation", {})
    if not meta.get("frozen") or "prefix_contract" not in meta:
        raise ValueError("start the frozen policy server with --emit-prefix")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("use a new empty embedding output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = sorted(args.run.glob("episodes/episode_*"))
    files, z_rows, masks, total = [], [], [], 0

    def flush():
        if not z_rows:
            return
        z, mask = np.stack(z_rows).astype(np.float16), np.stack(masks)
        validate_embeddings(z, mask)
        name = f"embeddings_{len(files):06d}.npz"
        np.savez(args.output / name, z=z, pad_mask=mask)
        files.append({"path": name, "num_samples": len(z)})
        z_rows.clear()
        masks.clear()

    run_manifest = json.loads((args.run / "manifest.json").read_text())
    policy = run_manifest["config"]["policy"]
    if run_manifest["contract"]["policy"]["checkpoint_id"] != meta["checkpoint_id"]:
        raise ValueError("recorded run and extraction policy checkpoint differ")
    for ep in episodes:
        outcome_file = ep / "outcome.json"
        if not outcome_file.exists() or json.loads(outcome_file.read_text())["success"] is None:
            continue
        with np.load(ep / "steps.npz", allow_pickle=False) as data:
            states = data["state"]
            if "wrench_history" not in data:
                raise ValueError("recorded causal wrench history required")
            histories = data["wrench_history"]
        for i in range(0, len(states), args.stride):
            with np.load(ep / "images" / f"{i:06d}.npz", allow_pickle=False) as images:
                obs = {
                    "observation/state": states[i],
                    "observation/wrench_history": histories[i],
                    "observation/image": images["view1"],
                    "observation/wrist_image": images["hand"],
                    "observation/right_wrist_image": images["view2"],
                    "prompt": policy["prompt"],
                }
            result = client.infer(obs)
            if "prefix_embeddings" not in result:
                raise ValueError("server did not return prefix; enable --emit-prefix")
            z_rows.append(np.asarray(result["prefix_embeddings"]))
            masks.append(np.asarray(result["prefix_mask"], bool))
            total += 1
            if len(z_rows) == args.shard_size:
                flush()
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break
    flush()
    if total == 0:
        raise ValueError("no labelled recorded episodes with saved images")
    manifest = {
        "format": "facet_prefix_embeddings_v1",
        "prefix_contract": meta["prefix_contract"],
        "total_samples": total,
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def train(args):
    if min(args.steps, args.batch_size, args.save_every, args.layers, args.heads) < 1:
        raise ValueError("positive training sizes required")
    manifest = json.loads((args.embeddings / "manifest.json").read_text())
    if manifest["format"] != "facet_prefix_embeddings_v1" or not manifest["files"]:
        raise ValueError("invalid or incomplete embedding manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    model_config = {
        "embedding_dim": 0,
        "encoder_layers": args.layers,
        "encoder_heads": args.heads,
        "decoder_layers": args.layers,
        "decoder_heads": args.heads,
        "reconstruction_reduction": "valid_token_mean",
    }
    with np.load(args.embeddings / manifest["files"][0]["path"], allow_pickle=False) as d:
        validate_embeddings(d["z"], d["pad_mask"])
        model_config["embedding_dim"] = d["z"].shape[-1]
    if model_config["embedding_dim"] % args.heads:
        raise ValueError("embedding width must be divisible by number of attention heads")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = RLTokenModel(**model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    counts = np.asarray([r["num_samples"] for r in manifest["files"]], float)
    shard, order, at = None, [], 0
    for step in range(1, args.steps + 1):
        if at >= len(order):
            file = manifest["files"][int(rng.choice(len(counts), p=counts / counts.sum()))]
            with np.load(args.embeddings / file["path"], allow_pickle=False) as d:
                shard = {"z": d["z"].copy(), "pad_mask": d["pad_mask"].copy()}
            validate_embeddings(shard["z"], shard["pad_mask"])
            order, at = rng.permutation(len(shard["z"])), 0
        ids = order[at : at + args.batch_size]
        at += len(ids)
        z = torch.tensor(shard["z"][ids], dtype=torch.float32, device=args.device)
        mask = torch.tensor(shard["pad_mask"][ids], dtype=torch.bool, device=args.device)
        loss, _, _ = model(z, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite encoder loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        warmup = min(1.0, step / max(1, args.warmup_steps))
        for g in optimizer.param_groups:
            g["lr"] = args.lr * warmup
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "reconstruction_loss": float(loss.detach())}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            atomic_torch_save(
                {
                    "format": "facet_rl_token_v1",
                    "model": model.state_dict(),
                    "model_config": model_config,
                    "step": step,
                    "prefix_contract": manifest["prefix_contract"],
                },
                args.output / f"rl_token_step{step}.pt",
            )
    (args.output / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--host", default="127.0.0.1")
    e.add_argument("--port", type=int, default=8000)
    e.add_argument("--stride", type=int, default=5)
    e.add_argument("--shard-size", type=int, default=64)
    e.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--embeddings", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-6)
    t.add_argument("--warmup-steps", type=int, default=500)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--device", default="cuda")
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.command == "extract":
        if args.stride < 1 or args.shard_size < 1 or args.limit < 0:
            ap.error("invalid extraction sizes")
        extract(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
