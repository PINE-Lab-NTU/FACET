"""Optional JAX/OpenPI and Facet-0 server. Install the matching policy checkout separately."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_id(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def checkpoint_id(directory, backend, config_name):
    directory = Path(directory).resolve()
    paths = sorted(p for p in (directory / "params").rglob("*") if p.is_file())
    paths += sorted((directory / "assets").rglob("norm_stats.json"))
    existing = [p for p in paths if p.is_file()]
    if not (directory / "params/_METADATA").is_file() or not any(
        p.name == "norm_stats.json" for p in existing
    ):
        raise ValueError("checkpoint needs JAX params/_METADATA and normalization assets")
    data = {
        "backend": backend,
        "config_name": config_name,
        "files": [(str(p.relative_to(directory)), file_id(p)) for p in existing],
    }
    return "sha256:" + hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def install_prefix_extractor(model):
    from openpi.shared import nnx_utils

    return nnx_utils.module_jit(model.extract_contact_features)


class FrozenPolicyServer:
    def __init__(
        self,
        backend,
        checkpoint,
        *,
        config_name="",
        token_checkpoint=None,
        token_device="cpu",
        emit_prefix=False,
        guidance_scale=1.0,
        num_steps=10,
        coarse_checkpoint=None,
        wrench_envelope=None,
        coarse_hz=5.0,
        refinement_hz=20.0,
    ):
        self.backend, self.emit_prefix = backend, emit_prefix
        self.token_device, self.token = token_device, None
        if backend != "facet0":
            raise ValueError("backend must be facet0")
        from facet0.contracts import read_manifest, identity, FEATURE_SCHEMA
        from facet0.policy import load

        manifest = read_manifest(checkpoint)
        ident = "sha256:" + identity(manifest)
        self.prefix_contract = {
            "checkpoint_id": ident,
            "backend": backend,
            "prompt_branch": "untagged",
            "feature_schema": FEATURE_SCHEMA,
            "history_hz": manifest["history_hz"],
            "wire_state": "xyz_rotvec_open_gripper_raw_wrench13",
        }
        hierarchy_contract = None
        if coarse_checkpoint:
            from facet0.policy.hierarchy import load as load_hierarchy

            coarse_manifest = read_manifest(coarse_checkpoint)
            hierarchy_contract = {
                "coarse_id": identity(coarse_manifest), "refinement_id": identity(manifest),
                "wrench_envelope": np.asarray(wrench_envelope, float).tolist(),
                "coarse_hz": coarse_hz, "refinement_hz": refinement_hz,
            }
            self.policy = load_hierarchy(
                coarse_checkpoint, checkpoint, wrench_envelope=wrench_envelope, coarse_hz=coarse_hz,
                refinement_hz=refinement_hz, guidance_scale=guidance_scale, num_steps=num_steps,
            )
            ident = "sha256:" + hashlib.sha256(json.dumps(hierarchy_contract, sort_keys=True).encode()).hexdigest()
        else:
            self.policy = load(checkpoint, guidance_scale=guidance_scale, num_steps=num_steps)
        token_id = ""
        if token_checkpoint:
            import torch
            from ..token_model import RLTokenModel

            payload = torch.load(token_checkpoint, map_location="cpu", weights_only=True)
            if (
                payload.get("format") != "facet_rl_token_v1"
                or payload["prefix_contract"] != self.prefix_contract
            ):
                raise ValueError("encoder checkpoint does not match this frozen policy/prompt branch")
            self.token = RLTokenModel(**payload["model_config"]).to(token_device).eval()
            self.token.load_state_dict(payload["model"])
            self.token.requires_grad_(False)
            token_id = file_id(token_checkpoint)
        self.metadata = {
            "facet_adaptation": {
                "schema": 1,
                "frozen": True,
                "gripper": "1=open",
                "checkpoint_id": ident,
                "token_id": token_id,
                "prefix_contract": self.prefix_contract,
                "hierarchy": hierarchy_contract,
                "supports_reset": True,
            }
        }

    def _model_obs(self, obs):
        state = np.asarray(obs["observation/state"], np.float32).copy()
        if state.shape != (13,) or not np.isfinite(state).all():
            raise ValueError("expected finite 13D wire state")
        history = np.asarray(obs["observation/wrench_history"], np.float32)
        if history.shape != (10, 6) or not np.isfinite(history).all():
            raise ValueError("expected finite 10x6 measured wrench history")
        model_obs = dict(obs)
        if self.backend == "facet0":
            state[6] = 1 - state[6]
            model_obs["observation/state"] = state
            model_obs["observation/extra_view_image"] = obs["observation/right_wrist_image"]
        return model_obs

    def embeddings(self, model_obs):
        return self.policy.contact_features(model_obs)

    def infer(self, obs):
        if obs.get("facet_control") == "reset":
            self.reset()
            return {"reset": True}
        model_obs = self._model_obs(obs)
        result = self.policy.infer(model_obs)
        actions = np.asarray(result["actions"], np.float32).copy()
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("policy must produce finite Hx7 absolute Cartesian actions")
        if self.backend == "facet0":
            actions[:, 6] = 1 - np.clip(actions[:, 6], 0, 1)
        if np.any((actions[:, 6] < 0) | (actions[:, 6] > 1)):
            raise ValueError("policy wire gripper must be open-is-one in [0,1]")
        output = {"actions": actions}
        if "wrench" in result:
            output["predicted_wrench"] = np.asarray(result["wrench"], np.float32)
        if self.token is not None or self.emit_prefix:
            z, mask = self.embeddings(model_obs)
            if not np.isfinite(z).all() or not mask.any():
                raise ValueError("invalid policy prefix")
            if self.token is not None:
                import torch

                with torch.inference_mode():
                    token = self.token.encode(
                        torch.tensor(z[None], device=self.token_device),
                        torch.tensor(mask[None], device=self.token_device),
                    )[0]
                output["rl_token"] = token.cpu().numpy()
            if self.emit_prefix:
                output["prefix_embeddings"], output["prefix_mask"] = z, mask
        return output

    def reset(self):
        reset = getattr(self.policy, "reset", None)
        if reset:
            reset()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["facet0"], required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", default="", help="matching installed OpenPI training config")
    ap.add_argument("--token-checkpoint", type=Path)
    ap.add_argument("--token-device", default="cpu")
    ap.add_argument("--emit-prefix", action="store_true", help="offline encoder extraction only")
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--coarse-checkpoint", type=Path)
    ap.add_argument("--wrench-envelope", type=Path, help="JSON array of six physical-axis limits for hierarchy gating")
    ap.add_argument("--coarse-hz", type=float, default=5.0)
    ap.add_argument("--refinement-hz", type=float, default=20.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    policy = FrozenPolicyServer(
        args.backend,
        args.checkpoint,
        config_name=args.config,
        token_checkpoint=args.token_checkpoint,
        token_device=args.token_device,
        emit_prefix=args.emit_prefix,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        coarse_checkpoint=args.coarse_checkpoint,
        wrench_envelope=json.loads(args.wrench_envelope.read_text()) if args.wrench_envelope else None,
        coarse_hz=args.coarse_hz,
        refinement_hz=args.refinement_hz,
    )
    print(json.dumps(policy.metadata, indent=2), flush=True)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
