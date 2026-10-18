"""Serve a Facet-0 policy over the openpi websocket protocol."""

from __future__ import annotations

import argparse
import logging

from openpi.serving import websocket_policy_server

from facet0.policy.policy import load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--guidance-scale", type=float, default=1.0, help="0 = unconditional, 1 = guided branch")
    ap.add_argument("--default-prompt", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    policy = load(args.model_dir, args.guidance_scale, default_prompt=args.default_prompt)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
