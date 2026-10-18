import json
from types import SimpleNamespace

import numpy as np
import torch

from facet_adaptation.encoder import train
from facet_adaptation.token_model import RLTokenModel


def test_encoder_training_save_reload_and_prefix_identity(tmp_path):
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    z = np.random.default_rng(0).normal(size=(3, 5, 16)).astype(np.float16)
    np.savez(embeddings / "shard.npz", z=z, pad_mask=np.ones((3, 5), bool))
    contract = {"checkpoint_id": "test-policy", "prompt_branch": "raw"}
    (embeddings / "manifest.json").write_text(
        json.dumps(
            {
                "format": "facet_prefix_embeddings_v1",
                "prefix_contract": contract,
                "files": [{"path": "shard.npz", "num_samples": 3}],
            }
        )
    )
    out = tmp_path / "encoder"
    train(
        SimpleNamespace(
            embeddings=embeddings,
            output=out,
            steps=2,
            batch_size=2,
            lr=1e-6,
            warmup_steps=1,
            save_every=2,
            layers=1,
            heads=2,
            device="cpu",
            seed=42,
        )
    )
    payload = torch.load(out / "rl_token_step2.pt", weights_only=True)
    model = RLTokenModel(**payload["model_config"])
    model.load_state_dict(payload["model"])
    model.eval()
    token = model.encode(torch.tensor(z[:1], dtype=torch.float32), torch.ones(1, 5, dtype=torch.bool))
    assert token.shape == (1, 16) and torch.isfinite(token).all() and torch.any(token)
    assert payload["prefix_contract"] == contract and payload["step"] == 2
