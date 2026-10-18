"""RL Token encoder-decoder model (Stage 1)."""

import torch
import torch.nn as nn
from torch import Tensor


class RLTokenEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.e_rl = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, z: Tensor, pad_mask: Tensor) -> Tensor:

        B = z.shape[0]

        e_rl = self.e_rl.expand(B, -1, -1)
        tokens = torch.cat([z, e_rl], dim=1)

        rl_mask = torch.ones(B, 1, dtype=torch.bool, device=z.device)
        extended_pad_mask = torch.cat([pad_mask, rl_mask], dim=1)

        ignore_mask = ~extended_pad_mask

        out = self.transformer(tokens, src_key_padding_mask=ignore_mask)

        z_rl = out[:, -1, :]
        return z_rl


class RLTokenDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        self.h_phi = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z_rl: Tensor, z: Tensor, pad_mask: Tensor) -> Tensor:

        tgt = torch.cat([z_rl.unsqueeze(1), z[:, :-1, :]], dim=1)

        M = tgt.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M,
            device=tgt.device,
        )

        memory = z_rl.unsqueeze(1)

        tgt_key_padding_mask = ~pad_mask

        out = self.transformer(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        z_hat = self.h_phi(out)
        return z_hat


class RLTokenModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 2048,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        reconstruction_reduction: str = "valid_token_mean",
    ) -> None:
        super().__init__()
        if reconstruction_reduction not in {"sum", "valid_token_mean"}:
            raise ValueError(
                "reconstruction_reduction must be 'sum' or 'valid_token_mean', "
                f"got {reconstruction_reduction!r}"
            )
        self.reconstruction_reduction = reconstruction_reduction
        self.encoder = RLTokenEncoder(embedding_dim, encoder_layers, encoder_heads)
        self.decoder = RLTokenDecoder(embedding_dim, decoder_layers, decoder_heads)

    def forward(
        self,
        z: Tensor,
        pad_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:

        z = z.detach()

        z_rl = self.encoder(z, pad_mask)
        z_hat = self.decoder(z_rl, z, pad_mask)

        squared_norm = (z_hat - z).pow(2).sum(dim=-1)
        valid = pad_mask.float()
        per_sample = (squared_norm * valid).sum(dim=-1)
        if self.reconstruction_reduction == "valid_token_mean":
            per_sample = per_sample / valid.sum(dim=-1).clamp_min(1.0)
        loss = per_sample.mean()

        return loss, z_rl, z_hat

    @torch.no_grad()
    def encode(self, z: Tensor, pad_mask: Tensor) -> Tensor:

        return self.encoder(z, pad_mask)
