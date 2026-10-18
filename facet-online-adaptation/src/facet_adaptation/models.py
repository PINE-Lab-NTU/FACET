"""Bounded action and wrench prediction networks."""

import copy
import torch
from torch import nn
from .networks import MLP, ModalityProjection


class MultiModalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.register_buffer("mean", torch.tensor(config.proprio_mean))
        self.register_buffer("std", torch.tensor(config.proprio_std))
        self.z_proj = ModalityProjection(config.embedding_dim, config.projection_dim)
        self.state_proj = ModalityProjection(7, config.projection_dim)
        self.wrench_proj = ModalityProjection(6, config.projection_dim)
        self.action_proj = ModalityProjection(7, config.projection_dim)
        self.output_dim = 4 * config.projection_dim

    def forward(self, x, reference):
        z, p = x[..., : self.embedding_dim], x[..., self.embedding_dim :]
        p = (p - self.mean) / self.std
        return torch.cat(
            [
                self.z_proj(z),
                self.state_proj(p[..., :7]),
                self.wrench_proj(p[..., 7:]),
                self.action_proj(reference),
            ],
            -1,
        )


class Actor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ref_dropout = config.reference_dropout
        self.encoder = MultiModalEncoder(config)
        self.mlp = MLP(self.encoder.output_dim, 13, config.hidden_dim, config.hidden_layers)
        self.wrench_head = nn.Sequential(
            nn.Linear(6 + 7, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 6)
        )
        self.register_buffer("minimum", torch.tensor(config.action_min))
        self.register_buffer("maximum", torch.tensor(config.action_max))
        self.register_buffer("mask", torch.tensor(config.action_mask))

    def outputs(self, x, reference, *, reference_dropout=None, executed_action=None):
        dropout = self.training if reference_dropout is None else reference_dropout
        conditioning = reference
        if dropout and self.ref_dropout:
            conditioning = reference * (
                torch.rand(reference.shape[0], 1, device=reference.device) >= self.ref_dropout
            )
        raw = self.mlp(self.encoder(x, conditioning))
        action = self.minimum + (torch.tanh(raw[..., :7]) + 1) * 0.5 * (self.maximum - self.minimum)
        action = torch.where(self.mask, action, reference)
        # Eq. 9 consequence supervision is tied to the command that caused w[t+1].
        consequence_action = action if executed_action is None else executed_action
        scaled = 2 * (consequence_action - self.minimum) / (self.maximum - self.minimum) - 1
        wrench = self.wrench_head(torch.cat([raw[..., 7:], scaled], dim=-1))
        return action, wrench

    def forward(self, x, reference, *, reference_dropout=None):
        return self.outputs(x, reference, reference_dropout=reference_dropout)[0]


class QNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MultiModalEncoder(config)
        self.executed_proj = ModalityProjection(7, config.projection_dim)
        self.mlp = MLP(5 * config.projection_dim, 1, config.hidden_dim, config.hidden_layers)

    def forward(self, x, action, reference):
        return self.mlp(torch.cat([self.encoder(x, reference), self.executed_proj(action)], -1))


class TwinQCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q1, self.q2 = QNetwork(config), QNetwork(config)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)

    def forward(self, x, action, reference):
        return self.q1(x, action, reference), self.q2(x, action, reference)

    def q_min(self, x, action, reference):
        return torch.minimum(*self(x, action, reference))

    def target_q_min(self, x, action, reference):
        return torch.minimum(self.q1_target(x, action, reference), self.q2_target(x, action, reference))

    @torch.no_grad()
    def update_targets(self, tau):
        for source, target in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
            for a, b in zip(source.parameters(), target.parameters(), strict=True):
                b.lerp_(a, tau)
