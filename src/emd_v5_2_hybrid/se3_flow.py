"""A compact SE(3)-aware flow-matching starter model for Colab debugging.

The coordinate head predicts a vector field from relative coordinate vectors
weighted by invariant scalar messages. That construction is translation
equivariant and rotation equivariant for the coordinate update, which is enough
for a defensible sprint prototype before heavier e3nn blocks are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass


def _torch_modules():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install torch to use the SE(3) flow model") from exc
    return torch, nn, functional


@dataclass(frozen=True)
class SE3FlowConfig:
    atom_feature_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.05


def sinusoidal_time_embedding(time_t, dim: int):
    torch, _nn, _functional = _torch_modules()
    half = dim // 2
    frequencies = torch.exp(
        torch.linspace(0, -8, half, device=time_t.device, dtype=time_t.dtype)
    )
    angles = time_t[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SE3FlowMatching(_torch_modules()[1].Module):
    """Small fully-connected molecular coordinate flow model."""

    def __init__(self, config: SE3FlowConfig | None = None, **kwargs):
        torch, nn, _functional = _torch_modules()
        super().__init__()
        if config is None:
            config = SE3FlowConfig(**kwargs)
        self.config = config
        hidden = config.hidden_dim
        self.atom_encoder = nn.Sequential(
            nn.Linear(config.atom_feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2 + 2, hidden),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(config.num_layers)
            ]
        )
        self.node_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(config.num_layers)
            ]
        )
        self.coord_scale = nn.Parameter(torch.tensor(0.05))
        self.velocity_norm = nn.LayerNorm(hidden)

    def forward(self, atom_features, coords_t, time_t, mask):
        torch, _nn, functional = _torch_modules()
        h = self.atom_encoder(atom_features)
        t_emb = self.time_encoder(sinusoidal_time_embedding(time_t, h.shape[-1]))
        h = h + t_emb[:, None, :]

        pair_mask = mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(mask.shape[1], device=mask.device, dtype=mask.dtype)[None, :, :]
        pair_mask = pair_mask * (1.0 - eye)

        coords = coords_t
        velocity = torch.zeros_like(coords_t)
        for edge_mlp, node_mlp in zip(self.edge_mlps, self.node_mlps):
            diff = coords[:, :, None, :] - coords[:, None, :, :]
            dist = torch.sqrt((diff.square().sum(dim=-1, keepdim=True) + 1e-8))
            hi = h[:, :, None, :].expand(-1, -1, h.shape[1], -1)
            hj = h[:, None, :, :].expand(-1, h.shape[1], -1, -1)
            t_pair = time_t[:, None, None, None].expand(-1, h.shape[1], h.shape[1], 1)
            edge_input = torch.cat([hi, hj, dist, t_pair], dim=-1)
            weights = edge_mlp(edge_input).squeeze(-1) * pair_mask
            weights = torch.tanh(weights)
            vector_message = (weights[..., None] * diff).sum(dim=2)
            scalar_message = (weights[..., None] * hj).sum(dim=2)
            h_update = node_mlp(torch.cat([h, scalar_message], dim=-1))
            h = self.velocity_norm(h + h_update)
            velocity = velocity + vector_message * self.coord_scale

        velocity = velocity * mask[..., None]
        return velocity


def flow_matching_loss(model, batch):
    torch, _nn, functional = _torch_modules()
    atom_features = batch["atom_features"]
    coords_1 = batch["coords"]
    mask = batch["mask"]
    coords_0 = torch.randn_like(coords_1)
    time_t = torch.rand(coords_1.shape[0], device=coords_1.device)
    coords_t = (1.0 - time_t[:, None, None]) * coords_0 + time_t[:, None, None] * coords_1
    target_velocity = coords_1 - coords_0
    pred_velocity = model(atom_features, coords_t, time_t, mask)
    mse = (pred_velocity - target_velocity).square().sum(dim=-1)
    loss = (mse * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


def sample_coordinates(model, atom_features, mask, steps: int = 40):
    torch, _nn, _functional = _torch_modules()
    coords = torch.randn((*atom_features.shape[:2], 3), device=atom_features.device)
    dt = 1.0 / max(steps, 1)
    for step in range(steps):
        t = torch.full((atom_features.shape[0],), step / max(steps - 1, 1), device=atom_features.device)
        velocity = model(atom_features, coords, t, mask)
        coords = coords + dt * velocity
        coords = coords * mask[..., None]
    return coords
