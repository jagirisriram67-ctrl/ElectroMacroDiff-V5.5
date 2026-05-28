"""V5.5 Pocket-Conditioned Anchor GNN.

Extends the V5.3 AnchorMessageGNN by concatenating a pocket electronic
feature vector to every node before message passing.  This means anchor
predictions are no longer blind to the protein — they know what residues
are in the binding pocket, what charges are present, and what interaction
types the pocket prefers.

Architecture:
    node_features (64) + pocket_vector (32) → project → GNN layers → head → P(anchor)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pocket_features import POCKET_FEATURE_DIM


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install torch for PocketAnchorGNN") from exc
    return torch, nn


class PocketAnchorGNN(_torch()[1].Module):
    """Graph neural network for anchor prediction conditioned on pocket electronics."""

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 6,
        pocket_dim: int = POCKET_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        torch, nn = _torch()
        super().__init__()
        self.is_pocket_anchor_gnn = True
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.pocket_dim = int(pocket_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)

        # Project concatenated [node_feat | pocket_vector] to hidden
        self.node_encoder = nn.Linear(node_dim + pocket_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.message_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.message_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.update_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_features, edge_index, edge_features, pocket_vector):
        """Forward pass.

        Parameters
        ----------
        node_features : Tensor [N, node_dim]
        edge_index : Tensor [2, E]
        edge_features : Tensor [E, edge_dim]
        pocket_vector : Tensor [pocket_dim]  (broadcast to all nodes)
        """
        torch, _nn = _torch()

        # Broadcast pocket vector to every node
        n_atoms = node_features.shape[0]
        pocket_expanded = pocket_vector.unsqueeze(0).expand(n_atoms, -1)
        h = self.node_encoder(torch.cat([node_features, pocket_expanded], dim=-1))

        if edge_features.numel():
            edge_h = self.edge_encoder(edge_features)
        else:
            edge_h = torch.empty((0, self.hidden_dim), dtype=h.dtype, device=h.device)

        for msg_mlp, upd_mlp, norm in zip(self.message_mlps, self.update_mlps, self.norms):
            aggregate = torch.zeros_like(h)
            degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)

            if edge_index.numel():
                src = edge_index[0].long()
                dst = edge_index[1].long()
                messages = msg_mlp(torch.cat([h[src], edge_h], dim=-1))
                aggregate.index_add_(0, dst, messages)
                degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
                aggregate = aggregate / degree.clamp_min(1.0)

            h = norm(h + upd_mlp(torch.cat([h, aggregate], dim=-1)))

        return self.head(h).view(-1)


def save_pocket_anchor_gnn_checkpoint(
    path: str | Path,
    model: PocketAnchorGNN,
    payload: dict[str, Any],
) -> Path:
    """Save a PocketAnchorGNN checkpoint with architecture metadata."""
    torch, _nn = _torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_type": "pocket_anchor_gnn",
        "model_state_dict": model.state_dict(),
        "node_dim": model.node_dim,
        "edge_dim": model.edge_dim,
        "pocket_dim": model.pocket_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout_rate,
    }
    data.update(payload)
    torch.save(data, output)
    return output


def load_pocket_anchor_gnn(path: str | Path, device: str = "cpu") -> tuple:
    """Load a PocketAnchorGNN from checkpoint.

    Returns (model, checkpoint_dict).
    """
    torch, _nn = _torch()
    checkpoint = torch.load(path, map_location=device)
    model = PocketAnchorGNN(
        node_dim=int(checkpoint.get("node_dim", 64)),
        edge_dim=int(checkpoint.get("edge_dim", 6)),
        pocket_dim=int(checkpoint.get("pocket_dim", POCKET_FEATURE_DIM)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint.get("num_layers", 4)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
