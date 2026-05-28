"""V5.5 Pocket-Conditioned Linker Policy Network.

Predicts both linker LENGTH and linker CHEMOTYPE conditioned on:
  - the selected anchor pair features
  - the seed molecule graph summary
  - the JAK2 pocket electronic feature vector

This replaces the V5.3 linker-size-only MLP with a dual-head policy
that can predict the best chemotype for each anchor pair given the
pocket's electronic environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pocket_features import POCKET_FEATURE_DIM


LINKER_SIZE_LABELS = list(range(3, 13))  # lengths 3-12
CHEMOTYPE_LABELS = [
    "alkyl", "oxa", "aza", "thioether", "oxa_aza",
    "dioxa", "diaza", "dithio", "aza_oxa", "oxa_thio", "aza_thio",
]


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install torch for PocketLinkerPolicy") from exc
    return torch, nn


class PocketLinkerPolicy(_torch()[1].Module):
    """Dual-head policy: predicts linker size AND chemotype.

    Input features:
        linker_features (16) + pocket_vector (32) = 48

    Output heads:
        - size_head → P(linker_length) over 10 classes
        - chemotype_head → P(chemotype) over 11 classes
    """

    def __init__(
        self,
        linker_feature_dim: int = 16,
        pocket_dim: int = POCKET_FEATURE_DIM,
        hidden_dim: int = 192,
        num_size_classes: int = 10,
        num_chemotype_classes: int = 11,
        dropout: float = 0.1,
    ):
        torch, nn = _torch()
        super().__init__()
        self.is_pocket_linker_policy = True
        self.linker_feature_dim = int(linker_feature_dim)
        self.pocket_dim = int(pocket_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_size_classes = int(num_size_classes)
        self.num_chemotype_classes = int(num_chemotype_classes)
        self.dropout_rate = float(dropout)

        input_dim = linker_feature_dim + pocket_dim

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Size head — predicts linker length (3-12)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_size_classes),
        )

        # Chemotype head — predicts linker chemotype (11 types)
        self.chemotype_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_chemotype_classes),
        )

    def forward(self, linker_features, pocket_vector):
        """Forward pass.

        Parameters
        ----------
        linker_features : Tensor [B, linker_feature_dim]
        pocket_vector : Tensor [B, pocket_dim] or [pocket_dim]

        Returns
        -------
        size_logits : Tensor [B, num_size_classes]
        chemotype_logits : Tensor [B, num_chemotype_classes]
        """
        torch, _nn = _torch()

        if pocket_vector.dim() == 1:
            pocket_vector = pocket_vector.unsqueeze(0).expand(linker_features.shape[0], -1)

        x = torch.cat([linker_features, pocket_vector], dim=-1)
        h = self.trunk(x)
        return self.size_head(h), self.chemotype_head(h)


def save_pocket_linker_policy_checkpoint(
    path: str | Path,
    model: PocketLinkerPolicy,
    payload: dict[str, Any],
) -> Path:
    torch, _nn = _torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_type": "pocket_linker_policy",
        "model_state_dict": model.state_dict(),
        "linker_feature_dim": model.linker_feature_dim,
        "pocket_dim": model.pocket_dim,
        "hidden_dim": model.hidden_dim,
        "num_size_classes": model.num_size_classes,
        "num_chemotype_classes": model.num_chemotype_classes,
        "dropout": model.dropout_rate,
        "size_labels": LINKER_SIZE_LABELS,
        "chemotype_labels": CHEMOTYPE_LABELS,
    }
    data.update(payload)
    torch.save(data, output)
    return output


def load_pocket_linker_policy(path: str | Path, device: str = "cpu") -> tuple:
    """Load a PocketLinkerPolicy from checkpoint.

    Returns (model, size_labels, chemotype_labels, checkpoint_dict).
    """
    torch, _nn = _torch()
    checkpoint = torch.load(path, map_location=device)
    model = PocketLinkerPolicy(
        linker_feature_dim=int(checkpoint.get("linker_feature_dim", 16)),
        pocket_dim=int(checkpoint.get("pocket_dim", POCKET_FEATURE_DIM)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_size_classes=int(checkpoint.get("num_size_classes", 10)),
        num_chemotype_classes=int(checkpoint.get("num_chemotype_classes", 11)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    size_labels = [int(v) for v in checkpoint.get("size_labels", LINKER_SIZE_LABELS)]
    chemotype_labels = list(checkpoint.get("chemotype_labels", CHEMOTYPE_LABELS))
    return model, size_labels, chemotype_labels, checkpoint
