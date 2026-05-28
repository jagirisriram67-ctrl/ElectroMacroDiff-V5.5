"""V5.5 pocket validity reward model.

The model is a lightweight binary classifier trained from generation attempt
logs. It predicts whether a proposed anchor/linker/chemotype attempt is likely
to become a valid macrocycle before RDKit construction and docking spend time
on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .pocket_features import POCKET_FEATURE_DIM


REWARD_FEATURE_COLUMNS = [
    "anchor_score_a",
    "anchor_score_b",
    "anchor_pair_product",
    "linker_length_norm",
    "linker_probability",
    "target_ring_norm",
    "path_bonds_norm",
    "linker_density",
    "chemotype_alkyl",
    "chemotype_oxa",
    "chemotype_aza",
    "chemotype_thioether",
    "chemotype_oxa_aza",
    "chemotype_dioxa",
    "chemotype_diaza",
    "chemotype_dithio",
    "chemotype_aza_oxa",
    "chemotype_oxa_thio",
    "chemotype_aza_thio",
]

CHEMOTYPE_LIST = [
    "alkyl",
    "oxa",
    "aza",
    "thioether",
    "oxa_aza",
    "dioxa",
    "diaza",
    "dithio",
    "aza_oxa",
    "oxa_thio",
    "aza_thio",
]

REWARD_INPUT_DIM = len(REWARD_FEATURE_COLUMNS) + POCKET_FEATURE_DIM


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install torch for PocketValidityRewardModel") from exc
    return torch, nn


class PocketValidityRewardModel(_torch()[1].Module):
    """Binary classifier returning logits for P(valid macrocycle attempt)."""

    def __init__(
        self,
        input_dim: int = REWARD_INPUT_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.15,
    ):
        _torch_mod, nn = _torch()
        super().__init__()
        self.is_validity_reward = True
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        return self.net(features).view(-1)


def attempt_to_feature_vector(attempt: dict[str, Any], pocket_vector: list[float]) -> list[float]:
    """Convert one attempt-log row plus pocket vector to a reward input row."""
    anchor_a = _safe_float(attempt.get("anchor_score_a", 0))
    anchor_b = _safe_float(attempt.get("anchor_score_b", 0))
    linker_len = _safe_float(attempt.get("linker_length", 0))
    target_ring = _safe_float(attempt.get("target_ring", 0))
    path_bonds = _safe_float(attempt.get("path_bonds", 0))
    linker_prob = _safe_float(
        attempt.get(
            "linker_probability",
            attempt.get("policy_probability", attempt.get("size_probability", 0)),
        )
    )
    chemotype = str(attempt.get("chemotype", "")).lower()
    chemotype_onehot = [float(chemotype == name) for name in CHEMOTYPE_LIST]
    features = [
        anchor_a,
        anchor_b,
        anchor_a * anchor_b,
        linker_len / 12.0,
        linker_prob,
        target_ring / 24.0,
        path_bonds / 24.0,
        linker_len / max(target_ring, 1.0),
    ] + chemotype_onehot
    features.extend(float(v) for v in pocket_vector)
    if len(features) != REWARD_INPUT_DIM:
        raise ValueError(f"Expected reward input dim {REWARD_INPUT_DIM}, got {len(features)}")
    return features


def attempt_label(status: str) -> float:
    return 1.0 if str(status).strip() == "valid_output" else 0.0


def save_validity_reward_checkpoint(
    path: str | Path,
    model: PocketValidityRewardModel,
    payload: dict[str, Any],
) -> Path:
    torch, _nn = _torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_type": "pocket_validity_reward",
        "model_state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout_rate,
        "feature_columns": REWARD_FEATURE_COLUMNS,
    }
    data.update(payload)
    torch.save(data, output)
    return output


def load_validity_reward(path: str | Path, device: str = "cpu") -> tuple:
    torch, _nn = _torch()
    checkpoint = torch.load(path, map_location=device)
    model = PocketValidityRewardModel(
        input_dim=int(checkpoint.get("input_dim", REWARD_INPUT_DIM)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint.get("dropout", 0.15)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default
