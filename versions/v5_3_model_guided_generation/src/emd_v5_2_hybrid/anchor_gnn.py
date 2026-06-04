"""Dependency-light graph neural network for V5.3 anchor prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chemistry import canonicalize_smiles
from .se3_dataset import atom_features, bond_features


def _rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install rdkit for anchor GNN graph construction") from exc
    return Chem


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install torch for anchor GNN training") from exc
    return torch, nn


class AnchorMessageGNN(_torch()[1].Module):
    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        torch, nn = _torch()
        super().__init__()
        self.is_anchor_gnn = True
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
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

    def forward(self, node_features, edge_index, edge_features):
        torch, _nn = _torch()
        h = self.node_encoder(node_features)
        if edge_features.numel():
            edge_h = self.edge_encoder(edge_features)
        else:
            edge_h = torch.empty((0, self.hidden_dim), dtype=h.dtype, device=h.device)
        for message_mlp, update_mlp, norm in zip(self.message_mlps, self.update_mlps, self.norms):
            aggregate = torch.zeros_like(h)
            degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            if edge_index.numel():
                src = edge_index[0].long()
                dst = edge_index[1].long()
                messages = message_mlp(torch.cat([h[src], edge_h], dim=-1))
                aggregate.index_add_(0, dst, messages)
                degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
                aggregate = aggregate / degree.clamp_min(1.0)
            h = norm(h + update_mlp(torch.cat([h, aggregate], dim=-1)))
        return self.head(h).view(-1)


def graph_from_mol_for_anchor_gnn(mol: Any, atom_feature_dim: int = 64) -> dict[str, Any]:
    torch, _nn = _torch()
    node_features = torch.tensor(
        [atom_features(atom, atom_feature_dim) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )
    edges: list[list[int]] = []
    edge_rows: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        features = bond_features(bond)
        edges.append([begin, end])
        edge_rows.append(features)
        edges.append([end, begin])
        edge_rows.append(features)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_rows, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_features = torch.empty((0, 6), dtype=torch.float32)
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "num_nodes": int(node_features.shape[0]),
    }


def graph_from_smiles_for_anchor_gnn(smiles: str, atom_feature_dim: int = 64) -> dict[str, Any] | None:
    Chem = _rdkit()
    canonical = canonicalize_smiles(smiles)
    if not canonical:
        return None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None
    return graph_from_mol_for_anchor_gnn(mol, atom_feature_dim=atom_feature_dim)


def collate_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _nn = _torch()
    node_rows = []
    edge_rows = []
    edge_feature_rows = []
    labels = []
    masks = []
    mol_ids: list[str] = []
    offset = 0
    for graph in graphs:
        node_features = graph["node_features"]
        edge_index = graph["edge_index"]
        edge_features = graph["edge_features"]
        node_rows.append(node_features)
        if edge_index.numel():
            edge_rows.append(edge_index + offset)
            edge_feature_rows.append(edge_features)
        label = graph.get("labels")
        mask = graph.get("label_mask")
        if label is not None:
            labels.append(label)
        if mask is not None:
            masks.append(mask)
        mol_ids.append(str(graph.get("mol_id", "")))
        offset += int(node_features.shape[0])
    output = {
        "node_features": torch.cat(node_rows, dim=0),
        "edge_index": torch.cat(edge_rows, dim=1) if edge_rows else torch.empty((2, 0), dtype=torch.long),
        "edge_features": torch.cat(edge_feature_rows, dim=0) if edge_feature_rows else torch.empty((0, 6), dtype=torch.float32),
        "mol_ids": mol_ids,
    }
    if labels:
        output["labels"] = torch.cat(labels, dim=0)
    if masks:
        output["label_mask"] = torch.cat(masks, dim=0)
    return output


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = dict(batch)
    for key in ["node_features", "edge_index", "edge_features", "labels", "label_mask"]:
        if key in moved:
            moved[key] = moved[key].to(device)
    return moved


def save_anchor_gnn_checkpoint(
    path: str | Path,
    model: AnchorMessageGNN,
    payload: dict[str, Any],
) -> Path:
    torch, _nn = _torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_type": "anchor_message_gnn",
        "model_state_dict": model.state_dict(),
        "node_dim": model.node_dim,
        "edge_dim": model.edge_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }
    data.update(payload)
    torch.save(data, output)
    return output
