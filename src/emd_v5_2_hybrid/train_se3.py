"""Training and debug utilities for the SE(3) flow model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .se3_flow import SE3FlowConfig, SE3FlowMatching, flow_matching_loss


def _torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install torch to train the SE(3) model") from exc
    return torch, DataLoader, Dataset


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas for training logs") from exc
    return pd


class GraphListDataset(_torch()[2]):
    def __init__(self, graphs: list[dict[str, Any]]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.graphs[index]


def load_graphs(path: str | Path) -> list[dict[str, Any]]:
    torch, _DataLoader, _Dataset = _torch()
    graphs = torch.load(path, map_location="cpu")
    if not isinstance(graphs, list):
        raise ValueError(f"Expected list of graph dictionaries in {path}")
    return graphs


def collate_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _DataLoader, _Dataset = _torch()
    if not graphs:
        raise ValueError("Cannot collate an empty graph batch")
    max_atoms = max(int(graph["coords"].shape[0]) for graph in graphs)
    atom_dim = int(graphs[0]["atom_features"].shape[-1])
    batch_size = len(graphs)
    atom_features = torch.zeros((batch_size, max_atoms, atom_dim), dtype=torch.float32)
    coords = torch.zeros((batch_size, max_atoms, 3), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_atoms), dtype=torch.float32)
    mol_ids: list[str] = []
    smiles: list[str] = []
    for index, graph in enumerate(graphs):
        n_atoms = int(graph["coords"].shape[0])
        atom_features[index, :n_atoms] = graph["atom_features"].float()
        coords[index, :n_atoms] = graph["coords"].float()
        mask[index, :n_atoms] = 1.0
        mol_ids.append(str(graph.get("mol_id", "")))
        smiles.append(str(graph.get("smiles", "")))
    coords = coords - ((coords * mask[..., None]).sum(dim=1, keepdim=True) / mask.sum(dim=1)[:, None, None])
    return {"atom_features": atom_features, "coords": coords, "mask": mask, "mol_ids": mol_ids, "smiles": smiles}


def make_dataloader(graphs: list[dict[str, Any]], batch_size: int = 4, shuffle: bool = True):
    _torch_mod, DataLoader, _Dataset = _torch()
    return DataLoader(
        GraphListDataset(graphs),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graphs,
    )


def move_batch(batch: dict[str, Any], device: str):
    torch, _DataLoader, _Dataset = _torch()
    moved = dict(batch)
    for key in ["atom_features", "coords", "mask"]:
        moved[key] = moved[key].to(device)
    return moved


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    train_loss: float,
    val_loss: float | None,
    config: dict[str, Any],
    scheduler=None,
) -> Path:
    torch, _DataLoader, _Dataset = _torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "config": config,
    }
    torch.save(checkpoint, output)
    return output


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, device: str = "cpu") -> dict[str, Any]:
    torch, _DataLoader, _Dataset = _torch()
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def debug_train_step(
    graphs: list[dict[str, Any]],
    output_checkpoint: str | Path,
    atom_feature_dim: int = 64,
    hidden_dim: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    torch, _DataLoader, _Dataset = _torch()
    if not graphs:
        raise ValueError("No graphs supplied for SE(3) debug train step")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch = collate_graphs(graphs[: min(4, len(graphs))])
    batch = move_batch(batch, device)
    model = SE3FlowMatching(SE3FlowConfig(atom_feature_dim=atom_feature_dim, hidden_dim=hidden_dim, num_layers=2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    model.train()
    loss = flow_matching_loss(model, batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite debug loss: {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    checkpoint_path = save_checkpoint(
        output_checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=1,
        train_loss=float(loss.item()),
        val_loss=None,
        config={"atom_feature_dim": atom_feature_dim, "hidden_dim": hidden_dim, "debug": True},
    )
    reloaded = SE3FlowMatching(SE3FlowConfig(atom_feature_dim=atom_feature_dim, hidden_dim=hidden_dim, num_layers=2)).to(device)
    load_checkpoint(checkpoint_path, reloaded, device=device)
    return {
        "device": device,
        "loss": float(loss.item()),
        "checkpoint": str(checkpoint_path),
        "num_graphs": len(graphs),
        "status": "passed",
    }


def train_epochs(
    train_graphs: list[dict[str, Any]],
    val_graphs: list[dict[str, Any]],
    checkpoint_dir: str | Path,
    atom_feature_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 4,
    batch_size: int = 4,
    epochs: int = 20,
    learning_rate: float = 2e-4,
    checkpoint_every: int = 10,
    device: str | None = None,
    init_checkpoint: str | Path | None = None,
    patience: int = 40,
    min_delta: float = 1e-4,
) -> Path:
    torch, _DataLoader, _Dataset = _torch()
    pd = _pandas()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    model = SE3FlowMatching(
        SE3FlowConfig(atom_feature_dim=atom_feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    train_loader = make_dataloader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_graphs or train_graphs[: min(8, len(train_graphs))], batch_size=batch_size, shuffle=False)
    best_loss = float("inf")
    best_path = checkpoint_root / "se3_best_checkpoint.pt"
    log_rows = []
    global_step = 0
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            losses.append(float(loss.item()))
        train_loss = sum(losses) / max(len(losses), 1)
        val_loss = evaluate_loss(model, val_loader, device)
        log_rows.append({"epoch": epoch, "global_step": global_step, "train_loss": train_loss, "val_loss": val_loss})
        latest_path = checkpoint_root / "se3_latest_checkpoint.pt"
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            config={"atom_feature_dim": atom_feature_dim, "hidden_dim": hidden_dim, "num_layers": num_layers},
        )
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                train_loss=train_loss,
                val_loss=val_loss,
                config={"atom_feature_dim": atom_feature_dim, "hidden_dim": hidden_dim, "num_layers": num_layers},
            )
        else:
            epochs_without_improvement += 1
        if checkpoint_every and epoch % checkpoint_every == 0:
            save_checkpoint(
                checkpoint_root / f"se3_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                train_loss=train_loss,
                val_loss=val_loss,
                config={"atom_feature_dim": atom_feature_dim, "hidden_dim": hidden_dim, "num_layers": num_layers},
            )
        pd.DataFrame(log_rows).to_csv(checkpoint_root / "training_log.csv", index=False)
        if patience > 0 and epochs_without_improvement >= patience:
            break
    return best_path


def evaluate_loss(model, loader, device: str) -> float:
    torch, _DataLoader, _Dataset = _torch()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            losses.append(float(flow_matching_loss(model, batch).item()))
    return sum(losses) / max(len(losses), 1)
