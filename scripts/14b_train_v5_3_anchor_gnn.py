from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.anchor_gnn import (
    AnchorMessageGNN,
    collate_graphs,
    graph_from_smiles_for_anchor_gnn,
    move_batch,
    save_anchor_gnn_checkpoint,
)
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress


def _torch():
    try:
        import torch
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install torch for anchor GNN training") from exc
    return torch, DataLoader, F


class GraphListDataset(_torch()[1].dataset if hasattr(_torch()[1], "dataset") else object):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


class FocalLoss:
    def __init__(self, alpha: float = 0.65, gamma: float = 2.0):
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def __call__(self, logits, targets):
        torch, _DataLoader, F = _torch()
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


def split_by_molecule(graphs, seed: int = 42, train_fraction: float = 0.75, val_fraction: float = 0.15):
    shuffled = list(graphs)
    random.Random(seed).shuffle(shuffled)
    train_cut = int(len(shuffled) * train_fraction)
    val_cut = int(len(shuffled) * (train_fraction + val_fraction))
    return shuffled[:train_cut], shuffled[train_cut:val_cut], shuffled[val_cut:]


def build_graph_dataset(anchor_csv: Path, fragment_csv: Path, atom_feature_dim: int):
    import pandas as pd
    torch, _DataLoader, _F = _torch()

    anchors = pd.read_csv(anchor_csv)
    fragments = pd.read_csv(fragment_csv)
    if anchors.empty:
        raise ValueError(f"Anchor CSV is empty: {anchor_csv}")
    if fragments.empty:
        raise ValueError(f"Fragment CSV is empty: {fragment_csv}")

    smiles_by_mol = (
        fragments.dropna(subset=["mol_id", "macrocycle_smiles"])
        .drop_duplicates("mol_id")
        .set_index("mol_id")["macrocycle_smiles"]
        .astype(str)
        .to_dict()
    )
    label_frame = (
        anchors.assign(mol_id=anchors["mol_id"].astype(str), atom_index=anchors["atom_index"].astype(int))
        .groupby(["mol_id", "atom_index"], as_index=False)["label_anchor"]
        .max()
    )
    grouped = {mol_id: rows for mol_id, rows in label_frame.groupby("mol_id")}
    graphs = []
    skipped = 0
    for mol_id, rows in grouped.items():
        smiles = smiles_by_mol.get(str(mol_id))
        if not smiles:
            skipped += 1
            continue
        graph = graph_from_smiles_for_anchor_gnn(smiles, atom_feature_dim=atom_feature_dim)
        if graph is None:
            skipped += 1
            continue
        num_nodes = int(graph["num_nodes"])
        labels = torch.zeros(num_nodes, dtype=torch.float32)
        mask = torch.zeros(num_nodes, dtype=torch.float32)
        ok = True
        for row in rows.to_dict(orient="records"):
            atom_idx = int(row["atom_index"])
            if atom_idx < 0 or atom_idx >= num_nodes:
                ok = False
                break
            labels[atom_idx] = max(float(labels[atom_idx]), float(row["label_anchor"]))
            mask[atom_idx] = 1.0
        if not ok or float(mask.sum().item()) == 0.0:
            skipped += 1
            continue
        graph["labels"] = labels
        graph["label_mask"] = mask
        graph["mol_id"] = str(mol_id)
        graph["smiles"] = smiles
        graphs.append(graph)
    return graphs, skipped


def make_loader(graphs, batch_size: int, shuffle: bool):
    _torch_mod, DataLoader, _F = _torch()
    return DataLoader(GraphListDataset(graphs), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graphs)


def binary_metrics(logits, labels, threshold: float = 0.5) -> dict:
    torch, _DataLoader, _F = _torch()
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    labels = labels.float()
    tp = float(((pred == 1) & (labels == 1)).sum().item())
    tn = float(((pred == 0) & (labels == 0)).sum().item())
    fp = float(((pred == 1) & (labels == 0)).sum().item())
    fn = float(((pred == 0) & (labels == 1)).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def best_threshold_metrics(logits, labels) -> dict:
    best = None
    for index in range(5, 96, 5):
        metrics = binary_metrics(logits, labels, threshold=index / 100.0)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    return best or binary_metrics(logits, labels)


def evaluate(model, loader, device: str, criterion=None):
    torch, _DataLoader, _F = _torch()
    model.eval()
    losses = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch["node_features"], batch["edge_index"], batch["edge_features"])
            mask = batch["label_mask"] > 0
            selected_logits = logits[mask]
            selected_labels = batch["labels"][mask]
            if criterion is not None and selected_logits.numel():
                losses.append(float(criterion(selected_logits, selected_labels).item()))
            all_logits.append(selected_logits.detach().cpu())
            all_labels.append(selected_labels.detach().cpu())
    if not all_logits:
        return {"loss": None, "metrics": binary_metrics(torch.tensor([]), torch.tensor([]))}
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {
        "loss": sum(losses) / max(len(losses), 1) if losses else None,
        "metrics": best_threshold_metrics(logits, labels),
        "logits": logits,
        "labels": labels,
    }


def snapshot_state_dict(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure-Torch V5.3 anchor GNN.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--anchor-csv", default=None)
    parser.add_argument("--fragment-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--atom-feature-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal")
    parser.add_argument("--focal-alpha", type=float, default=0.65)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-pos-weight", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import pandas as pd
    torch, _DataLoader, F = _torch()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base).resolve()
    anchor_csv = Path(args.anchor_csv).resolve() if args.anchor_csv else base / "03_features" / "v5_3_anchor_atom_training.csv"
    fragment_csv = Path(args.fragment_csv).resolve() if args.fragment_csv else base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base / "04_models_checkpoints" / "v5_3_anchor_gnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "anchor_gnn_model.pt"
    log_path = output_dir / "anchor_gnn_training_log.csv"
    summary_path = output_dir / "anchor_gnn_training_summary.json"

    graphs, skipped_graphs = build_graph_dataset(anchor_csv, fragment_csv, atom_feature_dim=args.atom_feature_dim)
    if not graphs:
        raise ValueError("No anchor GNN graphs could be built")
    train_graphs, val_graphs, test_graphs = split_by_molecule(graphs, seed=args.seed)
    train_loader = make_loader(train_graphs, args.batch_size, shuffle=True)
    val_loader = make_loader(val_graphs or train_graphs, args.batch_size, shuffle=False)
    test_loader = make_loader(test_graphs or val_graphs or train_graphs, args.batch_size, shuffle=False)

    model = AnchorMessageGNN(
        node_dim=args.atom_feature_dim,
        edge_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        if checkpoint.get("model_type") != "anchor_message_gnn":
            raise ValueError("Init checkpoint is not an anchor_message_gnn checkpoint")
        if (
            int(checkpoint.get("node_dim", args.atom_feature_dim)) != args.atom_feature_dim
            or int(checkpoint.get("hidden_dim", args.hidden_dim)) != args.hidden_dim
            or int(checkpoint.get("num_layers", args.num_layers)) != args.num_layers
        ):
            raise ValueError("Anchor GNN init checkpoint architecture does not match current settings.")
        model.load_state_dict(checkpoint["model_state_dict"])

    all_train_labels = torch.cat([graph["labels"][graph["label_mask"] > 0] for graph in train_graphs])
    positives = float(all_train_labels.sum().item())
    negatives = float(all_train_labels.numel() - positives)
    raw_pos_weight = negatives / max(positives, 1.0)
    capped_pos_weight = min(raw_pos_weight, args.max_pos_weight) if args.max_pos_weight > 0 else raw_pos_weight
    pos_weight = torch.tensor([capped_pos_weight], dtype=torch.float32, device=device)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = lambda logits, labels: F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    rows = []
    best_payload = None
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["node_features"], batch["edge_index"], batch["edge_features"])
            mask = batch["label_mask"] > 0
            loss = criterion(logits[mask], batch["labels"][mask])
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        train_eval = evaluate(model, train_loader, device)
        val_eval = evaluate(model, val_loader, device, criterion=criterion)
        train_metrics = train_eval["metrics"]
        val_metrics = val_eval["metrics"]
        row = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            "val_loss": val_eval["loss"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        rows.append(row)
        improved = val_metrics["f1"] > best_val_f1 + args.min_delta
        if improved:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            best_payload = {
                "epoch": epoch,
                "model_state_dict": snapshot_state_dict(model),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
            }
        else:
            epochs_without_improvement += 1
        if args.checkpoint_every and (epoch % args.checkpoint_every == 0 or epoch == args.epochs):
            pd.DataFrame(rows).to_csv(log_path, index=False)
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            pd.DataFrame(rows).to_csv(log_path, index=False)
            break

    if best_payload is None:
        raise RuntimeError("Training did not produce a best anchor GNN checkpoint")
    model.load_state_dict(best_payload["model_state_dict"])
    val_eval = evaluate(model, val_loader, device)
    tuned_threshold = float(val_eval["metrics"]["threshold"])
    test_eval = evaluate(model, test_loader, device)
    test_metrics = binary_metrics(test_eval["logits"], test_eval["labels"], threshold=tuned_threshold)
    val_metrics = binary_metrics(val_eval["logits"], val_eval["labels"], threshold=tuned_threshold)
    save_anchor_gnn_checkpoint(
        model_path,
        model,
        {
            "epoch": int(best_payload["epoch"]),
            "threshold": tuned_threshold,
            "anchor_csv": str(anchor_csv),
            "fragment_csv": str(fragment_csv),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )

    summary = {
        "status": "completed",
        "device": device,
        "anchor_csv": str(anchor_csv),
        "fragment_csv": str(fragment_csv),
        "model_path": str(model_path),
        "training_log": str(log_path),
        "epochs": args.epochs,
        "epochs_completed": int(rows[-1]["epoch"]) if rows else 0,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "atom_feature_dim": args.atom_feature_dim,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "loss": args.loss,
        "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "raw_pos_weight": round(float(raw_pos_weight), 6),
        "pos_weight": round(float(capped_pos_weight), 6),
        "num_graphs": len(graphs),
        "skipped_graphs": skipped_graphs,
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "num_test_graphs": len(test_graphs),
        "positive_fraction": round(float(positives / max(all_train_labels.numel(), 1)), 6),
        "best_epoch": int(best_payload["epoch"]),
        "selected_threshold": tuned_threshold,
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_progress(
        base / "00_project_registry" / "progress_v5_3_anchor_gnn_training.json",
        {
            "stage": "V5_3_anchor_gnn_training",
            "status": "completed",
            "best_epoch": int(best_payload["epoch"]),
            "val_f1": val_metrics["f1"],
            "test_f1": test_metrics["f1"],
        },
    )
    register_artifact(base, "V5_3_anchor_gnn_training", model_path, "anchor_gnn_model", owner="Student 3")
    register_artifact(base, "V5_3_anchor_gnn_training", log_path, "anchor_gnn_training_log", owner="Student 3")
    register_artifact(base, "V5_3_anchor_gnn_training", summary_path, "anchor_gnn_training_summary", owner="Student 3")
    register_run(
        base,
        stage="V5_3_anchor_gnn_training",
        status="completed",
        input_path=f"{anchor_csv}; {fragment_csv}",
        output_path=str(model_path),
        molecules_in=len(graphs),
        molecules_out=len(graphs),
        notes=f"best_epoch={best_payload['epoch']}; val_f1={val_metrics['f1']}; test_f1={test_metrics['f1']}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
