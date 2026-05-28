from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress


FEATURE_COLUMNS = [
    "atomic_num",
    "degree",
    "formal_charge",
    "is_aromatic",
    "is_ring",
    "total_h",
    "hybridization_sp",
    "hybridization_sp2",
    "hybridization_sp3",
    "mass",
    "gasteiger_charge",
    "num_rotatable_neighbors",
    "shortest_path_to_ring",
    "is_between_rings",
    "neighbor_heteroatom_count",
    "is_terminal_chain_atom",
    "local_connectivity_index",
    "ring_size_of_nearest_ring",
]


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install torch for V5.3 anchor model training") from exc
    return torch, nn


def split_by_molecule(frame, seed: int = 42, train_fraction: float = 0.75, val_fraction: float = 0.15):
    molecule_ids = sorted(frame["mol_id"].dropna().astype(str).unique())
    random.Random(seed).shuffle(molecule_ids)
    train_cut = int(len(molecule_ids) * train_fraction)
    val_cut = int(len(molecule_ids) * (train_fraction + val_fraction))
    train_ids = set(molecule_ids[:train_cut])
    val_ids = set(molecule_ids[train_cut:val_cut])
    test_ids = set(molecule_ids[val_cut:])
    return (
        frame[frame["mol_id"].astype(str).isin(train_ids)].copy(),
        frame[frame["mol_id"].astype(str).isin(val_ids)].copy(),
        frame[frame["mol_id"].astype(str).isin(test_ids)].copy(),
    )


def tensorize(frame, device: str):
    torch, _nn = _torch()
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Anchor training CSV is missing enhanced feature columns. "
            f"Rebuild it with scripts/13_build_v5_3_fragment_dataset.py first. Missing: {missing}"
        )
    x = torch.tensor(frame[FEATURE_COLUMNS].astype(float).values, dtype=torch.float32, device=device)
    y = torch.tensor(frame["label_anchor"].astype(float).values, dtype=torch.float32, device=device).view(-1, 1)
    return x, y


def make_model(input_dim: int, hidden_dim: int):
    torch, nn = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(make_model.dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(make_model.dropout),
        nn.Linear(hidden_dim, 1),
    )


make_model.dropout = 0.1


class FocalLoss(_torch()[1].Module):
    def __init__(self, alpha: float = 0.65, gamma: float = 2.0):
        torch, nn = _torch()
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        torch, _nn = _torch()
        import torch.nn.functional as F

        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * ((1.0 - p_t) ** self.gamma) * bce).mean()


def snapshot_state_dict(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def binary_metrics(logits, labels, threshold: float = 0.5) -> dict:
    torch, _nn = _torch()
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    labels = labels.float()
    tp = float(((pred == 1) & (labels == 1)).sum().item())
    tn = float(((pred == 0) & (labels == 0)).sum().item())
    fp = float(((pred == 1) & (labels == 0)).sum().item())
    fn = float(((pred == 0) & (labels == 1)).sum().item())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
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
        threshold = index / 100.0
        metrics = binary_metrics(logits, labels, threshold=threshold)
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    return best or binary_metrics(logits, labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V5.3 macrocycle anchor-site classifier.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.65)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--max-pos-weight", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import pandas as pd
    torch, nn = _torch()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else base / "03_features" / "v5_3_anchor_atom_training.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base / "04_models_checkpoints" / "v5_3_anchor"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_model.dropout = args.dropout
    model_path = output_dir / "anchor_site_model.pt"
    log_path = output_dir / "anchor_training_log.csv"
    summary_path = output_dir / "anchor_training_summary.json"

    frame = pd.read_csv(input_csv)
    if frame.empty:
        raise ValueError(f"Anchor dataset is empty: {input_csv}")
    train_frame, val_frame, test_frame = split_by_molecule(frame, seed=args.seed)
    x_train, y_train = tensorize(train_frame, device)
    x_val, y_val = tensorize(val_frame if not val_frame.empty else train_frame, device)
    x_test, y_test = tensorize(test_frame if not test_frame.empty else val_frame, device)

    model = make_model(len(FEATURE_COLUMNS), args.hidden_dim).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        checkpoint_hidden = int(checkpoint.get("hidden_dim", args.hidden_dim))
        checkpoint_features = checkpoint.get("feature_columns", FEATURE_COLUMNS)
        if checkpoint_hidden != args.hidden_dim or checkpoint_features != FEATURE_COLUMNS:
            raise ValueError("Anchor init checkpoint architecture/features do not match current settings.")
        model.load_state_dict(checkpoint["model_state_dict"])
    positives = float(y_train.sum().item())
    negatives = float(y_train.numel() - positives)
    raw_pos_weight = negatives / max(positives, 1.0)
    capped_pos_weight = min(raw_pos_weight, args.max_pos_weight) if args.max_pos_weight > 0 else raw_pos_weight
    pos_weight = torch.tensor([capped_pos_weight], dtype=torch.float32, device=device)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_payload = None
    rows = []
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(x_train)
        train_loss = criterion(train_logits, y_train)
        train_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val)
            train_metrics = best_threshold_metrics(train_logits.detach(), y_train)
            val_metrics = best_threshold_metrics(val_logits, y_val)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss.item()),
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
                "feature_columns": FEATURE_COLUMNS,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "loss": args.loss,
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
        raise RuntimeError("Training did not produce a best checkpoint")
    torch.save(best_payload, model_path)
    model.load_state_dict(best_payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        tuned_threshold = float(best_threshold_metrics(val_logits, y_val)["threshold"])
        test_metrics = binary_metrics(model(x_test), y_test, threshold=tuned_threshold)
        val_metrics = binary_metrics(val_logits, y_val, threshold=tuned_threshold)

    summary = {
        "status": "completed",
        "device": device,
        "input_csv": str(input_csv),
        "model_path": str(model_path),
        "training_log": str(log_path),
        "epochs": args.epochs,
        "epochs_completed": int(rows[-1]["epoch"]) if rows else 0,
        "hidden_dim": args.hidden_dim,
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
        "max_pos_weight": args.max_pos_weight,
        "init_checkpoint": args.init_checkpoint or "",
        "feature_columns": FEATURE_COLUMNS,
        "num_rows": int(len(frame)),
        "num_train_rows": int(len(train_frame)),
        "num_val_rows": int(len(val_frame)),
        "num_test_rows": int(len(test_frame)),
        "positive_fraction": round(float(frame["label_anchor"].mean()), 6),
        "best_epoch": int(best_payload["epoch"]),
        "selected_threshold": tuned_threshold,
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_progress(
        base / "00_project_registry" / "progress_v5_3_anchor_training.json",
        {
            "stage": "V5_3_anchor_training",
            "status": "completed",
            "best_epoch": int(best_payload["epoch"]),
            "best_val_f1": val_metrics["f1"],
            "test_f1": test_metrics["f1"],
        },
    )
    register_artifact(base, "V5_3_anchor_training", model_path, "anchor_site_model", owner="Student 3")
    register_artifact(base, "V5_3_anchor_training", log_path, "anchor_training_log", owner="Student 3")
    register_artifact(base, "V5_3_anchor_training", summary_path, "anchor_training_summary", owner="Student 3")
    register_run(
        base,
        stage="V5_3_anchor_training",
        status="completed",
        input_path=str(input_csv),
        output_path=str(model_path),
        molecules_in=int(frame["mol_id"].nunique()),
        molecules_out=int(len(frame)),
        notes=f"best_epoch={best_payload['epoch']}; val_f1={val_metrics['f1']}; test_f1={test_metrics['f1']}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
