from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.linker_size_features import LINKER_FEATURE_COLUMNS as FEATURE_COLUMNS
from emd_v5_2_hybrid.linker_size_features import core_features
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install torch for V5.3 linker-size training") from exc
    return torch, nn


def build_feature_frame(fragment_frame):
    import pandas as pd

    rows = []
    for row in fragment_frame.to_dict(orient="records"):
        linker_size = int(row["linker_heavy_atoms"])
        if linker_size < 3 or linker_size > 12:
            continue
        features = core_features(
            str(row.get("core_smiles", "")),
            desired_ring_size=int(row.get("macrocycle_ring_size", 0) or 0),
            core_heavy_atoms=int(row.get("core_heavy_atoms", 0) or 0),
        )
        features.update(
            {
                "fragment_id": row.get("fragment_id", ""),
                "mol_id": row.get("mol_id", ""),
                "linker_heavy_atoms": linker_size,
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


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


def make_model(input_dim: int, hidden_dim: int, output_dim: int):
    _torch_mod, nn = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(make_model.dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(make_model.dropout),
        nn.Linear(hidden_dim, output_dim),
    )


make_model.dropout = 0.1


def snapshot_state_dict(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def tensorize(frame, label_to_index: dict[int, int], device: str):
    torch, _nn = _torch()
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Linker-size feature frame is missing columns: {missing}")
    x = torch.tensor(frame[FEATURE_COLUMNS].astype(float).values, dtype=torch.float32, device=device)
    y = torch.tensor([label_to_index[int(v)] for v in frame["linker_heavy_atoms"]], dtype=torch.long, device=device)
    return x, y


def metrics(logits, labels, index_to_label: dict[int, int]) -> dict:
    torch, _nn = _torch()
    pred_idx = logits.argmax(dim=1)
    correct = (pred_idx == labels).float()
    pred_sizes = torch.tensor([index_to_label[int(idx)] for idx in pred_idx.cpu().tolist()], device=labels.device)
    true_sizes = torch.tensor([index_to_label[int(idx)] for idx in labels.cpu().tolist()], device=labels.device)
    within_one = (torch.abs(pred_sizes - true_sizes) <= 1).float()
    mae = torch.abs(pred_sizes.float() - true_sizes.float()).mean()
    return {
        "exact_accuracy": round(float(correct.mean().item()), 6),
        "within_one_accuracy": round(float(within_one.mean().item()), 6),
        "mae_atoms": round(float(mae.item()), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V5.3 linker-size predictor from fragment-linker pairs.")
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
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import pandas as pd
    torch, nn = _torch()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base).resolve()
    input_csv = (
        Path(args.input_csv).resolve()
        if args.input_csv
        else base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else base / "04_models_checkpoints" / "v5_3_linker_size"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    make_model.dropout = args.dropout
    model_path = output_dir / "linker_size_model.pt"
    feature_csv = output_dir / "linker_size_training_features.csv"
    log_path = output_dir / "linker_size_training_log.csv"
    summary_path = output_dir / "linker_size_training_summary.json"

    fragments = pd.read_csv(input_csv)
    frame = build_feature_frame(fragments)
    if frame.empty:
        raise ValueError(f"No linker-size training rows could be built from {input_csv}")
    frame.to_csv(feature_csv, index=False)

    labels = sorted(int(v) for v in frame["linker_heavy_atoms"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_frame, val_frame, test_frame = split_by_molecule(frame, seed=args.seed)
    x_train, y_train = tensorize(train_frame, label_to_index, device)
    x_val, y_val = tensorize(val_frame if not val_frame.empty else train_frame, label_to_index, device)
    x_test, y_test = tensorize(test_frame if not test_frame.empty else val_frame, label_to_index, device)

    model = make_model(len(FEATURE_COLUMNS), args.hidden_dim, len(labels)).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        checkpoint_hidden = int(checkpoint.get("hidden_dim", args.hidden_dim))
        checkpoint_features = checkpoint.get("feature_columns", FEATURE_COLUMNS)
        checkpoint_labels = [int(v) for v in checkpoint.get("labels", labels)]
        if checkpoint_hidden != args.hidden_dim or checkpoint_features != FEATURE_COLUMNS or checkpoint_labels != labels:
            raise ValueError("Linker-size init checkpoint architecture/features/labels do not match current settings.")
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.class_weighting == "balanced":
        counts = torch.bincount(y_train, minlength=len(labels)).float()
        class_weights = counts.sum() / counts.clamp_min(1.0)
        class_weights = class_weights / class_weights.mean().clamp_min(1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        class_weights = None
        criterion = nn.CrossEntropyLoss()

    rows = []
    best_score = -1.0
    best_payload = None
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
            train_metrics = metrics(train_logits.detach(), y_train, index_to_label)
            val_metrics = metrics(val_logits, y_val, index_to_label)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss.item()),
            "train_exact_accuracy": train_metrics["exact_accuracy"],
            "train_within_one_accuracy": train_metrics["within_one_accuracy"],
            "val_exact_accuracy": val_metrics["exact_accuracy"],
            "val_within_one_accuracy": val_metrics["within_one_accuracy"],
            "val_mae_atoms": val_metrics["mae_atoms"],
        }
        rows.append(row)
        improved = val_metrics["within_one_accuracy"] > best_score + args.min_delta
        if improved:
            best_score = val_metrics["within_one_accuracy"]
            epochs_without_improvement = 0
            best_payload = {
                "epoch": epoch,
                "model_state_dict": snapshot_state_dict(model),
                "feature_columns": FEATURE_COLUMNS,
                "hidden_dim": args.hidden_dim,
                "labels": labels,
                "dropout": args.dropout,
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
        raise RuntimeError("Training did not produce a best linker-size checkpoint")
    torch.save(best_payload, model_path)
    model.load_state_dict(best_payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        val_metrics = metrics(model(x_val), y_val, index_to_label)
        test_metrics = metrics(model(x_test), y_test, index_to_label)

    summary = {
        "status": "completed",
        "device": device,
        "input_csv": str(input_csv),
        "feature_csv": str(feature_csv),
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
        "class_weighting": args.class_weighting,
        "class_weights": None if class_weights is None else [round(float(v), 6) for v in class_weights.detach().cpu().tolist()],
        "init_checkpoint": args.init_checkpoint or "",
        "feature_columns": FEATURE_COLUMNS,
        "num_rows": int(len(frame)),
        "num_train_rows": int(len(train_frame)),
        "num_val_rows": int(len(val_frame)),
        "num_test_rows": int(len(test_frame)),
        "labels": labels,
        "best_epoch": int(best_payload["epoch"]),
        "best_val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_progress(
        base / "00_project_registry" / "progress_v5_3_linker_size_training.json",
        {
            "stage": "V5_3_linker_size_training",
            "status": "completed",
            "best_epoch": int(best_payload["epoch"]),
            "val_within_one_accuracy": val_metrics["within_one_accuracy"],
            "test_within_one_accuracy": test_metrics["within_one_accuracy"],
        },
    )
    register_artifact(base, "V5_3_linker_size_training", model_path, "linker_size_model", owner="Student 3")
    register_artifact(base, "V5_3_linker_size_training", feature_csv, "linker_size_training_features", owner="Student 3")
    register_artifact(base, "V5_3_linker_size_training", log_path, "linker_size_training_log", owner="Student 3")
    register_artifact(base, "V5_3_linker_size_training", summary_path, "linker_size_training_summary", owner="Student 3")
    register_run(
        base,
        stage="V5_3_linker_size_training",
        status="completed",
        input_path=str(input_csv),
        output_path=str(model_path),
        molecules_in=int(frame["mol_id"].nunique()),
        molecules_out=int(len(frame)),
        notes=(
            f"best_epoch={best_payload['epoch']}; "
            f"val_within_one={val_metrics['within_one_accuracy']}; "
            f"test_within_one={test_metrics['within_one_accuracy']}"
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
