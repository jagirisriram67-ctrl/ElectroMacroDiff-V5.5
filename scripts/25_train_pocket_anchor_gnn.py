"""V5.5 PocketAnchorGNN Training Script — KAGGLE ACCOUNT 1.

Self-contained training script designed for a dedicated Kaggle session.
Trains the pocket-conditioned anchor GNN that predicts attachment sites
informed by JAK2 pocket electronic features.

Run on Kaggle:
    python scripts/25_train_pocket_anchor_gnn.py --base . --device auto --epochs 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V5.5 PocketAnchorGNN.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--min-delta", type=float, default=0.00001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--focal-alpha", type=float, default=0.65)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--anchor-csv", default=None)
    parser.add_argument("--fragment-csv", default=None)
    parser.add_argument("--run-id", default="v5_5_pocket_anchor_gnn")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.checkpoint_every = 1
        args.max_rows = args.max_rows or 5000
        args.max_graphs = args.max_graphs or 128

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    from emd_v5_2_hybrid.device_helper import resolve_device, device_summary, xla_step
    from emd_v5_2_hybrid.pocket_anchor_gnn import PocketAnchorGNN, save_pocket_anchor_gnn_checkpoint
    from emd_v5_2_hybrid.pocket_features import POCKET_FEATURE_DIM, load_pocket_feature_vector
    from emd_v5_2_hybrid.anchor_gnn import graph_from_smiles_for_anchor_gnn, collate_graphs, move_batch
    from emd_v5_2_hybrid.macrocycle_fragmentation import ANCHOR_ATOM_COLUMNS

    base = Path(args.base).resolve()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(json.dumps(device_summary(device), indent=2))

    default_dir = "v5_5_pocket_anchor_gnn_smoke" if args.smoke_test else "v5_5_pocket_anchor_gnn"
    output_dir = Path(args.output_dir) if args.output_dir else base / "04_models_checkpoints" / default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pocket vector
    pocket_json = base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
    pocket_vector_list = load_pocket_feature_vector(pocket_json)
    pocket_tensor = torch.tensor(pocket_vector_list, dtype=torch.float32, device=device)
    print(f"Pocket vector: {len(pocket_vector_list)} dims")

    # Load training data
    train_csv = (
        Path(args.anchor_csv).resolve()
        if args.anchor_csv
        else base / "03_features" / "v5_3_anchor_atom_training.csv"
    )
    fragment_csv = (
        Path(args.fragment_csv).resolve()
        if args.fragment_csv
        else base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv"
    )
    print(f"Loading anchor training data from {train_csv.name}")
    read_kwargs = {"nrows": args.max_rows} if args.max_rows else {}
    df = pd.read_csv(train_csv, **read_kwargs)
    print(f"  Total rows: {len(df)}, Positive rate: {df['label_anchor'].mean():.4f}")
    smiles_by_mol = _load_smiles_by_mol_id(base, fragment_csv)

    # Build molecular graphs with anchor labels
    print("Building molecular graphs...")
    from rdkit import Chem
    graphs = []
    for mol_id, group in df.groupby("mol_id"):
        mol_id = str(mol_id)
        smiles = smiles_by_mol.get(mol_id)
        if not smiles:
            continue

        graph = graph_from_smiles_for_anchor_gnn(smiles, atom_feature_dim=64)
        if graph is None:
            continue

        # Build labels
        n_atoms = graph["num_nodes"]
        labels = torch.zeros(n_atoms, dtype=torch.float32)
        label_mask = torch.zeros(n_atoms, dtype=torch.float32)

        for _, row in group.iterrows():
            atom_idx = int(row["atom_index"])
            if atom_idx < n_atoms:
                labels[atom_idx] = float(row["label_anchor"])
                label_mask[atom_idx] = 1.0

        graph["labels"] = labels
        graph["label_mask"] = label_mask
        graph["mol_id"] = mol_id
        graphs.append(graph)

        if args.max_graphs and len(graphs) >= args.max_graphs:
            break

    print(f"Built {len(graphs)} molecular graphs")
    if not graphs:
        print("ERROR: No graphs built. Check training data.")
        sys.exit(1)

    train_graphs, val_graphs, test_graphs = _split_graphs_by_mol_id(graphs, seed=42)
    print(f"Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    # Model
    model = PocketAnchorGNN(
        node_dim=64,
        edge_dim=6,
        pocket_dim=POCKET_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    ).to(device)

    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"Loaded init checkpoint: {args.init_checkpoint}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    log_rows = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        np.random.shuffle(train_graphs)
        for batch_start in range(0, len(train_graphs), args.batch_size):
            batch_graphs = train_graphs[batch_start:batch_start + args.batch_size]
            if not batch_graphs:
                continue

            batch = collate_graphs(batch_graphs)
            batch = move_batch(batch, device)

            logits = model(
                batch["node_features"],
                batch["edge_index"],
                batch["edge_features"],
                pocket_tensor,
            )

            labels = batch["labels"]
            mask = batch["label_mask"]

            # Focal loss for class imbalance
            loss = _focal_loss(logits, labels, mask, args.focal_alpha, args.focal_gamma)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            xla_step()
            train_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = sum(train_losses) / max(len(train_losses), 1)

        # Validation
        val_metrics = _evaluate(model, val_graphs, pocket_tensor, device, args.batch_size)
        val_f1 = val_metrics["f1"]

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_f1": round(val_f1, 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 25 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d} | loss={train_loss:.4f} | val_F1={val_f1:.4f} "
                  f"| P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
                  f"| {elapsed:.0f}s")

        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            save_pocket_anchor_gnn_checkpoint(
                output_dir / "pocket_anchor_gnn_best.pt",
                model,
                {"epoch": epoch, "val_f1": val_f1, **val_metrics},
            )
        else:
            epochs_no_improve += 1

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save_pocket_anchor_gnn_checkpoint(
                output_dir / f"pocket_anchor_gnn_epoch_{epoch:04d}.pt",
                model,
                {"epoch": epoch, "val_f1": val_f1},
            )

        pd.DataFrame(log_rows).to_csv(output_dir / "pocket_anchor_gnn_training_log.csv", index=False)

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}, best F1={best_val_f1:.4f} at epoch {best_epoch}")
            break

    # Test evaluation
    test_metrics = _evaluate(model, test_graphs, pocket_tensor, device, args.batch_size)
    print(f"\nTest metrics: F1={test_metrics['f1']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}")

    summary = {
        "status": "completed",
        "run_id": args.run_id,
        "smoke_test": bool(args.smoke_test),
        "model_type": "pocket_anchor_gnn",
        "device": device,
        "anchor_csv": str(train_csv),
        "fragment_csv": str(fragment_csv),
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 6),
        "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
        "num_graphs": len(graphs),
        "train_graphs": len(train_graphs),
        "val_graphs": len(val_graphs),
        "test_graphs": len(test_graphs),
        "total_time_seconds": round(time.time() - start_time, 1),
        **device_summary(device),
    }
    (output_dir / "pocket_anchor_gnn_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _focal_loss(logits, labels, mask, alpha, gamma):
    """Focal loss for binary classification with class imbalance."""
    import torch
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    pt = labels * probs + (1 - labels) * (1 - probs)
    focal_weight = (1 - pt) ** gamma
    at = labels * alpha + (1 - labels) * (1 - alpha)
    loss = at * focal_weight * bce * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _evaluate(model, graphs, pocket_tensor, device, batch_size):
    """Evaluate anchor prediction metrics."""
    import torch
    from emd_v5_2_hybrid.anchor_gnn import collate_graphs, move_batch

    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch_start in range(0, len(graphs), batch_size):
            batch_graphs = graphs[batch_start:batch_start + batch_size]
            if not batch_graphs:
                continue
            batch = collate_graphs(batch_graphs)
            batch = move_batch(batch, device)
            logits = model(batch["node_features"], batch["edge_index"], batch["edge_features"], pocket_tensor)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["labels"].cpu())
            all_masks.append(batch["label_mask"].cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)

    # Find best threshold
    best_f1, best_thresh = 0.0, 0.5
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        binary = (preds > thresh).float() * masks
        tp = ((binary == 1) & (labels == 1) & (masks == 1)).sum().item()
        fp = ((binary == 1) & (labels == 0) & (masks == 1)).sum().item()
        fn = ((binary == 0) & (labels == 1) & (masks == 1)).sum().item()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    binary = (preds > best_thresh).float() * masks
    tp = ((binary == 1) & (labels == 1) & (masks == 1)).sum().item()
    fp = ((binary == 1) & (labels == 0) & (masks == 1)).sum().item()
    fn = ((binary == 0) & (labels == 1) & (masks == 1)).sum().item()
    tn = ((binary == 0) & (labels == 0) & (masks == 1)).sum().item()

    return {
        "threshold": best_thresh,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _split_graphs_by_mol_id(graphs, seed: int = 42, train_fraction: float = 0.75, val_fraction: float = 0.15):
    import random

    mol_ids = sorted({str(graph.get("mol_id", "")) for graph in graphs})
    random.Random(seed).shuffle(mol_ids)
    train_cut = int(len(mol_ids) * train_fraction)
    val_cut = int(len(mol_ids) * (train_fraction + val_fraction))
    train_ids = set(mol_ids[:train_cut])
    val_ids = set(mol_ids[train_cut:val_cut])
    test_ids = set(mol_ids[val_cut:])
    return (
        [graph for graph in graphs if str(graph.get("mol_id", "")) in train_ids],
        [graph for graph in graphs if str(graph.get("mol_id", "")) in val_ids],
        [graph for graph in graphs if str(graph.get("mol_id", "")) in test_ids],
    )


def _load_smiles_by_mol_id(base: Path, preferred_fragment_csv: Path | None = None) -> dict[str, str]:
    """Load macrocycle/canonical SMILES keyed by mol_id from fragment and curated tables."""
    import pandas as pd

    mapping: dict[str, str] = {}
    fragment_paths = []
    if preferred_fragment_csv is not None:
        fragment_paths.append(Path(preferred_fragment_csv))
    fragment_paths.extend(
        [
            base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv",
            base / "02_curated_data" / "v5_3_pretrain_fragment_linker_pairs.csv",
        ]
    )
    for path in fragment_paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda column: column in {"mol_id", "macrocycle_smiles"})
        if {"mol_id", "macrocycle_smiles"}.issubset(frame.columns):
            for row in frame.dropna(subset=["mol_id", "macrocycle_smiles"]).drop_duplicates("mol_id").to_dict(orient="records"):
                mapping[str(row["mol_id"])] = str(row["macrocycle_smiles"])

    curated = base / "02_curated_data" / "jak2_curated_ligands.csv"
    if not curated.exists():
        return mapping
    frame = pd.read_csv(curated, usecols=lambda column: column in {"mol_id", "canonical_smiles"})
    if {"mol_id", "canonical_smiles"}.issubset(frame.columns):
        for row in frame.dropna(subset=["mol_id", "canonical_smiles"]).drop_duplicates("mol_id").to_dict(orient="records"):
            mapping.setdefault(str(row["mol_id"]), str(row["canonical_smiles"]))
    return mapping


if __name__ == "__main__":
    main()
