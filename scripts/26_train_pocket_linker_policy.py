"""Train the V5.5 pocket-conditioned linker policy.

This sequential Kaggle role trains the dual-head model that predicts linker
heavy-atom count and chain-compatible chemotype from anchor-pair features plus
the JAK2 pocket electronic context vector.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V5.5 PocketLinkerPolicy.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default="v5_5_pocket_linker_policy")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.checkpoint_every = 1
        args.max_rows = args.max_rows or 5000

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    from emd_v5_2_hybrid.device_helper import device_summary, resolve_device, xla_step
    from emd_v5_2_hybrid.linker_size_features import LINKER_FEATURE_COLUMNS, core_features, feature_vector
    from emd_v5_2_hybrid.pocket_features import POCKET_FEATURE_DIM, load_pocket_feature_vector
    from emd_v5_2_hybrid.pocket_linker_policy import (
        CHEMOTYPE_LABELS,
        LINKER_SIZE_LABELS,
        PocketLinkerPolicy,
        save_pocket_linker_policy_checkpoint,
    )

    base = Path(args.base).resolve()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(json.dumps(device_summary(device), indent=2))

    default_dir = "v5_5_pocket_linker_policy_smoke" if args.smoke_test else "v5_5_pocket_linker_policy"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base / "04_models_checkpoints" / default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pocket_json = base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
    pocket_vector = load_pocket_feature_vector(pocket_json)
    pocket_tensor = torch.tensor(pocket_vector, dtype=torch.float32, device=device)

    fragment_csv = (
        Path(args.input_csv).resolve()
        if args.input_csv
        else base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv"
    )
    pretrain_csv = base / "02_curated_data" / "v5_3_pretrain_fragment_linker_pairs.csv"
    csv_paths = [fragment_csv] if args.input_csv else [pretrain_csv, fragment_csv]

    all_features: list[list[float]] = []
    all_size_labels: list[int] = []
    all_chemotype_labels: list[int] = []
    all_mol_ids: list[str] = []

    remaining_rows = args.max_rows
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"INFO: {csv_path.name} not found, skipping")
            continue
        read_kwargs = {"nrows": remaining_rows} if remaining_rows else {}
        frame = pd.read_csv(csv_path, **read_kwargs)
        if remaining_rows:
            remaining_rows = max(remaining_rows - len(frame), 0)
        print(f"Processing {csv_path.name}: {len(frame)} rows")

        for row in frame.to_dict(orient="records"):
            linker_heavy = int(row.get("linker_heavy_atoms", 0) or 0)
            if linker_heavy not in LINKER_SIZE_LABELS:
                continue
            ring_size = int(row.get("macrocycle_ring_size", 16) or 16)
            core_heavy = int(row.get("core_heavy_atoms", 20) or 20)
            features = feature_vector(
                core_features(str(row.get("core_smiles", "")), ring_size, core_heavy),
                LINKER_FEATURE_COLUMNS,
            )
            chemotype = infer_chemotype(str(row.get("linker_smiles", "")))
            all_features.append(features)
            all_size_labels.append(LINKER_SIZE_LABELS.index(linker_heavy))
            all_chemotype_labels.append(CHEMOTYPE_LABELS.index(chemotype) if chemotype in CHEMOTYPE_LABELS else 0)
            all_mol_ids.append(str(row.get("mol_id", "")))
        if remaining_rows == 0:
            break

    if not all_features:
        raise RuntimeError("No linker-policy training rows were built.")

    x = torch.tensor(all_features, dtype=torch.float32)
    y_size = torch.tensor(all_size_labels, dtype=torch.long)
    y_chemo = torch.tensor(all_chemotype_labels, dtype=torch.long)
    train_idx, val_idx, test_idx = split_indices_by_molecule(all_mol_ids, seed=42)
    if len(val_idx) == 0:
        val_idx = train_idx
    if len(test_idx) == 0:
        test_idx = val_idx
    print(f"Samples: {len(x)} | split train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    model = PocketLinkerPolicy(
        linker_feature_dim=len(LINKER_FEATURE_COLUMNS),
        pocket_dim=POCKET_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_size_classes=len(LINKER_SIZE_LABELS),
        num_chemotype_classes=len(CHEMOTYPE_LABELS),
        dropout=0.1,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    chemo_counts = torch.bincount(y_chemo[train_idx], minlength=len(CHEMOTYPE_LABELS)).float()
    chemo_weights = (chemo_counts.sum() / chemo_counts.clamp_min(1.0))
    chemo_weights = (chemo_weights / chemo_weights.mean().clamp_min(1e-6)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    size_criterion = nn.CrossEntropyLoss()
    chemo_criterion = nn.CrossEntropyLoss(weight=chemo_weights)

    best_score = -1.0
    best_epoch = 0
    no_improve = 0
    log_rows = []
    start = time.time()
    train_idx_np = np.array(train_idx, dtype=int)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        np.random.shuffle(train_idx_np)
        for start_idx in range(0, len(train_idx_np), args.batch_size):
            batch_idx = train_idx_np[start_idx : start_idx + args.batch_size]
            xb = x[batch_idx].to(device)
            ys = y_size[batch_idx].to(device)
            yc = y_chemo[batch_idx].to(device)
            size_logits, chemo_logits = model(xb, pocket_tensor)
            loss = size_criterion(size_logits, ys) + 0.5 * chemo_criterion(chemo_logits, yc)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            xla_step()
            losses.append(float(loss.item()))

        scheduler.step()
        metrics = evaluate_policy(model, x, y_size, y_chemo, val_idx, pocket_tensor, device)
        score = 0.6 * metrics["size_within_one_accuracy"] + 0.4 * metrics["chemo_accuracy"]
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(sum(losses) / max(len(losses), 1), 6),
                "val_size_exact_accuracy": round(metrics["size_exact_accuracy"], 6),
                "val_size_within_one_accuracy": round(metrics["size_within_one_accuracy"], 6),
                "val_chemo_accuracy": round(metrics["chemo_accuracy"], 6),
                "val_combined": round(score, 6),
            }
        )
        pd.DataFrame(log_rows).to_csv(output_dir / "pocket_linker_policy_training_log.csv", index=False)

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"Epoch {epoch:4d} | exact={metrics['size_exact_accuracy']:.3f} "
                f"within1={metrics['size_within_one_accuracy']:.3f} chemo={metrics['chemo_accuracy']:.3f}"
            )

        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            save_pocket_linker_policy_checkpoint(
                output_dir / "pocket_linker_policy_best.pt",
                model,
                {"epoch": epoch, "val_combined": score, "val_metrics": metrics},
            )
        else:
            no_improve += 1

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save_pocket_linker_policy_checkpoint(output_dir / f"pocket_linker_policy_epoch_{epoch:04d}.pt", model, {"epoch": epoch})
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}; best combined={best_score:.4f}")
            break

    test_metrics = evaluate_policy(model, x, y_size, y_chemo, test_idx, pocket_tensor, device)
    summary = {
        "status": "completed",
        "run_id": args.run_id,
        "smoke_test": bool(args.smoke_test),
        "model_type": "pocket_linker_policy",
        "device": device,
        "input_csv": str(fragment_csv),
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_combined": round(float(best_score), 6),
        "test_metrics": {key: round(float(value), 6) for key, value in test_metrics.items()},
        "total_samples": int(len(x)),
        "total_time_seconds": round(time.time() - start, 1),
        **device_summary(device),
    }
    (output_dir / "pocket_linker_policy_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def infer_chemotype(smiles: str) -> str:
    s = str(smiles or "")
    n_count = s.count("N") + s.count("n")
    o_count = s.count("O") + s.count("o")
    s_count = s.count("S") + s.count("s")
    if o_count >= 2 and n_count == 0 and s_count == 0:
        return "dioxa"
    if n_count >= 2 and o_count == 0 and s_count == 0:
        return "diaza"
    if s_count >= 2 and o_count == 0 and n_count == 0:
        return "dithio"
    if n_count >= 1 and o_count >= 1 and s_count == 0:
        return "aza_oxa"
    if o_count >= 1 and s_count >= 1:
        return "oxa_thio"
    if n_count >= 1 and s_count >= 1:
        return "aza_thio"
    if o_count == 1:
        return "oxa"
    if n_count == 1:
        return "aza"
    if s_count == 1:
        return "thioether"
    return "alkyl"


def split_indices_by_molecule(mol_ids: list[str], seed: int = 42):
    import numpy as np

    unique_ids = sorted({str(mol_id) for mol_id in mol_ids})
    random.Random(seed).shuffle(unique_ids)
    train_cut = int(len(unique_ids) * 0.75)
    val_cut = int(len(unique_ids) * 0.90)
    train_ids = set(unique_ids[:train_cut])
    val_ids = set(unique_ids[train_cut:val_cut])
    test_ids = set(unique_ids[val_cut:])
    train_idx = [idx for idx, mol_id in enumerate(mol_ids) if str(mol_id) in train_ids]
    val_idx = [idx for idx, mol_id in enumerate(mol_ids) if str(mol_id) in val_ids]
    test_idx = [idx for idx, mol_id in enumerate(mol_ids) if str(mol_id) in test_ids]
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int), np.array(test_idx, dtype=int)


def evaluate_policy(model, x, y_size, y_chemo, indices, pocket_tensor, device):
    import torch

    model.eval()
    idx = torch.as_tensor(indices, dtype=torch.long)
    with torch.no_grad():
        size_logits, chemo_logits = model(x[idx].to(device), pocket_tensor)
        size_preds = size_logits.argmax(dim=1).cpu()
        chemo_preds = chemo_logits.argmax(dim=1).cpu()
    ys = y_size[idx]
    yc = y_chemo[idx]
    size_exact = (size_preds == ys).float().mean().item()
    size_within_one = ((size_preds - ys).abs() <= 1).float().mean().item()
    size_mae = (size_preds - ys).abs().float().mean().item()
    chemo_acc = (chemo_preds == yc).float().mean().item()
    return {
        "size_exact_accuracy": size_exact,
        "size_within_one_accuracy": size_within_one,
        "size_mae": size_mae,
        "chemo_accuracy": chemo_acc,
    }


if __name__ == "__main__":
    main()
