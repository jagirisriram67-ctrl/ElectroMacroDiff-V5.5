"""V5.5 PocketValidityRewardModel Training Script — KAGGLE ACCOUNT 3.

Trains a reward model on generation attempt logs to predict whether a
proposed candidate will pass validity and macrocycle filters.

This model enables gated generation: reject bad candidates BEFORE
the expensive RDKit construction step, solving the validity collapse
(2.5%) observed in diversity-enforced generation.

Run on Kaggle:
    python scripts/27_train_validity_reward.py --base . --device auto --epochs 2000
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
    parser = argparse.ArgumentParser(description="Train V5.5 PocketValidityRewardModel.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default="v5_5_validity_reward")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.checkpoint_every = 1
        args.max_rows = args.max_rows or 1000

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    from emd_v5_2_hybrid.device_helper import resolve_device, device_summary, xla_step
    from emd_v5_2_hybrid.validity_reward import (
        PocketValidityRewardModel,
        REWARD_INPUT_DIM,
        attempt_to_feature_vector,
        attempt_label,
        save_validity_reward_checkpoint,
    )
    from emd_v5_2_hybrid.pocket_features import load_pocket_feature_vector

    base = Path(args.base).resolve()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    default_dir = "v5_5_validity_reward_smoke" if args.smoke_test else "v5_5_validity_reward"
    output_dir = Path(args.output_dir) if args.output_dir else base / "04_models_checkpoints" / default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pocket vector
    pocket_json = base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
    pocket_vector = load_pocket_feature_vector(pocket_json)

    # Load ALL attempt logs (logged + diverse = maximum training signal)
    attempt_log_files = [
        base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles_kaggle_logged_attempt_log.csv",
        base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles_kaggle_diverse_attempt_log.csv",
    ]

    all_features = []
    all_labels = []
    all_group_ids = []

    remaining_rows = args.max_rows
    for log_path in attempt_log_files:
        if not log_path.exists():
            print(f"INFO: {log_path.name} not found, skipping")
            continue
        read_kwargs = {"nrows": remaining_rows} if remaining_rows else {}
        df = pd.read_csv(log_path, **read_kwargs)
        if remaining_rows:
            remaining_rows = max(remaining_rows - len(df), 0)
        # Filter to rows with anchor scores (skip seed-level failures)
        valid_mask = df["anchor_score_a"].notna() & df["anchor_score_b"].notna()
        df = df[valid_mask]
        print(f"Loading {log_path.name}: {len(df)} attempts")

        for _, row in df.iterrows():
            feat = attempt_to_feature_vector(row.to_dict(), pocket_vector)
            label = attempt_label(str(row.get("status", "")))
            all_features.append(feat)
            all_labels.append(label)
            all_group_ids.append(f"{log_path.stem}:{row.get('parent_mol_id', row.get('seed_smiles', 'unknown'))}")
        if remaining_rows == 0:
            break

    print(f"Total training samples: {len(all_features)}")
    if not all_features:
        print("ERROR: No attempt log data found")
        sys.exit(1)

    X = torch.tensor(all_features, dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.float32)

    pos_count = y.sum().item()
    neg_count = len(y) - pos_count
    print(f"Positive (valid): {int(pos_count)}, Negative (invalid): {int(neg_count)}")
    print(f"Positive rate: {pos_count / len(y) * 100:.2f}%")

    # Class weight for imbalanced data
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device).clamp(max=20.0)
    print(f"Pos weight: {pos_weight.item():.2f}")

    train_idx, val_idx, test_idx = _split_indices_by_group(all_group_ids, seed=42)
    if len(val_idx) == 0:
        val_idx = train_idx
    if len(test_idx) == 0:
        test_idx = val_idx
    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    # Model
    model = PocketValidityRewardModel(
        input_dim=REWARD_INPUT_DIM,
        hidden_dim=args.hidden_dim,
        dropout=0.15,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    log_rows = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_idx))
        train_losses = []

        for batch_start in range(0, len(perm), args.batch_size):
            batch_perm = perm[batch_start:batch_start + args.batch_size]
            batch_idx = train_idx[batch_perm]
            x_batch = X[batch_idx].to(device)
            y_batch = y[batch_idx].to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            xla_step()
            train_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = sum(train_losses) / max(len(train_losses), 1)

        # Validation
        val_metrics = _evaluate_reward(model, X, y, val_idx, device)

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_roc_auc": round(val_metrics["roc_auc"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
        })

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d} | loss={train_loss:.4f} | AUC={val_metrics['roc_auc']:.4f} "
                  f"| F1={val_metrics['f1']:.3f} P={val_metrics['precision']:.3f} R={val_metrics['recall']:.3f} "
                  f"| {elapsed:.0f}s")

        if val_metrics["roc_auc"] > best_val_auc + args.min_delta:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            epochs_no_improve = 0
            save_validity_reward_checkpoint(
                output_dir / "validity_reward_best.pt",
                model,
                {"epoch": epoch, "val_roc_auc": val_metrics["roc_auc"], **val_metrics},
            )
        else:
            epochs_no_improve += 1

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save_validity_reward_checkpoint(
                output_dir / f"validity_reward_epoch_{epoch:04d}.pt",
                model, {"epoch": epoch},
            )

        pd.DataFrame(log_rows).to_csv(output_dir / "validity_reward_training_log.csv", index=False)

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}, best AUC={best_val_auc:.4f}")
            break

    # Test
    test_metrics = _evaluate_reward(model, X, y, test_idx, device)
    print(f"\nTest: AUC={test_metrics['roc_auc']:.4f} F1={test_metrics['f1']:.4f}")

    summary = {
        "status": "completed",
        "run_id": args.run_id,
        "smoke_test": bool(args.smoke_test),
        "model_type": "pocket_validity_reward",
        "device": device,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_roc_auc": round(best_val_auc, 6),
        "test_metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in test_metrics.items()},
        "total_samples": len(X),
        "positive_samples": int(pos_count),
        "negative_samples": int(neg_count),
        "total_time_seconds": round(time.time() - start_time, 1),
        **device_summary(device),
    }
    (output_dir / "validity_reward_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _evaluate_reward(model, X, y, indices, device):
    """Evaluate reward model: ROC-AUC and F1."""
    import torch
    model.eval()
    x = X[indices].to(device)
    labels = y[indices]

    with torch.no_grad():
        logits = model(x).cpu()
        probs = torch.sigmoid(logits)

    # ROC-AUC (manual, no sklearn dependency)
    auc = _manual_roc_auc(labels.numpy(), probs.numpy())

    # F1 at threshold 0.5
    preds = (probs > 0.5).float()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)

    return {
        "roc_auc": auc,
        "f1": f1,
        "precision": p,
        "recall": r,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _manual_roc_auc(labels, scores):
    """Simple ROC-AUC computation without sklearn."""
    import numpy as np
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Wilcoxon-Mann-Whitney statistic
    auc = 0.0
    for p in pos:
        auc += (neg < p).sum() + 0.5 * (neg == p).sum()
    return float(auc / (len(pos) * len(neg)))


def _split_indices_by_group(group_ids, seed: int = 42, train_fraction: float = 0.75, val_fraction: float = 0.15):
    """Split rows by parent molecule/source group to reduce leakage."""
    import random
    import numpy as np

    unique_groups = sorted({str(group_id) for group_id in group_ids})
    random.Random(seed).shuffle(unique_groups)
    train_cut = int(len(unique_groups) * train_fraction)
    val_cut = int(len(unique_groups) * (train_fraction + val_fraction))
    train_groups = set(unique_groups[:train_cut])
    val_groups = set(unique_groups[train_cut:val_cut])
    test_groups = set(unique_groups[val_cut:])
    train_idx = [index for index, group_id in enumerate(group_ids) if str(group_id) in train_groups]
    val_idx = [index for index, group_id in enumerate(group_ids) if str(group_id) in val_groups]
    test_idx = [index for index, group_id in enumerate(group_ids) if str(group_id) in test_groups]
    return np.array(train_idx, dtype=int), np.array(val_idx, dtype=int), np.array(test_idx, dtype=int)


if __name__ == "__main__":
    main()
