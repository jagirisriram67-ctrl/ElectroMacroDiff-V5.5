"""Train V6 Account 1 activity reward model.

This is the first V6 lane because it uses the target-specific ChEMBL/BindingDB
style activity tables already available or mounted on Kaggle.  It is still a
reward/prior model, not a docking replacement and not a protein-diffusion model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V6 JAK/JAK2 activity reward model.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--activity-csv", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--active-threshold", type=float, default=7.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.max_rows = args.max_rows or 300

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from emd_v5_2_hybrid.device_helper import device_summary, resolve_device, xla_step
    from emd_v5_2_hybrid.pocket_features import load_pocket_feature_vector
    from emd_v5_2_hybrid.v6_activity_reward import (
        V6_ACTIVITY_INPUT_DIM,
        build_activity_examples,
        make_activity_reward_mlp,
        save_activity_reward_checkpoint,
    )

    base = Path(args.base).resolve()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(json.dumps(device_summary(device), indent=2))

    output_dir = Path(args.output_dir) if args.output_dir else base / "04_models_checkpoints" / "v6_activity_reward"
    if args.smoke_test and args.output_dir is None:
        output_dir = base / "04_models_checkpoints" / "v6_activity_reward_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    pocket_json = base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
    pocket_vector = load_pocket_feature_vector(pocket_json)

    rows = _load_activity_rows(base, args.activity_csv, args.max_rows)
    examples = build_activity_examples(rows, pocket_vector, active_threshold=args.active_threshold)
    if len(examples) < 20 and not args.smoke_test:
        raise RuntimeError(f"Need at least 20 valid examples, found {len(examples)}")
    if not examples:
        raise RuntimeError("No valid activity examples found")

    training_csv = base / "03_features" / "v6_activity_reward_training.csv"
    _write_training_csv(training_csv, examples)
    print(f"Examples: {len(examples)} | active rate: {sum(e.active_label for e in examples) / len(examples):.3f}")
    print(f"Wrote training table: {training_csv}")

    train, val, test = _split_examples(examples)
    print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")

    model = make_activity_reward_mlp(input_dim=V6_ACTIVITY_INPUT_DIM, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_val = -math.inf
    best_epoch = 0
    stale = 0
    log_rows: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train)
        losses = []
        for batch in _batches(train, args.batch_size):
            x, y_active, y_pact = _tensorize(batch, device)
            logits, pred_pact = model(x)
            loss_cls = F.binary_cross_entropy_with_logits(logits, y_active)
            loss_reg = F.mse_loss(pred_pact, y_pact)
            loss = loss_cls + 0.25 * loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            xla_step()
            losses.append(float(loss.item()))

        val_metrics = _evaluate(model, val, device, args.batch_size)
        score = _metric_for_selection(val_metrics)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": round(sum(losses) / max(len(losses), 1), 6),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:4d} | val_auc={val_metrics.get('roc_auc')} "
                f"| val_rmse={val_metrics['rmse']:.4f} | {time.time() - start:.0f}s"
            )

        if score > best_val:
            best_val = score
            best_epoch = epoch
            stale = 0
            save_activity_reward_checkpoint(
                output_dir / "v6_activity_reward_best.pt",
                model,
                {
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "active_threshold": args.active_threshold,
                },
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(output_dir / "v6_activity_reward_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _evaluate(model, test, device, args.batch_size)

    log_csv = output_dir / "v6_activity_reward_training_log.csv"
    pd.DataFrame(log_rows).to_csv(log_csv, index=False)
    summary = {
        "status": "completed",
        "model_type": "v6_activity_reward",
        "examples": len(examples),
        "active_threshold": args.active_threshold,
        "best_epoch": best_epoch,
        "best_val_score": best_val,
        "test_metrics": test_metrics,
        "checkpoint": str(output_dir / "v6_activity_reward_best.pt"),
        "training_csv": str(training_csv),
        "claim_boundary": "Activity reward improves target-specific prioritization; it is not a protein-conditioned generator by itself.",
    }
    summary_path = output_dir / "v6_activity_reward_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _load_activity_rows(base: Path, activity_csvs: list[str], max_rows: int | None) -> list[dict[str, Any]]:
    paths = [Path(p) for p in activity_csvs]
    if not paths:
        paths = [
            base / "02_curated_data" / "jak2_curated_ligands.csv",
            base / "01_raw_data" / "chembl" / "jak2_activities_raw.csv",
        ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized = _normalize_activity_row(row)
                if normalized:
                    rows.append(normalized)
                if max_rows and len(rows) >= max_rows:
                    return rows
    return rows


def _normalize_activity_row(row: dict[str, str]) -> dict[str, Any] | None:
    smiles = row.get("canonical_smiles") or row.get("smiles") or row.get("SMILES")
    if not smiles:
        return None
    p_activity = _to_float(row.get("p_activity") or row.get("pchembl_value"))
    if p_activity is None:
        standard_nm = _to_float(row.get("activity_value_nM") or row.get("standard_value"))
        units = str(row.get("standard_units") or row.get("units") or "nM").lower()
        if standard_nm is not None and standard_nm > 0:
            if units in {"um", "micromolar", "micro molar"}:
                standard_nm *= 1000.0
            p_activity = 9.0 - math.log10(standard_nm)
    if p_activity is None or not math.isfinite(p_activity):
        return None
    return {
        "mol_id": row.get("mol_id") or row.get("molecule_chembl_id") or row.get("source_id"),
        "canonical_smiles": smiles,
        "inchikey": row.get("inchikey") or row.get("standard_inchi_key") or "",
        "p_activity": p_activity,
    }


def _write_training_csv(path: Path, examples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mol_id", "canonical_smiles", "inchikey", "p_activity", "active_label"],
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "mol_id": example.mol_id,
                    "canonical_smiles": example.canonical_smiles,
                    "inchikey": example.inchikey,
                    "p_activity": example.p_activity,
                    "active_label": example.active_label,
                }
            )


def _split_examples(examples):
    grouped: dict[str, list[Any]] = {}
    for example in examples:
        key = example.inchikey or example.canonical_smiles
        grouped.setdefault(key, []).append(example)
    keys = list(grouped)
    random.Random(42).shuffle(keys)
    n = len(keys)
    train_keys = set(keys[: int(0.7 * n)])
    val_keys = set(keys[int(0.7 * n): int(0.85 * n)])
    test_keys = set(keys[int(0.85 * n):])

    train = [ex for key in train_keys for ex in grouped[key]]
    val = [ex for key in val_keys for ex in grouped[key]]
    test = [ex for key in test_keys for ex in grouped[key]]
    if not val:
        val = train[: max(1, len(train) // 10)]
    if not test:
        test = val
    return train, val, test


def _batches(rows, batch_size: int):
    for idx in range(0, len(rows), batch_size):
        yield rows[idx:idx + batch_size]


def _tensorize(batch, device):
    import torch

    x = torch.tensor([row.feature_vector for row in batch], dtype=torch.float32, device=device)
    y_active = torch.tensor([row.active_label for row in batch], dtype=torch.float32, device=device)
    y_pact = torch.tensor([row.p_activity for row in batch], dtype=torch.float32, device=device)
    return x, y_active, y_pact


def _evaluate(model, rows, device, batch_size: int) -> dict[str, Any]:
    import numpy as np
    import torch

    model.eval()
    labels: list[float] = []
    probs: list[float] = []
    y_true: list[float] = []
    y_pred: list[float] = []
    with torch.no_grad():
        for batch in _batches(rows, batch_size):
            x, y_active, y_pact = _tensorize(batch, device)
            logits, pred_pact = model(x)
            labels.extend(y_active.detach().cpu().numpy().tolist())
            probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            y_true.extend(y_pact.detach().cpu().numpy().tolist())
            y_pred.extend(pred_pact.detach().cpu().numpy().tolist())

    y_true_np = np.array(y_true, dtype=float)
    y_pred_np = np.array(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true_np - y_pred_np) ** 2))) if len(y_true_np) else None
    pearson = _pearson(y_true_np, y_pred_np)
    auc = _roc_auc(labels, probs)
    return {
        "roc_auc": auc,
        "rmse": rmse,
        "pearson": pearson,
        "active_rate": float(np.mean(labels)) if labels else None,
    }


def _metric_for_selection(metrics: dict[str, Any]) -> float:
    auc = metrics.get("roc_auc")
    if auc is not None:
        return float(auc)
    pearson = metrics.get("pearson")
    if pearson is not None:
        return float(pearson)
    rmse = metrics.get("rmse")
    return -float(rmse if rmse is not None else 999.0)


def _roc_auc(labels: list[float], probs: list[float]) -> float | None:
    if len(set(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return _roc_auc_no_sklearn(labels, probs)
    return float(roc_auc_score(labels, probs))


def _roc_auc_no_sklearn(labels: list[float], probs: list[float]) -> float | None:
    pairs = sorted(zip(probs, labels), key=lambda x: x[0])
    pos = sum(1 for _, y in pairs if y >= 0.5)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = sum(idx + 1 for idx, (_, y) in enumerate(pairs) if y >= 0.5)
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def _pearson(a, b) -> float | None:
    import numpy as np

    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
