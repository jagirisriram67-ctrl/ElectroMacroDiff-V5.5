"""Ligand descriptor, split, and feature table generation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .chemistry import summarize_molecule, summary_as_dict
from .schemas import LIGAND_FEATURE_COLUMNS


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas to build feature tables") from exc
    return pd


def build_ligand_feature_records(
    curated_records: list[dict[str, Any]],
    mw_min: float = 250,
    mw_max: float = 900,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in curated_records:
        smiles = record.get("canonical_smiles")
        if not smiles:
            continue
        summary = summarize_molecule(str(smiles), mw_min=mw_min, mw_max=mw_max)
        if summary is None:
            rows.append(
                {
                    "mol_id": record.get("mol_id", ""),
                    "canonical_smiles": smiles,
                    "passes_rdkit": False,
                    "passes_basic_filters": False,
                }
            )
            continue
        item = summary_as_dict(summary)
        item["mol_id"] = record.get("mol_id", "")
        rows.append({column: item.get(column, "") for column in LIGAND_FEATURE_COLUMNS})
    return rows


def build_ligand_features_csv(
    curated_csv: str | Path,
    output_csv: str | Path,
    mw_min: float = 250,
    mw_max: float = 900,
) -> Path:
    pd = _pandas()
    curated = pd.read_csv(curated_csv)
    rows = build_ligand_feature_records(curated.to_dict(orient="records"), mw_min=mw_min, mw_max=mw_max)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=LIGAND_FEATURE_COLUMNS).to_csv(output, index=False)
    return output


def assign_random_splits(
    rows: list[dict[str, Any]],
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = [dict(row) for row in rows]
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_cut = int(n * train_fraction)
    val_cut = int(n * (train_fraction + val_fraction))
    for index, row in enumerate(shuffled):
        if index < train_cut:
            row["split"] = "train"
        elif index < val_cut:
            row["split"] = "val"
        else:
            row["split"] = "test"
    return shuffled


def write_split_files(curated_csv: str | Path, split_dir: str | Path) -> dict[str, Path]:
    pd = _pandas()
    frame = pd.read_csv(curated_csv)
    output_dir = Path(split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for split in ["train", "val", "test"]:
        path = output_dir / f"{split}_ids.txt"
        ids = frame.loc[frame["split"] == split, "mol_id"].astype(str).tolist()
        path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        outputs[split] = path
    return outputs


def add_splits_to_curated_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> Path:
    pd = _pandas()
    frame = pd.read_csv(input_csv)
    rows = assign_random_splits(
        frame.to_dict(orient="records"),
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
