from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.chemistry import canonicalize_smiles, mol_id_from_smiles, summarize_molecule
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress
from emd_v5_2_hybrid.schemas import CURATED_LIGAND_COLUMNS


def _requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install requests to collect ChEMBL macrocycles") from exc
    return requests


def write_rows_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def fetch_chembl_molecule_page(offset: int, limit: int, timeout: int = 90) -> list[dict[str, Any]]:
    requests = _requests()
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    params = {
        "limit": limit,
        "offset": offset,
        "molecule_structures__canonical_smiles__isnull": "false",
        "molecule_properties__mw_freebase__gte": 250,
        "molecule_properties__mw_freebase__lte": 1200,
    }
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json().get("molecules", [])


def macrocycle_row_from_chembl(
    record: dict[str, Any],
    seen_inchikeys: set[str],
    min_ring_size: int,
    max_ring_size: int,
) -> dict | None:
    structures = record.get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")
    if not smiles:
        return None
    canonical = canonicalize_smiles(str(smiles))
    if canonical is None:
        return None
    summary = summarize_molecule(canonical, mw_min=250, mw_max=1200, tpsa_max=260, logp_max=9)
    if summary is None:
        return None
    if summary.inchikey in seen_inchikeys:
        return None
    if not (min_ring_size <= summary.max_ring_size <= max_ring_size):
        return None
    seen_inchikeys.add(summary.inchikey)
    source_id = str(record.get("molecule_chembl_id", ""))
    return {
        "mol_id": mol_id_from_smiles(canonical, "PREMAC"),
        "source": "ChEMBL_macrocycle_scan",
        "source_id": source_id,
        "canonical_smiles": canonical,
        "inchikey": summary.inchikey,
        "activity_type": "",
        "activity_value_nM": "",
        "p_activity": "",
        "target": "broad_macrocycle_pretrain",
        "assay_id": "",
        "confidence_score": "",
        "max_ring_size": summary.max_ring_size,
        "has_macrocycle_12_20": summary.has_macrocycle_12_20,
        "has_constrained_ring_8_11": summary.has_constrained_ring_8_11,
        "split": "",
        "notes": f"pretraining_macrocycle; ring_window={min_ring_size}-{max_ring_size}; max_phase={record.get('max_phase', '')}",
    }


def assign_splits(rows: list[dict], seed: int = 42) -> list[dict]:
    import random

    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_cut = int(n * 0.8)
    val_cut = int(n * 0.9)
    for index, row in enumerate(shuffled):
        if index < train_cut:
            row["split"] = "pretrain_train"
        elif index < val_cut:
            row["split"] = "pretrain_val"
        else:
            row["split"] = "pretrain_test"
    return shuffled


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect broad ChEMBL macrocycles for V5.3 pretraining.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--scan-limit", type=int, default=50000, help="Maximum ChEMBL molecules to scan.")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-macrocycles", type=int, default=3000)
    parser.add_argument("--min-ring-size", type=int, default=12)
    parser.add_argument("--max-ring-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    raw_path = base / "01_raw_data" / "chembl" / "v5_3_macrocycle_pretrain_scan_raw.csv"
    curated_path = base / "02_curated_data" / "v5_3_pretrain_macrocycles.csv"
    progress_path = base / "00_project_registry" / "progress_v5_3_macrocycle_pretrain_collection.json"

    raw_rows: list[dict] = []
    curated_rows: list[dict] = []
    seen_inchikeys: set[str] = set()
    scanned = 0
    offset = 0
    while scanned < args.scan_limit and len(curated_rows) < args.max_macrocycles:
        page_size = min(args.page_size, args.scan_limit - scanned)
        records = fetch_chembl_molecule_page(offset=offset, limit=page_size)
        if not records:
            break
        for record in records:
            scanned += 1
            source_id = str(record.get("molecule_chembl_id", ""))
            structures = record.get("molecule_structures") or {}
            raw_rows.append(
                {
                    "source_id": source_id,
                    "canonical_smiles": structures.get("canonical_smiles", ""),
                    "max_phase": record.get("max_phase", ""),
                    "pref_name": record.get("pref_name", ""),
                }
            )
            row = macrocycle_row_from_chembl(record, seen_inchikeys, args.min_ring_size, args.max_ring_size)
            if row is not None:
                curated_rows.append(row)
                if len(curated_rows) >= args.max_macrocycles:
                    break
        offset += len(records)
        save_progress(
            progress_path,
            {
                "stage": "V5_3_macrocycle_pretrain_collection",
                "status": "running",
                "scanned": scanned,
                "macrocycles_found": len(curated_rows),
                "offset": offset,
                "min_ring_size": args.min_ring_size,
                "max_ring_size": args.max_ring_size,
            },
        )
        if len(records) < page_size:
            break

    curated_rows = assign_splits(curated_rows, seed=args.seed)
    write_rows_csv(raw_path, raw_rows)
    write_rows_csv(curated_path, curated_rows, CURATED_LIGAND_COLUMNS)
    save_progress(
        progress_path,
        {
            "stage": "V5_3_macrocycle_pretrain_collection",
            "status": "completed",
            "scanned": scanned,
            "macrocycles_found": len(curated_rows),
            "min_ring_size": args.min_ring_size,
            "max_ring_size": args.max_ring_size,
        },
    )
    register_artifact(base, "V5_3_macrocycle_pretrain_collection", raw_path, "chembl_macrocycle_scan_raw", owner="Student 2")
    register_artifact(base, "V5_3_macrocycle_pretrain_collection", curated_path, "pretrain_macrocycle_curated_csv", owner="Student 2")
    register_run(
        base,
        stage="V5_3_macrocycle_pretrain_collection",
        status="completed",
        input_path="ChEMBL molecule API",
        output_path=str(curated_path),
        molecules_in=scanned,
        molecules_out=len(curated_rows),
        notes=f"scan_limit={args.scan_limit}; max_macrocycles={args.max_macrocycles}; ring_window={args.min_ring_size}-{args.max_ring_size}",
    )

    print(f"Scanned molecules: {scanned}")
    print(f"Pretraining macrocycles: {len(curated_rows)}")
    print(f"Ring size window: {args.min_ring_size}-{args.max_ring_size}")
    print(f"Raw scan CSV: {raw_path}")
    print(f"Curated macrocycle CSV: {curated_path}")


if __name__ == "__main__":
    main()
