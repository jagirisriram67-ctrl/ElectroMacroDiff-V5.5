from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.data_collection import (
    curate_chembl_records,
    download_pdb,
    fetch_chembl_activities,
    write_dicts_csv,
)
from emd_v5_2_hybrid.features import assign_random_splits
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress
from emd_v5_2_hybrid.schemas import CURATED_LIGAND_COLUMNS


def write_curated_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURATED_LIGAND_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CURATED_LIGAND_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and curate JAK2 ChEMBL/PDB inputs.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--tiny-debug", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    raw_path = base / "01_raw_data" / "chembl" / "jak2_activities_raw.csv"
    curated_path = base / "02_curated_data" / "jak2_curated_ligands.csv"
    pdb_path = base / "01_raw_data" / "pdb" / "5AEP.pdb"
    progress_path = base / "00_project_registry" / "progress_m1_data_collection.json"

    limit = 50 if args.tiny_debug else args.limit
    records = fetch_chembl_activities(limit=limit, output_path=raw_path)
    curated = curate_chembl_records(records, max_records=limit)
    curated = assign_random_splits(curated, seed=42)
    write_curated_csv(curated_path, curated)
    write_dicts_csv(base / "02_curated_data" / "jak2_macrocycle_constrained.csv", curated)
    download_pdb("5AEP", pdb_path)

    save_progress(progress_path, {"last_completed_index": len(curated) - 1, "stage": "M1_data_collection"})
    register_artifact(base, "M1_data_collection", raw_path, "raw_chembl_csv", owner="Student 2")
    register_artifact(base, "M1_data_collection", curated_path, "curated_ligand_csv", owner="Student 2")
    register_artifact(base, "M1_data_collection", pdb_path, "receptor_pdb", owner="Student 5")
    register_run(
        base,
        stage="M1_data_collection",
        status="completed",
        input_path="ChEMBL API; RCSB PDB",
        output_path=str(curated_path),
        molecules_in=len(records),
        molecules_out=len(curated),
        notes="Tiny Debug mode" if args.tiny_debug else "Main collection",
    )
    print(f"Raw records: {len(records)}")
    print(f"Curated ligands: {len(curated)}")
    print(f"Wrote: {curated_path}")
    print(f"Downloaded: {pdb_path}")


if __name__ == "__main__":
    main()
