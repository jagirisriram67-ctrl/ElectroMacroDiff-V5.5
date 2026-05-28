from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.docking import infer_grid_from_ligand, write_grid_json
from emd_v5_2_hybrid.docking_prep import write_ligand_sdf_batch
from emd_v5_2_hybrid.registry import register_artifact, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer docking grid and prepare top candidate 3D SDF files.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--top-n", type=int, default=150)
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--clean", action="store_true", help="Remove old ligand SDF files before writing the current set.")
    parser.add_argument("--macrocycle-only", action="store_true", help="Prepare only 12-20 atom macrocycle candidates.")
    parser.add_argument("--input-csv", default=None, help="Optional candidate/ranking CSV to prepare instead of the default ranking or merged table.")
    parser.add_argument("--ligand-sdf-dir", default=None, help="Optional output directory for per-ligand SDF files.")
    parser.add_argument("--combined-sdf", default=None, help="Optional combined SDF output path.")
    parser.add_argument("--grid-json", default=None, help="Optional docking grid JSON output path.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel CPU workers for RDKit SDF preparation.")
    parser.add_argument("--stage", default="M5_docking_prep", help="Registry stage label.")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    pdb_path = base / "01_raw_data" / "pdb" / "5AEP.pdb"
    ranking_path = base / "08_final_ranking" / "final_ranked_candidates.csv"
    candidate_path = base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
    input_path = Path(args.input_csv).resolve() if args.input_csv else (ranking_path if ranking_path.exists() else candidate_path)
    frame = pd.read_csv(input_path)
    if args.macrocycle_only:
        if "has_macrocycle_12_20" not in frame.columns:
            raise ValueError("Cannot use --macrocycle-only; input table lacks has_macrocycle_12_20.")
        macro_mask = frame["has_macrocycle_12_20"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[macro_mask].copy()
        if frame.empty:
            raise ValueError("No macrocycle candidates available for docking preparation.")
    if "rank" in frame.columns:
        frame = frame.sort_values("rank", ascending=True)
    rows = frame.head(args.top_n).to_dict(orient="records")

    grid = infer_grid_from_ligand(pdb_path, padding=args.padding)
    grid_path = write_grid_json(
        grid,
        Path(args.grid_json).resolve() if args.grid_json else base / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json",
    )

    ligand_sdf_dir = Path(args.ligand_sdf_dir).resolve() if args.ligand_sdf_dir else base / "06_docking" / "ligands_sdf"
    if args.clean and ligand_sdf_dir.exists():
        for path in ligand_sdf_dir.glob("CAND_*.sdf"):
            path.unlink()
        for name in ["candidates_for_docking.sdf", "docking_input_manifest.csv"]:
            path = ligand_sdf_dir / name
            if path.exists():
                path.unlink()

    combined_sdf = Path(args.combined_sdf).resolve() if args.combined_sdf else ligand_sdf_dir / "candidates_for_docking.sdf"
    result = write_ligand_sdf_batch(
        rows,
        output_dir=ligand_sdf_dir,
        combined_sdf=combined_sdf,
        workers=max(int(args.workers), 1),
    )
    register_artifact(base, args.stage, grid_path, "docking_grid_json", owner="Student 5")
    register_artifact(base, args.stage, result["combined_sdf"], "candidate_ligands_sdf", owner="Student 5")
    register_artifact(base, args.stage, result["manifest"], "docking_input_manifest", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=str(input_path),
        output_path=result["combined_sdf"],
        molecules_in=len(rows),
        molecules_out=result["prepared"],
        notes=f"Grid from {grid['reference_ligand']}; {result['failed']} ligand embedding failures; workers={result.get('workers', 1)}",
    )
    print(f"Grid: {grid_path}")
    print(f"Reference ligand: {grid['reference_ligand']}")
    print(f"Center: {grid['center']}")
    print(f"Size: {grid['size']}")
    print(f"Prepared SDF ligands: {result['prepared']} / {len(rows)}")
    print(f"Combined SDF: {result['combined_sdf']}")
    print(f"Workers: {result.get('workers', 1)}")


if __name__ == "__main__":
    main()
