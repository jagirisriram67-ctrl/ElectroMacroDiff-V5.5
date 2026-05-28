from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.docking_prep import prepare_ligand_pdbqt_batch, prepare_receptor_pdbqt
from emd_v5_2_hybrid.registry import register_artifact, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare receptor and ligand PDBQT files using Meeko.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--ligand-limit", type=int, default=None)
    parser.add_argument("--skip-receptor", action="store_true")
    parser.add_argument("--skip-ligands", action="store_true")
    parser.add_argument("--raw-receptor", action="store_true", help="Use raw PDB without cleaning HETATM/altloc records.")
    parser.add_argument("--clean-ligands", action="store_true", help="Remove old ligand PDBQT files before preparing current SDF set.")
    parser.add_argument("--grid-json", default=None)
    parser.add_argument("--ligand-sdf-dir", default=None)
    parser.add_argument("--ligand-pdbqt-dir", default=None)
    parser.add_argument("--ligand-log-dir", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--receptor-pdbqt", default=None)
    parser.add_argument("--receptor-log", default=None)
    parser.add_argument("--workers", type=int, default=1, help="Parallel CPU workers for ligand PDBQT preparation.")
    parser.add_argument("--stage", default="M5_pdbqt_prep")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    summary_path = Path(args.summary_json).resolve() if args.summary_json else base / "06_docking" / "scores" / "pdbqt_preparation_summary.json"
    if summary_path.exists():
        try:
            results = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}
    else:
        results = {}
    grid_path = Path(args.grid_json).resolve() if args.grid_json else base / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing docking grid: {grid_path}. Run 06_prepare_docking_inputs.py first.")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))

    if not args.skip_receptor:
        receptor_result = prepare_receptor_pdbqt(
            receptor_pdb=base / "01_raw_data" / "pdb" / "5AEP.pdb",
            output_pdbqt=Path(args.receptor_pdbqt).resolve() if args.receptor_pdbqt else base / "06_docking" / "receptor" / "jak2_prepared.pdbqt",
            log_path=Path(args.receptor_log).resolve() if args.receptor_log else base / "06_docking" / "receptor" / "mk_prepare_receptor.log",
            center=grid["center"],
            size=grid["size"],
            clean_input=not args.raw_receptor,
        )
        results["receptor"] = receptor_result
        register_artifact(base, args.stage, receptor_result["log_path"], "receptor_prep_log", owner="Student 5")
        if receptor_result["status"] == "prepared":
            register_artifact(base, args.stage, receptor_result["pdbqt"], "receptor_pdbqt", owner="Student 5")

    if not args.skip_ligands:
        ligand_pdbqt_dir = Path(args.ligand_pdbqt_dir).resolve() if args.ligand_pdbqt_dir else base / "06_docking" / "ligands_pdbqt"
        if args.clean_ligands and ligand_pdbqt_dir.exists():
            for path in ligand_pdbqt_dir.glob("CAND_*.pdbqt"):
                path.unlink()
            manifest = ligand_pdbqt_dir / "ligand_pdbqt_manifest.csv"
            if manifest.exists():
                manifest.unlink()
        ligand_result = prepare_ligand_pdbqt_batch(
            sdf_dir=Path(args.ligand_sdf_dir).resolve() if args.ligand_sdf_dir else base / "06_docking" / "ligands_sdf",
            output_dir=ligand_pdbqt_dir,
            log_dir=Path(args.ligand_log_dir).resolve() if args.ligand_log_dir else base / "06_docking" / "scores" / "meeko_ligand_logs",
            limit=args.ligand_limit,
            workers=max(int(args.workers), 1),
        )
        results["ligands"] = ligand_result
        register_artifact(base, args.stage, ligand_result["manifest"], "ligand_pdbqt_manifest", owner="Student 5")

    status = "completed"
    notes = []
    if "receptor" in results:
        notes.append(f"receptor={results['receptor']['status']}")
        if results["receptor"]["status"] != "prepared":
            status = "completed_with_failures"
    if "ligands" in results:
        notes.append(
            f"ligands_prepared={results['ligands']['prepared']}, skipped={results['ligands']['skipped']}, failed={results['ligands']['failed']}, workers={results['ligands'].get('workers', 1)}"
        )
        if results["ligands"]["failed"]:
            status = "completed_with_failures"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    register_artifact(base, args.stage, summary_path, "pdbqt_prep_summary", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status=status,
        input_path=str(Path(args.ligand_sdf_dir).resolve() if args.ligand_sdf_dir else base / "06_docking" / "ligands_sdf"),
        output_path=str(summary_path),
        notes="; ".join(notes),
    )
    print(json.dumps(results, indent=2))
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
