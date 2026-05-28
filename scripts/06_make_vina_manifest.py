from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.docking import write_vina_manifest
from emd_v5_2_hybrid.registry import register_artifact, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a resumable Vina command manifest from ligand PDBQT files.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--center", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--grid-json", default=None, help="Optional grid JSON from 06_prepare_docking_inputs.py")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--ligand-pdbqt-dir", default=None)
    parser.add_argument("--receptor-pdbqt", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--stage", default="M5_docking_manifest")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    if args.grid_json:
        grid_path = Path(args.grid_json)
    else:
        grid_path = base / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json"
    if grid_path.exists():
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        center = (
            float(grid["center"]["x"]),
            float(grid["center"]["y"]),
            float(grid["center"]["z"]),
        )
        size = (
            float(grid["size"]["x"]),
            float(grid["size"]["y"]),
            float(grid["size"]["z"]),
        )
    else:
        if args.center is None or args.size is None:
            raise ValueError("Provide --center/--size or a valid --grid-json file.")
        center = tuple(args.center)
        size = tuple(args.size)

    ligand_pdbqt_dir = Path(args.ligand_pdbqt_dir).resolve() if args.ligand_pdbqt_dir else base / "06_docking" / "ligands_pdbqt"
    ligands = sorted(ligand_pdbqt_dir.glob("*.pdbqt"))
    receptor = Path(args.receptor_pdbqt).resolve() if args.receptor_pdbqt else base / "06_docking" / "receptor" / "jak2_prepared.pdbqt"
    if not receptor.exists():
        raise FileNotFoundError(f"Missing receptor PDBQT: {receptor}")
    manifest = Path(args.manifest_output).resolve() if args.manifest_output else base / "06_docking" / "scores" / "vina_command_manifest.csv"
    write_vina_manifest(
        ligands,
        receptor,
        manifest,
        pose_dir=Path(args.pose_dir).resolve() if args.pose_dir else base / "06_docking" / "poses",
        log_dir=Path(args.log_dir).resolve() if args.log_dir else base / "06_docking" / "scores",
        center=center,
        size=size,
        exhaustiveness=args.exhaustiveness,
    )
    register_artifact(base, args.stage, manifest, "vina_command_manifest", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=str(ligand_pdbqt_dir),
        output_path=str(manifest),
        molecules_in=len(ligands),
        molecules_out=len(ligands),
        notes="Run commands manually or through a controlled notebook loop",
    )
    print(f"Wrote manifest for {len(ligands)} ligands: {manifest}")


if __name__ == "__main__":
    main()
