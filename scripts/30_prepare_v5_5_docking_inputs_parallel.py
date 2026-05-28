from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.docking import infer_grid_from_ligand, write_grid_json
from emd_v5_2_hybrid.registry import register_artifact, register_run


def _safe_prop_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return text or "value"


def _sdf_record_from_row(row: dict, random_seed: int) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    candidate_id = str(row.get("candidate_id", "candidate"))
    smiles = str(row.get("smiles", row.get("canonical_smiles", "")))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"candidate_id": candidate_id, "smiles": smiles, "sdf": "", "status": "failed_parse"}

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(
            mol,
            randomSeed=int(random_seed),
            useRandomCoords=True,
            maxAttempts=500,
        )
    if status != 0:
        return {"candidate_id": candidate_id, "smiles": smiles, "sdf": "", "status": "failed_embed"}

    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
        except Exception:
            pass

    mol.SetProp("_Name", candidate_id)
    block = Chem.MolToMolBlock(mol)
    properties: list[str] = []
    for key, value in row.items():
        if value is None:
            continue
        text = str(value)
        if not text or text.lower() == "nan":
            continue
        properties.append(f">  <{_safe_prop_name(key)}>\n{text}\n")
    sdf = block + "\n" + "\n".join(properties) + "\n$$$$\n"
    return {"candidate_id": candidate_id, "smiles": smiles, "sdf": sdf, "status": "prepared"}


def clean_sdf_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["CAND_*.sdf", "candidates_for_docking.sdf", "docking_input_manifest.csv"]:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel V5.5 docking SDF/grid preparation for Kaggle.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--top-n", type=int, default=2000)
    parser.add_argument("--padding", type=float, default=8.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--ligand-sdf-dir", required=True)
    parser.add_argument("--combined-sdf", required=True)
    parser.add_argument("--grid-json", required=True)
    parser.add_argument("--num-workers", type=int, default=0, help="0 uses max(os.cpu_count()-1, 1).")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--stage", default="V5_5_parallel_docking_prep")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    input_path = Path(args.input_csv).resolve()
    ligand_sdf_dir = Path(args.ligand_sdf_dir).resolve()
    combined_sdf = Path(args.combined_sdf).resolve()
    grid_json = Path(args.grid_json).resolve()
    workers = int(args.num_workers) if int(args.num_workers) > 0 else max((os.cpu_count() or 2) - 1, 1)

    frame = pd.read_csv(input_path)
    if "rank" in frame.columns:
        frame = frame.sort_values("rank", ascending=True)
    rows = frame.head(int(args.top_n)).to_dict(orient="records")

    grid = infer_grid_from_ligand(base / "01_raw_data" / "pdb" / "5AEP.pdb", padding=float(args.padding))
    grid_path = write_grid_json(grid, grid_json)

    if args.clean:
        clean_sdf_dir(ligand_sdf_dir)
    ligand_sdf_dir.mkdir(parents=True, exist_ok=True)
    combined_sdf.parent.mkdir(parents=True, exist_ok=True)

    results_by_id: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_id = {
            pool.submit(_sdf_record_from_row, row, int(args.random_seed) + index): str(row.get("candidate_id", "candidate"))
            for index, row in enumerate(rows)
        }
        completed = 0
        total = len(future_to_id)
        for future in as_completed(future_to_id):
            candidate_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"candidate_id": candidate_id, "smiles": "", "sdf": "", "status": f"failed_exception:{type(exc).__name__}"}
            results_by_id[candidate_id] = result
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"Prepared/attempted SDF conformers: {completed} / {total}", flush=True)

    manifest_rows = []
    prepared = 0
    failed = 0
    with combined_sdf.open("w", encoding="utf-8") as combined:
        for row in rows:
            candidate_id = str(row.get("candidate_id", "candidate"))
            result = results_by_id.get(candidate_id, {"candidate_id": candidate_id, "smiles": "", "sdf": "", "status": "missing_result"})
            status = str(result["status"])
            sdf_path = ""
            if status == "prepared":
                sdf_path_obj = ligand_sdf_dir / f"{candidate_id}.sdf"
                sdf_path_obj.write_text(str(result["sdf"]), encoding="utf-8")
                combined.write(str(result["sdf"]))
                sdf_path = str(sdf_path_obj)
                prepared += 1
            else:
                failed += 1
            manifest_rows.append(
                {
                    "candidate_id": candidate_id,
                    "smiles": result.get("smiles", str(row.get("canonical_smiles", ""))),
                    "sdf_path": sdf_path,
                    "status": status,
                }
            )

    manifest_path = ligand_sdf_dir / "docking_input_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_id", "smiles", "sdf_path", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    register_artifact(base, args.stage, grid_path, "docking_grid_json", owner="Student 5")
    register_artifact(base, args.stage, combined_sdf, "candidate_ligands_sdf", owner="Student 5")
    register_artifact(base, args.stage, manifest_path, "docking_input_manifest", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed" if prepared else "completed_no_ligands",
        input_path=str(input_path),
        output_path=str(combined_sdf),
        molecules_in=len(rows),
        molecules_out=prepared,
        notes=f"workers={workers}; failed={failed}; grid from {grid['reference_ligand']}",
    )
    print(f"Grid: {grid_path}")
    print(f"Center: {grid['center']}")
    print(f"Size: {grid['size']}")
    print(f"Prepared SDF ligands: {prepared} / {len(rows)}")
    print(f"Combined SDF: {combined_sdf}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
