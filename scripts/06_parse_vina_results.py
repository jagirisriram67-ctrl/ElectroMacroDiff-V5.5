from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.docking import parse_vina_best_score
from emd_v5_2_hybrid.registry import register_artifact, register_run
from emd_v5_2_hybrid.schemas import DOCKING_SCORE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Vina log files into docking_scores.csv.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--candidates-csv", default=None)
    parser.add_argument("--grid-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--stage", default="M5_docking_parse")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else base / "06_docking" / "scores" / "vina_command_manifest.csv"
    candidates_path = (
        Path(args.candidates_csv).resolve()
        if args.candidates_csv
        else base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
    )
    grid_path = Path(args.grid_json).resolve() if args.grid_json else base / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json"
    output_path = Path(args.output_csv).resolve() if args.output_csv else base / "06_docking" / "scores" / "docking_scores.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    candidates = pd.read_csv(candidates_path)
    smiles_by_id = dict(zip(candidates["candidate_id"].astype(str), candidates["canonical_smiles"].astype(str)))

    rows = []
    for row in manifest.to_dict(orient="records"):
        ligand_path = Path(str(row.get("ligand_pdbqt", "")))
        candidate_id = ligand_path.stem
        log_path = Path(str(row.get("log_path", "")))
        pose_path = str(row.get("pose_pdbqt", ""))
        best_score = None
        status_note = "missing_log"
        if log_path.exists():
            best_score = parse_vina_best_score(log_path.read_text(encoding="utf-8", errors="ignore"))
            status_note = "parsed" if best_score is not None else "no_score_found"
        rows.append(
            {
                "candidate_id": candidate_id,
                "smiles": smiles_by_id.get(candidate_id, ""),
                "docking_engine": "vina",
                "receptor_pdb": "5AEP",
                "grid_center_x": "",
                "grid_center_y": "",
                "grid_center_z": "",
                "grid_size_x": "",
                "grid_size_y": "",
                "grid_size_z": "",
                "best_score": best_score,
                "pose_rank": 1 if best_score is not None else "",
                "pose_file": pose_path,
                "pose_image": "",
                "control_or_generated": "generated",
                "notes": status_note,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=DOCKING_SCORE_COLUMNS)
    if grid_path.exists():
        import json

        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        frame["grid_center_x"] = grid["center"]["x"]
        frame["grid_center_y"] = grid["center"]["y"]
        frame["grid_center_z"] = grid["center"]["z"]
        frame["grid_size_x"] = grid["size"]["x"]
        frame["grid_size_y"] = grid["size"]["y"]
        frame["grid_size_z"] = grid["size"]["z"]
    frame.to_csv(output_path, index=False)

    parsed = int(frame["best_score"].notna().sum())
    register_artifact(base, args.stage, output_path, "docking_scores", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed" if parsed else "completed_no_scores",
        input_path=str(manifest_path),
        output_path=str(output_path),
        molecules_in=len(frame),
        molecules_out=parsed,
        notes=f"Parsed {parsed} Vina scores",
    )
    print(f"Wrote docking scores: {output_path}")
    print(f"Parsed scores: {parsed} / {len(frame)}")


if __name__ == "__main__":
    main()
