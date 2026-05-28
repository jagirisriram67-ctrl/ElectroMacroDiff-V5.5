from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run
from emd_v5_2_hybrid.schemas import DOCKING_SCORE_COLUMNS

SCORE_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)", re.I)
TABLE_SCORE_RE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)\s+", re.M)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse V5.5 Vina-GPU raw pose outputs.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--branch", default="v5_5_pocket_guided")
    parser.add_argument("--candidates-csv", default=None)
    parser.add_argument("--raw-pose-dir", default=None)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--grid-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--stage", default="V5_5_vina_gpu_parse")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    branch = args.branch
    candidates_path = (
        Path(args.candidates_csv).resolve()
        if args.candidates_csv
        else base / "05_generated_candidates" / branch / "generated_v5_5_pocket_guided.csv"
    )
    raw_pose_dir = (
        Path(args.raw_pose_dir).resolve()
        if args.raw_pose_dir
        else base / "06_docking" / branch / "poses_gpu_raw"
    )
    pose_dir = (
        Path(args.pose_dir).resolve()
        if args.pose_dir
        else base / "06_docking" / branch / "poses"
    )
    grid_path = (
        Path(args.grid_json).resolve()
        if args.grid_json
        else base / "06_docking" / branch / "receptor" / "docking_grid_5AEP_QUP.json"
    )
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else base / "06_docking" / branch / "scores" / "docking_scores_full_vina_gpu_2_1.csv"
    )

    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidates CSV: {candidates_path}")
    if not raw_pose_dir.exists():
        raise FileNotFoundError(f"Missing raw pose directory: {raw_pose_dir}")

    candidates = pd.read_csv(candidates_path)
    smiles_by_id = dict(
        zip(candidates["candidate_id"].astype(str), candidates["canonical_smiles"].astype(str))
    )
    grid = _load_grid(grid_path)
    pose_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    raw_files = sorted(raw_pose_dir.rglob("*.pdbqt"))
    for raw_pose in raw_files:
        candidate_id = _candidate_id_from_pose(raw_pose)
        best_score = parse_score(raw_pose)
        final_pose = pose_dir / f"{candidate_id}_vina_out.pdbqt"
        note = "parsed_vina_gpu_2_1_fast_screen" if best_score is not None else "gpu_pose_no_score_found"
        if best_score is not None:
            shutil.copy2(raw_pose, final_pose)
        rows.append(
            {
                "candidate_id": candidate_id,
                "smiles": smiles_by_id.get(candidate_id, ""),
                "docking_engine": "vina_gpu_2_1",
                "receptor_pdb": "5AEP",
                "grid_center_x": grid["center"].get("x", ""),
                "grid_center_y": grid["center"].get("y", ""),
                "grid_center_z": grid["center"].get("z", ""),
                "grid_size_x": grid["size"].get("x", ""),
                "grid_size_y": grid["size"].get("y", ""),
                "grid_size_z": grid["size"].get("z", ""),
                "best_score": best_score,
                "pose_rank": 1 if best_score is not None else "",
                "pose_file": str(final_pose) if final_pose.exists() else "",
                "pose_image": "",
                "control_or_generated": "generated",
                "notes": note,
            }
        )

    frame = pd.DataFrame(rows, columns=DOCKING_SCORE_COLUMNS)
    frame = frame.sort_values(["best_score", "candidate_id"], na_position="last").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    parsed = int(frame["best_score"].notna().sum())
    summary = {
        "stage": args.stage,
        "status": "completed" if parsed else "completed_no_scores",
        "branch": branch,
        "raw_pose_dir": str(raw_pose_dir),
        "pose_dir": str(pose_dir),
        "output_csv": str(output_csv),
        "raw_pose_files": len(raw_files),
        "parsed_scores": parsed,
        "best_score": None if not parsed else float(frame["best_score"].min()),
        "median_score": None if not parsed else float(frame["best_score"].median()),
        "note": "Parsed Vina-GPU 2.1 raw pose PDBQT outputs from Kaggle GPU-only docking.",
    }
    summary_path = output_csv.parent / "v5_5_vina_gpu_parse_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for path, artifact_type in [
        (output_csv, "v5_5_vina_gpu_docking_scores"),
        (summary_path, "v5_5_vina_gpu_parse_summary"),
    ]:
        register_artifact(base, args.stage, path, artifact_type, owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status=summary["status"],
        input_path=str(raw_pose_dir),
        output_path=f"{output_csv}; {summary_path}",
        molecules_in=len(raw_files),
        molecules_out=parsed,
        notes=f"best={summary['best_score']}; median={summary['median_score']}",
    )
    print(json.dumps(summary, indent=2))


def parse_score(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = SCORE_RE.search(text) or TABLE_SCORE_RE.search(text)
    return float(match.group(1)) if match else None


def _candidate_id_from_pose(path: Path) -> str:
    stem = path.stem
    for suffix in ["_vina_out", "_out"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_grid(path: Path) -> dict:
    if not path.exists():
        return {"center": {"x": "", "y": "", "z": ""}, "size": {"x": "", "y": "", "z": ""}}
    grid = json.loads(path.read_text(encoding="utf-8"))
    grid.setdefault("center", {})
    grid.setdefault("size", {})
    return grid


if __name__ == "__main__":
    main()
