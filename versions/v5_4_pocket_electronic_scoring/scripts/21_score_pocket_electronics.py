from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.pocket_electronics import score_pocket_electronic_fit, write_pocket_profile
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress
from emd_v5_2_hybrid.ranking import decision_label


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score pocket/electronic fit for docked candidate poses.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--ranking-csv", default=None)
    parser.add_argument("--docking-csv", default=None)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--receptor", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-ranking-csv", default=None)
    parser.add_argument("--profile-json", default=None)
    parser.add_argument("--top-n", type=int, default=0, help="0 means score all rows with pose files.")
    parser.add_argument("--stage", default="V5_4_pocket_electronic_fit")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    ranking_path = (
        Path(args.ranking_csv).resolve()
        if args.ranking_csv
        else base / "08_final_ranking" / "v5_3_model_guided_ranked_candidates.csv"
    )
    docking_path = (
        Path(args.docking_csv).resolve()
        if args.docking_csv
        else base / "06_docking" / "v5_3_model_guided" / "scores" / "docking_scores_full_vina_gpu_2_1.csv"
    )
    pose_dir = (
        Path(args.pose_dir).resolve()
        if args.pose_dir
        else base / "06_docking" / "v5_3_model_guided" / "poses"
    )
    receptor_path = Path(args.receptor).resolve() if args.receptor else base / "06_docking" / "receptor" / "jak2_prepared.pdbqt"
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else base / "06_docking" / "v5_3_model_guided" / "scores" / "pocket_electronic_fit_scores.csv"
    )
    output_ranking = (
        Path(args.output_ranking_csv).resolve()
        if args.output_ranking_csv
        else base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates.csv"
    )
    profile_json = (
        Path(args.profile_json).resolve()
        if args.profile_json
        else base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
    )

    ranking = pd.read_csv(ranking_path)
    docking = pd.read_csv(docking_path) if docking_path.exists() else pd.DataFrame(columns=["candidate_id", "best_score"])
    score_by_id = {}
    if "candidate_id" in docking.columns and "best_score" in docking.columns:
        score_by_id = {
            str(row["candidate_id"]): _safe_float(row.get("best_score"))
            for row in docking.to_dict(orient="records")
        }
    rows_to_score = ranking.sort_values("rank").to_dict(orient="records")
    if args.top_n > 0:
        rows_to_score = rows_to_score[: args.top_n]

    results = []
    first_pose = None
    for row in rows_to_score:
        candidate_id = str(row["candidate_id"])
        pose_file = pose_dir / f"{candidate_id}_vina_out.pdbqt"
        if not pose_file.exists():
            alt_pose = row.get("pose_file", "")
            pose_file = Path(str(alt_pose)).resolve() if alt_pose else pose_file
        if not pose_file.exists():
            continue
        if first_pose is None:
            first_pose = pose_file
        result = score_pocket_electronic_fit(
            candidate_id=candidate_id,
            pose_pdbqt=pose_file,
            receptor_path=receptor_path,
            docking_score=score_by_id.get(candidate_id, _safe_float(row.get("best_score"))),
        )
        results.append(result)

    if not results:
        raise ValueError(f"No pocket/electronic scores produced. Check pose dir: {pose_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(results).sort_values("pocket_electronic_fit_score", ascending=False)
    scores.to_csv(output_csv, index=False)

    if first_pose:
        write_pocket_profile(profile_json, receptor_path, first_pose)

    merged = ranking.merge(
        scores[
            [
                "candidate_id",
                "pocket_electronic_fit_score",
                "electrostatic_score",
                "hbond_score",
                "hydrophobic_score",
                "aromatic_score",
                "hbond_opportunity_pairs",
                "hydrophobic_contacts",
                "aromatic_contacts",
                "polar_contacts",
                "pocket_fit_decision",
                "contact_residues",
            ]
        ],
        on="candidate_id",
        how="left",
    )
    merged["pocket_electronic_fit_score"] = merged["pocket_electronic_fit_score"].fillna(0.0)
    if "pose_score" not in merged.columns:
        merged["pose_score"] = 0.5
    merged["pose_score_with_pocket_electronics"] = (
        0.55 * merged["pose_score"].fillna(0.5).astype(float)
        + 0.45 * merged["pocket_electronic_fit_score"].fillna(0.0).astype(float)
    ).round(6)
    base_final = merged["final_weighted_score"].fillna(0.0).astype(float) if "final_weighted_score" in merged.columns else 0.0
    # Preserve the original score and add a second, pocket-aware ordering.
    merged["pocket_guided_final_score"] = (
        0.78 * base_final
        + 0.22 * merged["pocket_electronic_fit_score"].fillna(0.0).astype(float)
    ).round(6)
    merged = merged.sort_values("pocket_guided_final_score", ascending=False).reset_index(drop=True)
    merged["pocket_guided_rank"] = merged.index + 1
    merged["pocket_guided_decision"] = merged["pocket_guided_final_score"].map(decision_label)
    output_ranking.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_ranking, index=False)

    summary = {
        "stage": args.stage,
        "status": "completed",
        "ranking_csv": str(ranking_path),
        "docking_csv": str(docking_path),
        "pose_dir": str(pose_dir),
        "receptor": str(receptor_path),
        "output_csv": str(output_csv),
        "output_ranking_csv": str(output_ranking),
        "profile_json": str(profile_json),
        "scored_candidates": int(len(scores)),
        "best_pocket_electronic_fit_score": float(scores["pocket_electronic_fit_score"].max()),
        "median_pocket_electronic_fit_score": float(scores["pocket_electronic_fit_score"].median()),
        "top_candidate_by_pocket_fit": str(scores.iloc[0]["candidate_id"]),
        "note": "Pocket/electronic fit is a proxy from PDBQT partial charges and interactions, not quantum electron density.",
    }
    progress_path = base / "00_project_registry" / "progress_v5_4_pocket_electronic_fit.json"
    save_progress(progress_path, summary)
    for path, artifact_type in [
        (output_csv, "pocket_electronic_fit_scores"),
        (output_ranking, "pocket_electronic_guided_ranking"),
        (profile_json, "jak2_pocket_electronic_profile"),
        (progress_path, "pocket_electronic_fit_progress"),
    ]:
        register_artifact(base, args.stage, path, artifact_type, owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=f"{ranking_path}; {docking_path}; {pose_dir}; {receptor_path}",
        output_path=f"{output_csv}; {output_ranking}",
        molecules_in=len(ranking),
        molecules_out=len(scores),
        notes=f"top={summary['top_candidate_by_pocket_fit']}; median_fit={summary['median_pocket_electronic_fit_score']:.4f}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
