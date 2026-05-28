from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.docking import parse_vina_best_score
from emd_v5_2_hybrid.registry import register_artifact, register_run


def read_csv(path: Path):
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def path_exists(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return False
    return Path(text).exists()


def stem_from_path(value: Any) -> str:
    text = str(value).strip()
    return Path(text).stem if text and text.lower() != "nan" else ""


def numeric_count(series) -> int:
    import pandas as pd

    return int(pd.to_numeric(series, errors="coerce").notna().sum())


def selected_final_candidates(ranking):
    if ranking.empty:
        return ranking
    if "selection_tier" in ranking.columns:
        selected = ranking[ranking["selection_tier"] == "final_candidate"].copy()
        if not selected.empty:
            return selected.sort_values("diverse_rank" if "diverse_rank" in selected.columns else "rank").head(5)
    return ranking.sort_values("rank").head(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cross-artifact consistency for the EMD V5.2 project.")
    parser.add_argument("--base", default=str(ROOT))
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    paths = {
        "generated": base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv",
        "sdf_manifest": base / "06_docking" / "ligands_sdf" / "docking_input_manifest.csv",
        "pdbqt_manifest": base / "06_docking" / "ligands_pdbqt" / "ligand_pdbqt_manifest.csv",
        "vina_manifest": base / "06_docking" / "scores" / "vina_command_manifest.csv",
        "docking_scores": base / "06_docking" / "scores" / "docking_scores.csv",
        "pose_sanity": base / "06_docking" / "scores" / "pose_sanity_scores.csv",
        "ranking": base / "08_final_ranking" / "final_ranked_candidates.csv",
        "validation": base / "00_project_registry" / "validation_report.json",
        "final_report": base / "09_reports" / "EMD_V5_2_Hybrid_Final_Report_Draft.md",
        "audit_summary": base / "09_reports" / "final_audit_summary.json",
    }

    generated = read_csv(paths["generated"])
    sdf_manifest = read_csv(paths["sdf_manifest"])
    pdbqt_manifest = read_csv(paths["pdbqt_manifest"])
    vina_manifest = read_csv(paths["vina_manifest"])
    docking = read_csv(paths["docking_scores"])
    pose_sanity = read_csv(paths["pose_sanity"])
    ranking = read_csv(paths["ranking"])
    validation = read_json(paths["validation"])

    failures: list[str] = []
    warnings: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def warn(condition: bool, message: str) -> None:
        if not condition:
            warnings.append(message)

    for name, path in paths.items():
        require(path.exists(), f"Missing required artifact: {name} -> {path}")

    require(not generated.empty and "candidate_id" in generated.columns, "Generated candidate table is empty or missing candidate_id.")
    require(not ranking.empty and "candidate_id" in ranking.columns, "Ranking table is empty or missing candidate_id.")
    require(not docking.empty and "candidate_id" in docking.columns, "Docking score table is empty or missing candidate_id.")

    generated_ids = set(generated.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))
    ranking_ids = set(ranking.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))
    docking_ids = set(docking.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))

    require(len(generated_ids) == len(generated), "Generated candidate IDs are not unique or contain blanks.")
    require(len(ranking_ids) == len(ranking), "Ranking candidate IDs are not unique or contain blanks.")
    require(len(docking_ids) == len(docking), "Docking score candidate IDs are not unique or contain blanks.")
    require(ranking_ids == generated_ids, "Final ranking does not contain exactly the current generated candidate set.")
    require(docking_ids.issubset(generated_ids), "Docking scores contain IDs outside current generated candidates.")
    require(numeric_count(docking["best_score"]) == len(docking), "Docking score table has missing or non-numeric best_score values.")

    sdf_prepared = sdf_manifest[sdf_manifest.get("status", "") == "prepared"] if not sdf_manifest.empty else pd.DataFrame()
    pdbqt_ready_statuses = {"prepared", "already_prepared"}
    pdbqt_prepared = (
        pdbqt_manifest[pdbqt_manifest.get("status", "").isin(pdbqt_ready_statuses)]
        if not pdbqt_manifest.empty
        else pd.DataFrame()
    )
    sdf_ids = set(sdf_prepared.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))
    pdbqt_ids = set(pdbqt_prepared.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str))

    require(sdf_ids.issubset(generated_ids), "Prepared SDF IDs include stale candidates.")
    require(pdbqt_ids.issubset(generated_ids), "Prepared PDBQT IDs include stale candidates.")
    require(pdbqt_ids.issubset(sdf_ids), "Prepared PDBQT IDs are not backed by prepared SDF entries.")

    if not sdf_prepared.empty and "sdf_path" in sdf_prepared.columns:
        missing_sdf = [row["candidate_id"] for _, row in sdf_prepared.iterrows() if not path_exists(row.get("sdf_path"))]
        require(not missing_sdf, f"Prepared SDF files missing for: {missing_sdf[:10]}")
    if not pdbqt_prepared.empty and "pdbqt_path" in pdbqt_prepared.columns:
        missing_pdbqt = [row["candidate_id"] for _, row in pdbqt_prepared.iterrows() if not path_exists(row.get("pdbqt_path"))]
        require(not missing_pdbqt, f"Prepared ligand PDBQT files missing for: {missing_pdbqt[:10]}")

    if not vina_manifest.empty:
        manifest_ids = {stem_from_path(value) for value in vina_manifest.get("ligand_pdbqt", pd.Series(dtype=str))}
        manifest_ids.discard("")
        completed = vina_manifest[vina_manifest.get("status", "") == "completed"]
        require(manifest_ids == docking_ids, "Vina manifest IDs and parsed docking score IDs differ.")
        require(manifest_ids.issubset(generated_ids), "Vina manifest includes IDs outside current generated candidates.")
        require(len(completed) == len(vina_manifest), "Vina manifest has pending or failed rows.")

        bad_completed: list[str] = []
        for _, row in completed.iterrows():
            cid = stem_from_path(row.get("ligand_pdbqt"))
            pose_ok = path_exists(row.get("pose_pdbqt"))
            log_path = Path(str(row.get("log_path", "")))
            log_ok = log_path.exists() and parse_vina_best_score(log_path.read_text(encoding="utf-8", errors="ignore")) is not None
            if not (pose_ok and log_ok):
                bad_completed.append(cid)
        require(not bad_completed, f"Completed Vina rows missing valid pose/log artifacts: {bad_completed[:10]}")

    final_candidates = selected_final_candidates(ranking)
    final_ids = final_candidates.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
    final_scaffolds = final_candidates.get("murcko_scaffold", pd.Series(dtype=str)).dropna().astype(str).tolist()
    require(len(final_ids) == 5, f"Expected 5 final candidates, found {len(final_ids)}.")
    require(len(set(final_ids)) == len(final_ids), "Final candidate IDs are not unique.")
    require(len(set(final_scaffolds)) == len(final_scaffolds), "Final candidates are not scaffold-diverse.")
    require(set(final_ids).issubset(docking_ids), "Not all final candidates have parsed docking scores.")

    if not pose_sanity.empty:
        pose_by_id = pose_sanity.set_index("candidate_id", drop=False) if "candidate_id" in pose_sanity.columns else pd.DataFrame()
        for cid in final_ids:
            require(cid in pose_by_id.index, f"Final candidate missing pose sanity row: {cid}")
            if cid in pose_by_id.index:
                row = pose_by_id.loc[cid]
                if hasattr(row, "iloc"):
                    row = row.iloc[0] if len(getattr(row, "shape", [])) > 0 and getattr(row, "ndim", 1) > 1 else row
                require(str(row.get("pose_decision", "")).lower() == "pass", f"Final candidate pose sanity did not pass: {cid}")
                require(float(row.get("hard_clashes_lt_1_8A", 1)) == 0, f"Final candidate has hard clashes: {cid}")
                require(path_exists(row.get("pose_file")), f"Final candidate pose file missing: {cid}")
                require(path_exists(row.get("pose_image")), f"Final candidate pose image missing: {cid}")

    for cid in final_ids:
        require((base / "09_reports" / "candidate_cards" / f"{cid}_card.md").exists(), f"Candidate card missing: {cid}")
        require((base / "09_reports" / "molecule_images" / f"{cid}.png").exists(), f"2D molecule image missing: {cid}")

    validation_summary = validation.get("summary", {})
    require(validation_summary.get("failed", 0) == 0, "Validation report contains failed checks.")
    warn(validation_summary.get("passed", 0) >= 48, "Validation report has fewer checks than expected.")

    counts = {
        "generated_candidates": len(generated),
        "prepared_sdf": int(len(sdf_prepared)),
        "prepared_pdbqt": int(len(pdbqt_prepared)),
        "vina_manifest_rows": int(len(vina_manifest)),
        "docking_score_rows": int(len(docking)),
        "docking_numeric_scores": numeric_count(docking["best_score"]) if "best_score" in docking.columns else 0,
        "ranked_candidates": int(len(ranking)),
        "final_candidates": len(final_ids),
        "pose_sanity_rows": int(len(pose_sanity)),
        "validation_passed": int(validation_summary.get("passed", 0) or 0),
        "validation_failed": int(validation_summary.get("failed", 0) or 0),
    }

    top_final = []
    for _, row in final_candidates.iterrows():
        top_final.append(
            {
                "rank": int(row.get("rank")) if pd.notna(row.get("rank")) else None,
                "diverse_rank": int(row.get("diverse_rank")) if pd.notna(row.get("diverse_rank")) else None,
                "candidate_id": row.get("candidate_id"),
                "source_generator": row.get("source_generator"),
                "docking_score": float(row.get("docking_score")) if pd.notna(row.get("docking_score")) else None,
                "final_weighted_score": float(row.get("final_weighted_score")) if pd.notna(row.get("final_weighted_score")) else None,
                "pose_decision": row.get("pose_decision"),
                "main_risk": row.get("main_risk"),
            }
        )

    report = {
        "project": "ElectroMacroDiff V5.2 Hybrid",
        "base": str(base),
        "status": "pass" if not failures else "fail",
        "counts": counts,
        "final_candidate_ids": final_ids,
        "top_final_candidates": top_final,
        "failures": failures,
        "warnings": warnings,
    }

    output_dir = base / "09_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "consistency_audit.json"
    output_md = output_dir / "consistency_audit.md"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Consistency Audit",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "## Counts",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in counts.items())
    lines.extend(["", "## Final Candidates", ""])
    for row in top_final:
        lines.append(
            f"- Diverse rank {row['diverse_rank']}: {row['candidate_id']} "
            f"(dock {row['docking_score']}, final {row['final_weighted_score']}, pose {row['pose_decision']})"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in failures)
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    register_artifact(base, "M8_consistency_audit", output_json, "consistency_audit_json", owner="Student 1", status=report["status"])
    register_artifact(base, "M8_consistency_audit", output_md, "consistency_audit_markdown", owner="Student 1", status=report["status"])
    register_run(
        base,
        stage="M8_consistency_audit",
        status=report["status"],
        output_path=str(output_json),
        molecules_in=counts["generated_candidates"],
        molecules_out=counts["final_candidates"],
        notes=f"{len(failures)} failures, {len(warnings)} warnings",
    )

    print(json.dumps({"status": report["status"], "counts": counts, "failures": failures, "warnings": warnings}, indent=2))
    print(f"Consistency audit JSON: {output_json}")
    print(f"Consistency audit Markdown: {output_md}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
