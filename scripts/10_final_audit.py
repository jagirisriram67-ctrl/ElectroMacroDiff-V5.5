from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run


def count_rows(path: Path) -> int:
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return 0
    return int(len(pd.read_csv(path)))


def read_json_or_empty(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final audit summary for the EMD V5.2 pipeline.")
    parser.add_argument("--base", default=str(ROOT))
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    paths = {
        "curated_ligands": base / "02_curated_data" / "jak2_curated_ligands.csv",
        "ligand_features": base / "03_features" / "ligand_features.csv",
        "generated_candidates": base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv",
        "docking_scores": base / "06_docking" / "scores" / "docking_scores.csv",
        "final_ranking": base / "08_final_ranking" / "final_ranked_candidates.csv",
        "validation_report": base / "00_project_registry" / "validation_report.json",
        "docking_summary": base / "06_docking" / "scores" / "docking_summary.json",
    }

    docking = pd.read_csv(paths["docking_scores"]) if paths["docking_scores"].exists() else pd.DataFrame()
    ranking = pd.read_csv(paths["final_ranking"]) if paths["final_ranking"].exists() else pd.DataFrame()
    validation = read_json_or_empty(paths["validation_report"])
    docking_summary = read_json_or_empty(paths["docking_summary"])

    scored = pd.to_numeric(docking.get("best_score", pd.Series(dtype=float)), errors="coerce") if not docking.empty else pd.Series(dtype=float)
    if not ranking.empty and "selection_tier" in ranking.columns:
        selected = ranking[ranking["selection_tier"] == "final_candidate"].sort_values("rank").head(5)
        if selected.empty:
            selected = ranking.sort_values("rank").head(5)
    else:
        selected = ranking.sort_values("rank").head(5) if not ranking.empty else ranking
    top5 = selected.to_dict(orient="records") if not ranking.empty else []

    audit = {
        "project": "ElectroMacroDiff V5.2 Hybrid",
        "artifact_counts": {
            "curated_ligands": count_rows(paths["curated_ligands"]),
            "ligand_features": count_rows(paths["ligand_features"]),
            "generated_candidates": count_rows(paths["generated_candidates"]),
            "docking_rows": count_rows(paths["docking_scores"]),
            "docking_numeric_scores": int(scored.notna().sum()) if not scored.empty else 0,
            "final_ranked_candidates": count_rows(paths["final_ranking"]),
        },
        "docking_summary": docking_summary,
        "validation_summary": validation.get("summary", {}),
        "top5_diversity_selected": [
            {
                "rank": row.get("rank"),
                "candidate_id": row.get("candidate_id"),
                "source_generator": row.get("source_generator"),
                "docking_score": row.get("docking_score"),
                "final_weighted_score": row.get("final_weighted_score"),
                "decision": row.get("decision"),
                "selection_tier": row.get("selection_tier"),
                "main_risk": row.get("main_risk"),
            }
            for row in top5
        ],
        "scientific_boundary": (
            "All candidates are computational hypotheses. Vina scores, ADMET proxies, and SA proxies "
            "are prioritization signals, not experimental proof of potency, selectivity, safety, or synthesis."
        ),
    }

    audit_dir = base / "09_reports"
    audit_json = audit_dir / "final_audit_summary.json"
    audit_md = audit_dir / "final_audit_summary.md"
    audit_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# Final Audit Summary",
        "",
        "## Counts",
        "",
    ]
    for key, value in audit["artifact_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Docking", ""])
    for key in ["num_manifest_ligands", "num_scored", "best_score_min", "best_score_median", "best_score_max", "top_candidate_id", "top_candidate_score"]:
        if key in docking_summary:
            lines.append(f"- {key}: {docking_summary[key]}")
    lines.extend(["", "## Top 5", ""])
    for row in audit["top5_diversity_selected"]:
        lines.append(
            f"- Rank {row['rank']}: {row['candidate_id']} ({row['source_generator']}), "
            f"docking {row['docking_score']}, final score {row['final_weighted_score']}, {row['decision']}"
        )
    lines.extend(["", "## Boundary", "", audit["scientific_boundary"], ""])
    audit_md.write_text("\n".join(lines), encoding="utf-8")

    register_artifact(base, "M8_final_audit", audit_json, "final_audit_json", owner="Student 1")
    register_artifact(base, "M8_final_audit", audit_md, "final_audit_markdown", owner="Student 1")
    register_run(
        base,
        stage="M8_final_audit",
        status="completed",
        output_path=str(audit_json),
        molecules_in=audit["artifact_counts"]["generated_candidates"],
        molecules_out=audit["artifact_counts"]["docking_numeric_scores"],
        notes="Final audit summary generated",
    )
    print(json.dumps(audit["artifact_counts"], indent=2))
    print(f"Audit JSON: {audit_json}")
    print(f"Audit Markdown: {audit_md}")


if __name__ == "__main__":
    main()
