from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run


def dataframe_to_markdown(frame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        values = [str(row.get(column, "")).replace("|", "/") for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a markdown final report draft from verified artifacts.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    report_dir = base / "09_reports"
    ranking = pd.read_csv(base / "08_final_ranking" / "final_ranked_candidates.csv").sort_values("rank")
    metrics = pd.read_csv(base / "05_generated_candidates" / "merged" / "generation_comparison_metrics.csv")
    docking_summary = json.loads((base / "06_docking" / "scores" / "docking_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((base / "00_project_registry" / "validation_report.json").read_text(encoding="utf-8"))
    audit = json.loads((base / "09_reports" / "final_audit_summary.json").read_text(encoding="utf-8"))
    selected = ranking[ranking.get("selection_tier", "") == "final_candidate"].copy()
    top = selected.head(args.top_n) if not selected.empty else ranking.head(args.top_n)

    report_path = report_dir / "EMD_V5_2_Hybrid_Final_Report_Draft.md"
    tpp_path = report_dir / "EMD_V5_2_Hybrid_TPP_Summary.md"

    lines = [
        "# ElectroMacroDiff V5.2 Hybrid Final Report Draft",
        "",
        "## Abstract",
        "",
        (
            "ElectroMacroDiff V5.2 Hybrid is a low-resource, checkpointed computational "
            "drug-discovery pipeline for JAK2-focused macrocycle/constrained inhibitor "
            "candidate generation and prioritization. The workflow combines curated JAK2 "
            "ligand data, RDKit descriptor/graph preparation, a custom SE(3)-aware flow "
            "matching debug-trained generator path, RDKit/SELFIES baseline generation, "
            "AutoDock Vina docking, ADMET/synthesis proxy scoring, and consensus ranking."
        ),
        "",
        "## Scientific Boundary",
        "",
        (
            "The proposed molecules are computational hypotheses for experimental follow-up. "
            "Docking scores, ADMET proxies, and synthesizability proxies are prioritization "
            "signals, not experimental proof of potency, selectivity, safety, or synthesis."
        ),
        "",
        "## Dataset And Preparation",
        "",
        f"- Curated JAK2 ligands: {audit['artifact_counts']['curated_ligands']}",
        f"- Ligand feature rows: {audit['artifact_counts']['ligand_features']}",
        "- Primary receptor: PDB 5AEP, JAK2 kinase domain",
        "- Docking grid source: co-crystallized QUP ligand in 5AEP",
        "",
        "## Candidate Generation",
        "",
        f"- Filtered generated candidates: {audit['artifact_counts']['generated_candidates']}",
        "",
        dataframe_to_markdown(metrics),
        "",
        "## SE(3) Model Status",
        "",
        (
            "The custom SE(3)-aware flow-matching implementation passed the critical debug gate: "
            "graph loading, batch collation, forward pass, finite loss, backward pass, optimizer "
            "step, checkpoint save, and checkpoint reload. In the current sprint it should be "
            "presented as a working architecture/debug-trained generator path unless a longer "
            "training run is added."
        ),
        "",
        "## Docking",
        "",
        f"- Vina-scored ligands: {docking_summary['num_scored']} / {docking_summary['num_manifest_ligands']}",
        f"- Best Vina score: {docking_summary['best_score_min']} kcal/mol",
        f"- Median Vina score: {docking_summary['best_score_median']} kcal/mol",
        f"- Weakest parsed Vina score: {docking_summary['best_score_max']} kcal/mol",
        "",
        "## Final Diversity-Selected Top Candidates",
        "",
    ]
    for row in top.to_dict(orient="records"):
        lines.extend(
            [
                f"### Rank {row['rank']}: {row['candidate_id']}",
                "",
                f"- Source: {row.get('source_generator', '')}",
                f"- SMILES: `{row.get('smiles', row.get('canonical_smiles', ''))}`",
                f"- Docking score: {row.get('docking_score', '')} kcal/mol",
                f"- Final weighted score: {row.get('final_weighted_score', '')}",
                f"- Decision: {row.get('decision', '')}",
                f"- Main risk: {row.get('main_risk', '')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation",
            "",
        f"- Validation checks passed: {validation['summary']['passed']} / {validation['summary']['total']}",
            f"- Failed checks: {validation['summary']['failed']}",
            "",
            "## Limitations",
            "",
            "- Vina docking approximates pose/affinity and must be followed by pose inspection.",
            "- ADMET and safety estimates are descriptor/proxy based.",
            "- SE(3) model currently has debug-gate evidence; extended training would strengthen the research claim.",
            "- Candidate synthesis and biological activity require wet-lab validation.",
            "",
            "## References To Cite",
            "",
            "- AutoDock Vina manual and releases: https://vina.scripps.edu/manual/",
            "- AutoDock Vina GitHub releases: https://github.com/ccsb-scripps/AutoDock-Vina/releases",
            "- Meeko documentation: https://meeko.readthedocs.io/",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    tpp_lines = ["# EMD V5.2 Hybrid TPP Summary", ""]
    for row in top.to_dict(orient="records"):
        image_path = report_dir / "molecule_images" / f"{row['candidate_id']}.png"
        tpp_lines.extend(
            [
                f"## {row['candidate_id']}",
                "",
                f"- Rank: {row['rank']}",
                f"- Source: {row.get('source_generator', '')}",
                f"- Docking score: {row.get('docking_score', '')}",
                f"- QED: {row.get('qed', '')}",
                f"- SA score/proxy: {row.get('sa_score', '')}",
                f"- ADMET score: {row.get('admet_score', '')}",
                f"- Safety proxy score: {row.get('safety_proxy_score', '')}",
                f"- Main risk: {row.get('main_risk', '')}",
                f"- Image: `{image_path}`",
                "",
            ]
        )
    tpp_path.write_text("\n".join(tpp_lines), encoding="utf-8")

    register_artifact(base, "M8_reporting", report_path, "final_report_markdown_draft", owner="Student 1")
    register_artifact(base, "M8_reporting", tpp_path, "tpp_summary_markdown", owner="Student 1")
    register_run(
        base,
        stage="M8_report_draft",
        status="completed",
        input_path=str(base / "08_final_ranking" / "final_ranked_candidates.csv"),
        output_path=str(report_path),
        molecules_in=len(ranking),
        molecules_out=len(top),
        notes="Markdown report and TPP draft generated from verified artifacts",
    )
    print(f"Report draft: {report_path}")
    print(f"TPP summary: {tpp_path}")


if __name__ == "__main__":
    main()
