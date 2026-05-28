from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.benchmarking import (
    MED_REFERENCE,
    compare_to_reference,
    summarize_candidate_csv,
    summarize_docking_csv,
    summarize_pose_csv,
    write_json,
)
from emd_v5_2_hybrid.registry import register_artifact, register_run


def _percent(value) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value):.2f}%"


def _number(value) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def apply_gap_summary(row: dict, summary_path: Path) -> dict:
    if not summary_path.exists():
        return row
    gap_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    linker_novelty = gap_summary.get("linker_novelty_percent")
    if linker_novelty is not None:
        row["linker_novelty_percent"] = linker_novelty
    raw_validity = gap_summary.get("raw_attempt_validity_percent")
    if raw_validity is not None:
        row["validity_percent"] = raw_validity
        row["benchmark_scope"] = "raw_attempt_logged"
    row["attempt_log_exists"] = bool(gap_summary.get("attempt_log_exists", False))
    row["raw_attempt_rows"] = int(gap_summary.get("raw_attempt_rows", 0) or 0)
    row["valid_output_rows"] = int(gap_summary.get("valid_output_rows", 0) or 0)
    row["attempt_status_counts"] = gap_summary.get("status_counts", {})
    row["interpretation_note"] = (
        "Includes measured linker novelty and raw-attempt validity when attempt logs are available; "
        "still not paper-equivalent unless generated from a fixed MED-style test split and sample size."
    )
    return row


def build_report(
    candidate_rows: list[dict],
    docking_rows: list[dict],
    pose_rows: list[dict],
    med_comparison: dict,
    active_branch: str,
    output_path: Path,
) -> Path:
    med = MED_REFERENCE["MED"]
    best_emd = next(
        (row for row in candidate_rows if row["branch"] == active_branch),
        candidate_rows[-1],
    )
    reference_rows = []
    for name, values in MED_REFERENCE.items():
        reference_rows.append(
            [
                name,
                _percent(values["validity_percent"]),
                _percent(values["uniqueness_percent"]),
                _percent(values["macrocyclization_percent"]),
                _percent(values["linker_novelty_percent"]),
            ]
        )

    candidate_table = []
    for row in candidate_rows:
        candidate_table.append(
            [
                row["branch"],
                row.get("benchmark_scope", "filtered_survivor"),
                str(row["num_candidates"]),
                _percent(row["validity_percent"]),
                _percent(row["uniqueness_percent"]),
                _percent(row["macrocyclization_percent"]),
                _percent(row["novel_molecule_percent"]),
                _number(row["median_qed"]),
                _number(row["median_sa_score"]),
            ]
        )

    docking_table = []
    for row in docking_rows:
        docking_table.append(
            [
                row["branch"],
                f"{row['scored_rows']} / {row['docking_rows']}",
                _percent(row["coverage_percent"]),
                _number(row["best_score"]),
                _number(row["median_score"]),
                _number(row["mean_score"]),
            ]
        )

    pose_table = []
    for row in pose_rows:
        pose_table.append(
            [
                row["branch"],
                f"{row['pose_pass_rows']} / {row['pose_rows']}",
                _percent(row["pose_pass_percent"]),
            ]
        )

    comparison = med_comparison["comparisons"]
    status_rows = [
        ["Validity", _percent(best_emd["validity_percent"]), _percent(med["validity_percent"]), comparison["validity_percent"]],
        [
            "Uniqueness",
            _percent(best_emd["uniqueness_percent"]),
            _percent(med["uniqueness_percent"]),
            comparison["uniqueness_percent"],
        ],
        [
            "Macrocyclization",
            _percent(best_emd["macrocyclization_percent"]),
            _percent(med["macrocyclization_percent"]),
            comparison["macrocyclization_percent"],
        ],
        [
            "Linker novelty",
            _percent(best_emd["linker_novelty_percent"]),
            _percent(med["linker_novelty_percent"]),
            comparison["linker_novelty_percent"],
        ],
    ]

    text = "\n".join(
        [
            "# EMD V5.3 Benchmark Report",
            "",
            "This report benchmarks ElectroMacroDiff V5.3 against MED-style macrocycle generation metrics using local artifacts plus sidecar Kaggle rerun outputs. The frozen 114-candidate branch remains the project reference branch, while Kaggle reruns are treated as evidence-upgrade branches rather than replacements.",
            "",
            "## Current Decision",
            "",
            "ElectroMacroDiff is currently best described as a low-resource macrocycle generation plus pocket-electronic prioritization system for JAK2. It is not yet a fully protein-conditioned generator. Claims against MED should remain conservative until a logged rerun closes the raw-attempt validity gap and linker novelty is re-measured on the new branch.",
            "",
            "## Published Reference Metrics",
            "",
            markdown_table(
                ["Method", "Validity", "Uniqueness", "Macrocyclization", "Linker novelty"],
                reference_rows,
            ),
            "",
            "Source: Macro-Equi-Diff PDF Table 1, extracted into `09_reports/med_comparison_assets/macro_equidiff_extracted_text.txt` and summarized by `scripts/12_build_med_vs_emd_doc.py`.",
            "",
            "## EMD Candidate Metrics",
            "",
            markdown_table(
                [
                    "Branch",
                    "Scope",
                    "Candidates",
                    "Validity",
                    "Uniqueness",
                    "Macrocycle",
                    "Novel molecules",
                    "Median QED",
                    "Median SA proxy",
                ],
                candidate_table,
            ),
            "",
            "## MED Comparison Status",
            "",
            markdown_table(["Metric", active_branch, "MED", "Status"], status_rows),
            "",
            med_comparison["claim_note"],
            "",
            "## Docking Benchmark",
            "",
            markdown_table(
                ["Branch", "Scored", "Coverage", "Best Vina", "Median Vina", "Mean Vina"],
                docking_table,
            ),
            "",
            "Docking scores are AutoDock Vina proxy scores against JAK2 PDB `5AEP`; more negative is better. They are not experimental affinity.",
            "",
            "## Pose Sanity",
            "",
            markdown_table(["Branch", "Passed", "Pass rate"], pose_table),
            "",
            "## Free-Cost Path To A Defensible Benchmark Claim",
            "",
            "1. Run the sidecar benchmark branches on Kaggle with GPU when available; hardware and quota can vary by account and week.",
            "2. Use the raw attempt log from `scripts/17_generate_v5_3_model_guided_macrocycles.py` for each rerun branch.",
            "3. Use `scripts/20_measure_v5_3_benchmark_gaps.py` after each rerun to measure raw-attempt validity and linker novelty.",
            "4. Run the same fixed benchmark split three times with fixed seeds and report mean plus standard deviation.",
            "5. Only claim MED is beaten after EMD exceeds MED on validity, uniqueness, macrocyclization, and linker novelty using the same raw-attempt style.",
            "",
            "## Current Target Bar",
            "",
            "- Validity target: above `93.82%`.",
            "- Uniqueness target: above `99.94%`.",
            "- Macrocyclization target: above `99.92%`.",
            "- Linker novelty target: above `82.81%`.",
            "- JAK2 docking target: improve the branch best/median Vina scores while preserving pose sanity and ADMET/synthesis filters.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5.3 MED-style benchmark report.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base / "09_reports" / "v5_3_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_inputs = [
        (
            "selfies",
            base / "05_generated_candidates" / "selfies" / "generated_selfies_filtered.csv",
            "filtered_survivor",
        ),
        (
            "rdkit",
            base / "05_generated_candidates" / "rdkit" / "generated_rdkit_filtered.csv",
            "filtered_survivor",
        ),
        (
            "macrocycle_linker",
            base / "05_generated_candidates" / "macrocycle_linker" / "generated_macrocycle_linker_filtered.csv",
            "filtered_survivor",
        ),
        (
            "frozen_v5_3_baseline",
            base
            / "05_generated_candidates"
            / "model_guided_macrocycle"
            / "generated_v5_3_model_guided_macrocycles.csv",
            "filtered_survivor",
        ),
        (
            "kaggle_logged_rerun",
            base
            / "05_generated_candidates"
            / "model_guided_macrocycle"
            / "generated_v5_3_model_guided_macrocycles_kaggle_logged.csv",
            "filtered_survivor",
        ),
        (
            "kaggle_diverse_rerun",
            base
            / "05_generated_candidates"
            / "model_guided_macrocycle"
            / "generated_v5_3_model_guided_macrocycles_kaggle_diverse.csv",
            "filtered_survivor",
        ),
        (
            "kaggle_v5_5_pocket_guided",
            base
            / "05_generated_candidates"
            / "v5_5_pocket_guided"
            / "generated_v5_5_pocket_guided.csv",
            "filtered_survivor",
        ),
        (
            "merged_with_v5_3",
            base
            / "05_generated_candidates"
            / "merged"
            / "generated_merged_with_v5_3_model_guided.csv",
            "filtered_survivor",
        ),
    ]
    candidate_rows = [
        summarize_candidate_csv(path, branch, scope)
        for branch, path, scope in candidate_inputs
        if path.exists()
    ]
    gap_summary_paths = {
        "frozen_v5_3_baseline": base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_benchmark_gap_summary.json",
        "kaggle_logged_rerun": base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_benchmark_gap_summary_kaggle_logged.json",
        "kaggle_diverse_rerun": base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_benchmark_gap_summary_kaggle_diverse.json",
        "kaggle_v5_5_pocket_guided": base / "05_generated_candidates" / "v5_5_pocket_guided" / "v5_5_pocket_guided_benchmark_gap_summary.json",
    }
    candidate_rows = [apply_gap_summary(row, gap_summary_paths.get(row["branch"], Path("__missing__"))) for row in candidate_rows]

    docking_inputs = [
        ("v5_2_macrocycle_campaign", base / "06_docking" / "scores" / "docking_scores.csv"),
        (
            "kaggle_v5_5_pocket_guided",
            base
            / "06_docking"
            / "v5_5_pocket_guided"
            / "scores"
            / "docking_scores_full_vina_gpu_2_1.csv",
        ),
        (
            "v5_3_model_guided_vina_gpu_2_1",
            base
            / "06_docking"
            / "v5_3_model_guided"
            / "scores"
            / "docking_scores_full_vina_gpu_2_1.csv",
        ),
        (
            "v5_3_model_guided_full_exh16",
            base
            / "06_docking"
            / "v5_3_model_guided"
            / "scores"
            / "docking_scores_partial_full_exh16.csv",
        ),
        (
            "v5_3_model_guided_fast_exh1",
            base
            / "06_docking"
            / "v5_3_model_guided_fast"
            / "scores"
            / "docking_scores.csv",
        ),
    ]
    docking_rows = [summarize_docking_csv(path, branch) for branch, path in docking_inputs if path.exists()]
    if not docking_rows:
        docking_rows = [summarize_docking_csv(Path("__missing__"), "no_docking_scores")]

    pose_inputs = [
        ("v5_2_macrocycle_campaign", base / "06_docking" / "scores" / "pose_sanity_scores.csv"),
        (
            "kaggle_v5_5_pocket_guided",
            base / "06_docking" / "v5_5_pocket_guided" / "scores" / "pose_sanity_scores.csv",
        ),
        (
            "v5_3_model_guided_vina_gpu_2_1",
            base / "06_docking" / "v5_3_model_guided" / "scores" / "pose_sanity_scores.csv",
        ),
    ]
    pose_rows = [summarize_pose_csv(path, branch) for branch, path in pose_inputs if path.exists()]
    if not pose_rows:
        pose_rows = [summarize_pose_csv(Path("__missing__"), "no_pose_sanity")]

    preferred_branches = ["kaggle_v5_5_pocket_guided", "kaggle_diverse_rerun", "kaggle_logged_rerun", "frozen_v5_3_baseline"]
    active_row = None
    for branch in preferred_branches:
        active_row = next((row for row in candidate_rows if row["branch"] == branch), None)
        if active_row is not None:
            break
    if active_row is None:
        active_row = candidate_rows[-1]
    med_comparison = compare_to_reference(active_row, MED_REFERENCE["MED"])

    candidate_metrics_csv = output_dir / "emd_candidate_benchmark_metrics.csv"
    docking_metrics_csv = output_dir / "emd_docking_benchmark_metrics.csv"
    pose_metrics_csv = output_dir / "emd_pose_benchmark_metrics.csv"
    reference_csv = output_dir / "reference_med_style_metrics.csv"
    summary_json = output_dir / "emd_v5_3_benchmark_summary.json"
    report_md = output_dir / "EMD_V5_3_Benchmark_Report.md"

    pd.DataFrame(candidate_rows).to_csv(candidate_metrics_csv, index=False)
    pd.DataFrame(docking_rows).to_csv(docking_metrics_csv, index=False)
    pd.DataFrame(pose_rows).to_csv(pose_metrics_csv, index=False)
    pd.DataFrame(
        [{"method": name, **values} for name, values in MED_REFERENCE.items()]
    ).to_csv(reference_csv, index=False)
    write_json(
        summary_json,
        {
            "candidate_metrics": candidate_rows,
            "docking_metrics": docking_rows,
            "pose_metrics": pose_rows,
            "med_comparison": med_comparison,
            "active_branch_for_med_comparison": active_row["branch"],
        },
    )
    build_report(candidate_rows, docking_rows, pose_rows, med_comparison, active_row["branch"], report_md)

    for path, artifact_type in [
        (candidate_metrics_csv, "v5_3_candidate_benchmark_metrics"),
        (docking_metrics_csv, "v5_3_docking_benchmark_metrics"),
        (pose_metrics_csv, "v5_3_pose_benchmark_metrics"),
        (reference_csv, "reference_med_style_metrics"),
        (summary_json, "v5_3_benchmark_summary"),
        (report_md, "v5_3_benchmark_report"),
    ]:
        register_artifact(base, "V5_3_benchmark", path, artifact_type, owner="Student 1")
    register_run(
        base,
        stage="V5_3_benchmark",
        status="completed",
        input_path="; ".join(str(path) for _branch, path, _scope in candidate_inputs if path.exists()),
        output_path=str(report_md),
        molecules_in=sum(row["num_candidates"] for row in candidate_rows),
        molecules_out=len(candidate_rows),
        notes="MED-style benchmark report; filtered survivor metrics are marked separately from paper-equivalent raw metrics.",
    )

    print(f"Benchmark report: {report_md}")
    print(f"Candidate metrics: {candidate_metrics_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
