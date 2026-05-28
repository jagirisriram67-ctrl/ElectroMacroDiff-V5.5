"""Benchmark helpers for MED-style EMD comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MED_REFERENCE = {
    "MED": {
        "validity_percent": 93.82,
        "uniqueness_percent": 99.94,
        "macrocyclization_percent": 99.92,
        "linker_novelty_percent": 82.81,
        "source": "Macro-Equi-Diff PDF Table 1; local extracted summary in 09_reports/med_comparison_assets.",
    },
    "Macformer": {
        "validity_percent": 72.91,
        "uniqueness_percent": 47.74,
        "macrocyclization_percent": 96.39,
        "linker_novelty_percent": 44.24,
        "source": "Macro-Equi-Diff PDF Table 1; local extracted summary in 09_reports/med_comparison_assets.",
    },
    "MacLS": {
        "validity_percent": 89.67,
        "uniqueness_percent": 95.04,
        "macrocyclization_percent": 100.0,
        "linker_novelty_percent": 0.0,
        "source": "Macro-Equi-Diff PDF Table 1; local extracted summary in 09_reports/med_comparison_assets.",
    },
}


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas to build benchmark reports") from exc
    return pd


def _is_true_series(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _numeric_percent(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value) * 100.0, 4)


def _safe_median(frame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    pd = _pandas()
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return round(float(values.median()), 4)


def summarize_candidate_frame(frame, branch_name: str, benchmark_scope: str) -> dict[str, Any]:
    """Summarize a generated candidate table.

    The current project stores filtered survivor CSVs, so validity and basic
    filter rates describe the saved candidate population unless a raw attempt
    manifest is supplied in a future version.
    """

    total = int(len(frame))
    unique = int(frame["inchikey"].nunique()) if total and "inchikey" in frame.columns else total
    validity_fraction = None
    if total and "valid_rdkit" in frame.columns:
        validity_fraction = float(_is_true_series(frame["valid_rdkit"]).mean())
    elif total:
        validity_fraction = 1.0

    macrocycle_fraction = None
    if total and "has_macrocycle_12_20" in frame.columns:
        macrocycle_fraction = float(_is_true_series(frame["has_macrocycle_12_20"]).mean())

    basic_filter_fraction = None
    if total and "passes_basic_filters" in frame.columns:
        basic_filter_fraction = float(_is_true_series(frame["passes_basic_filters"]).mean())

    novelty_fraction = None
    if total and "novel_flag" in frame.columns:
        novelty_fraction = float(_is_true_series(frame["novel_flag"]).mean())

    return {
        "branch": branch_name,
        "benchmark_scope": benchmark_scope,
        "num_candidates": total,
        "num_unique_inchikeys": unique,
        "validity_percent": _numeric_percent(validity_fraction),
        "uniqueness_percent": _numeric_percent(unique / total) if total else None,
        "macrocyclization_percent": _numeric_percent(macrocycle_fraction),
        "novel_molecule_percent": _numeric_percent(novelty_fraction),
        "basic_filter_pass_percent": _numeric_percent(basic_filter_fraction),
        "linker_novelty_percent": None,
        "median_mw": _safe_median(frame, "mw"),
        "median_logp": _safe_median(frame, "logp"),
        "median_qed": _safe_median(frame, "qed"),
        "median_sa_score": _safe_median(frame, "sa_score"),
        "interpretation_note": (
            "Filtered survivor metrics; not directly equivalent to MED raw generation metrics."
        ),
    }


def summarize_candidate_csv(path: str | Path, branch_name: str, benchmark_scope: str) -> dict[str, Any]:
    pd = _pandas()
    return summarize_candidate_frame(pd.read_csv(path), branch_name, benchmark_scope)


def summarize_docking_csv(path: str | Path, branch_name: str) -> dict[str, Any]:
    pd = _pandas()
    docking_path = Path(path)
    if not docking_path.exists():
        return {
            "branch": branch_name,
            "docking_rows": 0,
            "scored_rows": 0,
            "coverage_percent": 0.0,
            "best_score": None,
            "median_score": None,
            "mean_score": None,
        }
    frame = pd.read_csv(docking_path)
    scores = pd.to_numeric(frame.get("best_score"), errors="coerce")
    scored = scores.dropna()
    return {
        "branch": branch_name,
        "docking_rows": int(len(frame)),
        "scored_rows": int(len(scored)),
        "coverage_percent": round(float(len(scored) / len(frame) * 100.0), 4) if len(frame) else 0.0,
        "best_score": None if scored.empty else round(float(scored.min()), 4),
        "median_score": None if scored.empty else round(float(scored.median()), 4),
        "mean_score": None if scored.empty else round(float(scored.mean()), 4),
    }


def summarize_pose_csv(path: str | Path, branch_name: str) -> dict[str, Any]:
    pd = _pandas()
    pose_path = Path(path)
    if not pose_path.exists():
        return {"branch": branch_name, "pose_rows": 0, "pose_pass_rows": 0, "pose_pass_percent": None}
    frame = pd.read_csv(pose_path)
    if frame.empty or "pose_decision" not in frame.columns:
        return {"branch": branch_name, "pose_rows": int(len(frame)), "pose_pass_rows": 0, "pose_pass_percent": None}
    passes = frame["pose_decision"].astype(str).str.lower().eq("pass")
    return {
        "branch": branch_name,
        "pose_rows": int(len(frame)),
        "pose_pass_rows": int(passes.sum()),
        "pose_pass_percent": round(float(passes.mean() * 100.0), 4),
    }


def compare_to_reference(row: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Compare one EMD branch row to a MED-style reference row."""

    metrics = [
        "validity_percent",
        "uniqueness_percent",
        "macrocyclization_percent",
        "linker_novelty_percent",
    ]
    comparisons: dict[str, Any] = {}
    comparable = True
    beats_all_comparable = True
    for metric in metrics:
        current = row.get(metric)
        target = reference.get(metric)
        if metric == "validity_percent" and row.get("benchmark_scope") != "raw_attempt_logged":
            comparable = False
            comparisons[metric] = "not_comparable"
            beats_all_comparable = False
            continue
        if current is None or target is None:
            comparable = False
            comparisons[metric] = "not_comparable"
            beats_all_comparable = False
        else:
            beats = float(current) > float(target)
            comparisons[metric] = "exceeds" if beats else "does_not_exceed"
            beats_all_comparable = beats_all_comparable and beats
    return {
        "branch": row["branch"],
        "reference": "MED",
        "beats_all_comparable_metrics": bool(beats_all_comparable),
        "paper_equivalent_claim_allowed": False if not comparable else bool(beats_all_comparable),
        "comparisons": comparisons,
        "claim_note": (
            "Raw-attempt validity is unavailable or the current candidate row is still a filtered-survivor table; "
            "do not claim MED is beaten on the original paper benchmark yet."
            if not comparable
            else "All MED-style metrics are comparable for this row."
        ),
    }


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
