"""Metrics for hybrid candidate generation."""

from __future__ import annotations

from pathlib import Path


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas to compute generation metrics") from exc
    return pd


def summarize_candidates(frame) -> list[dict]:
    if frame.empty:
        return []
    rows = []
    for source, group in frame.groupby("source_generator", dropna=False):
        rows.append(
            {
                "source_generator": source,
                "num_candidates": int(len(group)),
                "num_unique_inchikeys": int(group["inchikey"].nunique()) if "inchikey" in group else 0,
                "novel_fraction": round(float(group["novel_flag"].mean()), 6) if "novel_flag" in group else "",
                "basic_filter_pass_fraction": round(float(group["passes_basic_filters"].mean()), 6)
                if "passes_basic_filters" in group
                else "",
                "macrocycle_fraction": round(float(group["has_macrocycle_12_20"].mean()), 6)
                if "has_macrocycle_12_20" in group
                else "",
                "constrained_ring_fraction": round(float(group["has_constrained_ring_8_11"].mean()), 6)
                if "has_constrained_ring_8_11" in group
                else "",
                "median_mw": round(float(group["mw"].median()), 4) if "mw" in group else "",
                "median_logp": round(float(group["logp"].median()), 4) if "logp" in group else "",
                "median_qed": round(float(group["qed"].median()), 4) if "qed" in group else "",
                "median_sa_score": round(float(group["sa_score"].median()), 4) if "sa_score" in group else "",
            }
        )
    rows.append(
        {
            "source_generator": "ALL",
            "num_candidates": int(len(frame)),
            "num_unique_inchikeys": int(frame["inchikey"].nunique()) if "inchikey" in frame else 0,
            "novel_fraction": round(float(frame["novel_flag"].mean()), 6) if "novel_flag" in frame else "",
            "basic_filter_pass_fraction": round(float(frame["passes_basic_filters"].mean()), 6)
            if "passes_basic_filters" in frame
            else "",
            "macrocycle_fraction": round(float(frame["has_macrocycle_12_20"].mean()), 6)
            if "has_macrocycle_12_20" in frame
            else "",
            "constrained_ring_fraction": round(float(frame["has_constrained_ring_8_11"].mean()), 6)
            if "has_constrained_ring_8_11" in frame
            else "",
            "median_mw": round(float(frame["mw"].median()), 4) if "mw" in frame else "",
            "median_logp": round(float(frame["logp"].median()), 4) if "logp" in frame else "",
            "median_qed": round(float(frame["qed"].median()), 4) if "qed" in frame else "",
            "median_sa_score": round(float(frame["sa_score"].median()), 4) if "sa_score" in frame else "",
        }
    )
    return rows


def write_generation_metrics(candidate_csv: str | Path, output_csv: str | Path) -> Path:
    pd = _pandas()
    frame = pd.read_csv(candidate_csv)
    rows = summarize_candidates(frame)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
