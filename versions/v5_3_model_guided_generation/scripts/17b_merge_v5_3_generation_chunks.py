from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.baseline_generation import write_candidates_csv
from emd_v5_2_hybrid.generation_metrics import write_generation_metrics
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress


def _read_csvs(paths: list[Path]):
    import pandas as pd

    frames = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_chunk_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent V5.3 generation chunks from multiple Colab members.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--chunk-dir", default=None)
    parser.add_argument("--candidate-pattern", default="generated_v5_3_model_guided_macrocycles_chunk_*.csv")
    parser.add_argument("--attempt-pattern", default="generated_v5_3_model_guided_macrocycles_chunk_*_attempt_log.csv")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--attempt-log-output", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--merged-csv", default=None)
    parser.add_argument("--merged-metrics-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--stage", default="V5_3_model_guided_generation_chunk_merge")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    chunk_dir = (
        Path(args.chunk_dir).resolve()
        if args.chunk_dir
        else base / "05_generated_candidates" / "model_guided_macrocycle" / "chunks"
    )
    candidate_paths = sorted(chunk_dir.glob(args.candidate_pattern))
    attempt_paths = sorted(chunk_dir.glob(args.attempt_pattern))
    if not candidate_paths:
        raise FileNotFoundError(f"No candidate chunk CSV files matched {args.candidate_pattern} in {chunk_dir}")

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles.csv"
    )
    attempt_log_output = (
        Path(args.attempt_log_output).resolve()
        if args.attempt_log_output
        else output_csv.with_name("v5_3_model_guided_generation_attempt_log.csv")
    )
    metrics_csv = (
        Path(args.metrics_csv).resolve()
        if args.metrics_csv
        else base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_generation_metrics.csv"
    )
    merged_csv = (
        Path(args.merged_csv).resolve()
        if args.merged_csv
        else base / "05_generated_candidates" / "merged" / "generated_merged_with_v5_3_model_guided.csv"
    )
    merged_metrics_csv = (
        Path(args.merged_metrics_csv).resolve()
        if args.merged_metrics_csv
        else base / "05_generated_candidates" / "merged" / "generation_comparison_metrics_with_v5_3_model_guided.csv"
    )
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else base / "00_project_registry" / "progress_v5_3_model_guided_generation_chunk_merge.json"
    )

    candidate_frame = _read_csvs(candidate_paths)
    if candidate_frame.empty:
        write_candidates_csv([], output_csv)
    else:
        records = candidate_frame.drop(columns=["source_chunk_file"], errors="ignore").to_dict(orient="records")
        write_candidates_csv(records, output_csv)
    write_generation_metrics(output_csv, metrics_csv)

    attempts = _read_csvs(attempt_paths)
    attempt_log_output.parent.mkdir(parents=True, exist_ok=True)
    attempts.to_csv(attempt_log_output, index=False)

    existing_merged = base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
    guided = pd.read_csv(output_csv)
    if existing_merged.exists():
        existing = pd.read_csv(existing_merged)
        merged = pd.concat([existing, guided], ignore_index=True)
        if "inchikey" in merged.columns:
            merged = merged.drop_duplicates("inchikey")
    else:
        merged = guided
    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(merged_csv, index=False)
    write_generation_metrics(merged_csv, merged_metrics_csv)

    valid_attempts = int((attempts.get("status", pd.Series(dtype=str)).astype(str) == "valid_output").sum()) if not attempts.empty else 0
    final_unique = int(len(pd.read_csv(output_csv)))
    raw_attempts = int(len(attempts))
    summary = {
        "stage": args.stage,
        "status": "completed",
        "chunk_dir": str(chunk_dir),
        "candidate_chunk_files": [str(path) for path in candidate_paths],
        "attempt_chunk_files": [str(path) for path in attempt_paths],
        "raw_candidate_rows_before_dedup": int(len(candidate_frame)),
        "final_unique_candidates": final_unique,
        "raw_attempt_rows": raw_attempts,
        "valid_attempt_rows": valid_attempts,
        "raw_attempt_validity_percent": round(valid_attempts / max(raw_attempts, 1) * 100.0, 4),
        "output_csv": str(output_csv),
        "attempt_log_output": str(attempt_log_output),
        "metrics_csv": str(metrics_csv),
        "merged_csv": str(merged_csv),
    }
    save_progress(summary_json, summary)
    register_artifact(base, args.stage, output_csv, "merged_model_guided_generation_chunks", owner="Student 1")
    register_artifact(base, args.stage, attempt_log_output, "merged_model_guided_attempt_log", owner="Student 1")
    register_artifact(base, args.stage, metrics_csv, "merged_generation_chunk_metrics", owner="Student 1")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=str(chunk_dir),
        output_path=str(output_csv),
        molecules_in=int(len(candidate_frame)),
        molecules_out=final_unique,
        notes=f"chunks={len(candidate_paths)}; attempts={raw_attempts}; validity={summary['raw_attempt_validity_percent']}%",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
