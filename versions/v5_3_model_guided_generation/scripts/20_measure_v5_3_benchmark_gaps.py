from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.macrocycle_fragmentation import extract_linker_smiles_from_smiles
from emd_v5_2_hybrid.registry import register_artifact, register_run


def load_training_linkers(paths: list[Path]) -> set[str]:
    import pandas as pd

    linkers: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "linker_smiles" not in frame.columns:
            continue
        linkers.update(frame["linker_smiles"].dropna().astype(str))
    return linkers


def measure_linker_novelty(generated_csv: Path, training_linkers: set[str], output_csv: Path) -> dict:
    import pandas as pd

    generated = pd.read_csv(generated_csv)
    rows: list[dict] = []
    for row in generated.to_dict(orient="records"):
        linker_smiles = extract_linker_smiles_from_smiles(
            str(row.get("canonical_smiles", row.get("smiles", ""))),
            mol_id=str(row.get("candidate_id", "")),
            max_pairs_per_molecule=50,
            min_ring_size=12,
            max_ring_size_allowed=24,
        )
        known = [smiles for smiles in linker_smiles if smiles in training_linkers]
        novel = [smiles for smiles in linker_smiles if smiles not in training_linkers]
        rows.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "extracted_linker_count": len(linker_smiles),
                "known_linker_count": len(known),
                "novel_linker_count": len(novel),
                "linker_novel": bool(linker_smiles and len(known) == 0),
                "linker_smiles": ";".join(linker_smiles),
                "known_linker_smiles": ";".join(known),
                "novel_linker_smiles": ";".join(novel),
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    measured = [row for row in rows if row["extracted_linker_count"] > 0]
    novel_count = sum(1 for row in measured if row["linker_novel"])
    return {
        "generated_candidates": int(len(generated)),
        "training_linker_count": int(len(training_linkers)),
        "linker_novelty_rows": int(len(rows)),
        "linker_novelty_measured_rows": int(len(measured)),
        "linker_novel_rows": int(novel_count),
        "linker_novelty_percent": round(float(novel_count / max(len(measured), 1) * 100.0), 4),
        "linker_novelty_csv": str(output_csv),
    }


def summarize_attempt_log(attempt_log: Path) -> dict:
    import pandas as pd

    if not attempt_log.exists():
        return {
            "attempt_log_exists": False,
            "raw_attempt_rows": 0,
            "valid_output_rows": 0,
            "raw_attempt_validity_percent": None,
            "status_counts": {},
        }
    frame = pd.read_csv(attempt_log)
    statuses = Counter(frame["status"].astype(str)) if "status" in frame.columns else Counter()
    valid = int(statuses.get("valid_output", 0))
    total = int(len(frame))
    return {
        "attempt_log_exists": True,
        "attempt_log_csv": str(attempt_log),
        "raw_attempt_rows": total,
        "valid_output_rows": valid,
        "raw_attempt_validity_percent": round(float(valid / max(total, 1) * 100.0), 4),
        "status_counts": dict(sorted(statuses.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure V5.3 benchmark gaps: raw attempts and linker novelty.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--generated-csv", default=None)
    parser.add_argument("--attempt-log", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--training-fragment-csv", action="append", default=None)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    generated_csv = (
        Path(args.generated_csv).resolve()
        if args.generated_csv
        else base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles.csv"
    )
    attempt_log = (
        Path(args.attempt_log).resolve()
        if args.attempt_log
        else base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_generation_attempt_log.csv"
    )
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_linker_novelty.csv"
    )
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_benchmark_gap_summary.json"
    )
    training_fragment_paths = (
        [Path(path).resolve() for path in args.training_fragment_csv]
        if args.training_fragment_csv
        else [
            base / "02_curated_data" / "v5_3_pretrain_fragment_linker_pairs.csv",
            base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv",
        ]
    )

    training_linkers = load_training_linkers(training_fragment_paths)
    novelty_summary = measure_linker_novelty(generated_csv, training_linkers, output_csv)
    attempt_summary = summarize_attempt_log(attempt_log)
    summary = {
        "status": "completed",
        "generated_csv": str(generated_csv),
        "training_fragment_paths": [str(path) for path in training_fragment_paths if path.exists()],
        **novelty_summary,
        **attempt_summary,
        "benchmark_note": (
            "Raw-attempt validity is paper-equivalent only after generation logs every attempted molecule. "
            "Linker novelty is computed from RDKit two-bond macrocycle fragmentation and exact linker SMILES matching."
        ),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    register_artifact(base, "V5_3_benchmark_gap_measurement", output_csv, "v5_3_linker_novelty_csv", owner="Student 1")
    register_artifact(base, "V5_3_benchmark_gap_measurement", summary_json, "v5_3_benchmark_gap_summary", owner="Student 1")
    register_run(
        base,
        stage="V5_3_benchmark_gap_measurement",
        status="completed",
        input_path=f"{generated_csv}; {attempt_log}",
        output_path=f"{output_csv}; {summary_json}",
        molecules_in=novelty_summary["generated_candidates"],
        molecules_out=novelty_summary["linker_novelty_measured_rows"],
        notes=f"linker_novelty_percent={novelty_summary['linker_novelty_percent']}; raw_attempt_validity_percent={attempt_summary['raw_attempt_validity_percent']}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
