from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.baseline_generation import (
    generate_macrocycle_linker_candidates,
    generate_rdkit_aromatic_substitutions,
    generate_selfies_candidates,
    write_candidates_csv,
)
from emd_v5_2_hybrid.generation_metrics import write_generation_metrics
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RDKit and SELFIES baseline candidates.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--selfies-per-seed", type=int, default=10)
    parser.add_argument("--rdkit-per-seed", type=int, default=5)
    parser.add_argument("--macrocycle-per-seed", type=int, default=8)
    parser.add_argument("--no-macrocycle-linker", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    curated_csv = base / "02_curated_data" / "jak2_curated_ligands.csv"
    curated = pd.read_csv(curated_csv).sort_values("p_activity", ascending=False).head(args.seed_count)
    seed_rows = curated.to_dict(orient="records")
    training_inchikeys = set(pd.read_csv(curated_csv)["inchikey"].dropna().astype(str))

    try:
        selfies_records = generate_selfies_candidates(
            seed_rows,
            n_per_seed=args.selfies_per_seed,
            training_inchikeys=training_inchikeys,
        )
    except RuntimeError as exc:
        print(f"SELFIES generation skipped: {exc}")
        selfies_records = []
    rdkit_records = generate_rdkit_aromatic_substitutions(
        seed_rows,
        max_products_per_seed=args.rdkit_per_seed,
        training_inchikeys=training_inchikeys,
    )
    if args.no_macrocycle_linker:
        macrocycle_records = []
    else:
        macrocycle_records = generate_macrocycle_linker_candidates(
            seed_rows,
            max_products_per_seed=args.macrocycle_per_seed,
            training_inchikeys=training_inchikeys,
        )
    selfies_path = base / "05_generated_candidates" / "selfies" / "generated_selfies_filtered.csv"
    rdkit_path = base / "05_generated_candidates" / "rdkit" / "generated_rdkit_filtered.csv"
    macrocycle_path = (
        base / "05_generated_candidates" / "macrocycle_linker" / "generated_macrocycle_linker_filtered.csv"
    )
    merged_path = base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
    metrics_path = base / "05_generated_candidates" / "merged" / "generation_comparison_metrics.csv"
    write_candidates_csv(selfies_records, selfies_path)
    write_candidates_csv(rdkit_records, rdkit_path)
    write_candidates_csv(macrocycle_records, macrocycle_path)
    merged = selfies_records + rdkit_records + macrocycle_records
    write_candidates_csv(merged, merged_path)
    write_generation_metrics(merged_path, metrics_path)
    merged_count = len(pd.read_csv(merged_path))

    save_progress(
        base / "00_project_registry" / "progress_m4_baseline_generation.json",
        {"last_completed_index": len(merged) - 1, "stage": "M4_baseline_generation"},
    )
    register_artifact(base, "M4_generation", selfies_path, "selfies_candidates", owner="Student 4")
    register_artifact(base, "M4_generation", rdkit_path, "rdkit_candidates", owner="Student 4")
    register_artifact(base, "M4_generation", macrocycle_path, "macrocycle_linker_candidates", owner="Student 4")
    register_artifact(base, "M4_generation", merged_path, "merged_candidate_csv", owner="Student 1")
    register_artifact(base, "M4_generation", metrics_path, "generation_metrics", owner="Student 1")
    register_run(
        base,
        stage="M4_generation",
        status="completed",
        input_path=str(curated_csv),
        output_path=str(merged_path),
        molecules_in=len(seed_rows),
        molecules_out=merged_count,
        notes=(
            f"SELFIES {len(selfies_records)}, RDKit {len(rdkit_records)}, "
            f"macrocycle_linker {len(macrocycle_records)}"
        ),
    )
    print(f"SELFIES candidates: {len(selfies_records)}")
    print(f"RDKit candidates: {len(rdkit_records)}")
    print(f"Macrocycle linker candidates: {len(macrocycle_records)}")
    print(f"Merged filtered candidates: {merged_count}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
