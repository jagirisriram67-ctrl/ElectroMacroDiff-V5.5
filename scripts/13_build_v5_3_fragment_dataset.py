from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.macrocycle_fragmentation import build_anchor_atom_dataset, build_fragment_linker_dataset
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5.3 macrocycle fragment-linker and anchor atom datasets.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--fragment-output", default=None)
    parser.add_argument("--anchor-output", default=None)
    parser.add_argument("--max-pairs-per-molecule", type=int, default=20)
    parser.add_argument("--min-ring-size", type=int, default=12)
    parser.add_argument("--max-ring-size", type=int, default=24)
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    curated_csv = Path(args.input_csv).resolve() if args.input_csv else base / "02_curated_data" / "jak2_curated_ligands.csv"
    fragment_csv = (
        Path(args.fragment_output).resolve()
        if args.fragment_output
        else base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv"
    )
    anchor_csv = (
        Path(args.anchor_output).resolve()
        if args.anchor_output
        else base / "03_features" / "v5_3_anchor_atom_training.csv"
    )

    build_fragment_linker_dataset(
        curated_csv,
        fragment_csv,
        max_pairs_per_molecule=args.max_pairs_per_molecule,
        min_ring_size=args.min_ring_size,
        max_ring_size_allowed=args.max_ring_size,
    )
    build_anchor_atom_dataset(fragment_csv, anchor_csv)

    fragments = pd.read_csv(fragment_csv)
    anchors = pd.read_csv(anchor_csv)
    macrocycles = int(fragments["mol_id"].nunique()) if not fragments.empty else 0
    positives = int(anchors["label_anchor"].sum()) if not anchors.empty else 0

    save_progress(
        base / "00_project_registry" / "progress_v5_3_fragment_dataset.json",
        {
            "stage": "V5_3_fragment_dataset",
            "status": "completed",
            "fragment_pairs": len(fragments),
            "macrocycles_used": macrocycles,
            "anchor_atom_rows": len(anchors),
            "positive_anchor_rows": positives,
            "min_ring_size": args.min_ring_size,
            "max_ring_size": args.max_ring_size,
        },
    )
    register_artifact(base, "V5_3_fragment_dataset", fragment_csv, "macrocycle_fragment_linker_pairs", owner="Student 3")
    register_artifact(base, "V5_3_fragment_dataset", anchor_csv, "anchor_atom_training_csv", owner="Student 3")
    register_run(
        base,
        stage="V5_3_fragment_dataset",
        status="completed",
        input_path=str(curated_csv),
        output_path=f"{fragment_csv}; {anchor_csv}",
        molecules_in=macrocycles,
        molecules_out=len(fragments),
        notes=f"anchor_atom_rows={len(anchors)}; positive_anchor_rows={positives}",
    )

    print(f"Fragment-linker pairs: {len(fragments)}")
    print(f"Macrocycles used: {macrocycles}")
    print(f"Anchor atom rows: {len(anchors)}")
    print(f"Positive anchor rows: {positives}")
    print(f"Ring size window: {args.min_ring_size}-{args.max_ring_size}")
    print(f"Fragment CSV: {fragment_csv}")
    print(f"Anchor CSV: {anchor_csv}")


if __name__ == "__main__":
    main()
