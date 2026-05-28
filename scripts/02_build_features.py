from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.features import build_ligand_features_csv, write_split_files
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress
from emd_v5_2_hybrid.se3_dataset import build_graphs_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ligand descriptors, conformers, and SE(3) graph tensors.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--feature-output", default=None)
    parser.add_argument("--graph-output", default=None)
    parser.add_argument("--graph-index-output", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--tiny-debug", action="store_true")
    parser.add_argument("--atom-feature-dim", type=int, default=64)
    parser.add_argument("--skip-split-files", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    curated_csv = Path(args.input_csv).resolve() if args.input_csv else base / "02_curated_data" / "jak2_curated_ligands.csv"
    feature_csv = Path(args.feature_output).resolve() if args.feature_output else base / "03_features" / "ligand_features.csv"
    graph_pt = Path(args.graph_output).resolve() if args.graph_output else base / "03_features" / "ligand_graphs" / "se3_graphs.pt"
    graph_index = (
        Path(args.graph_index_output).resolve()
        if args.graph_index_output
        else base / "03_features" / "ligand_graphs" / "se3_graph_index.csv"
    )
    split_dir = Path(args.split_dir).resolve() if args.split_dir else base / "03_features" / "splits"
    limit = 25 if args.tiny_debug else None

    build_ligand_features_csv(curated_csv, feature_csv)
    build_graphs_from_csv(
        curated_csv,
        graph_pt,
        graph_index,
        atom_feature_dim=args.atom_feature_dim,
        limit=limit,
    )
    split_paths = {} if args.skip_split_files else write_split_files(curated_csv, split_dir)

    save_progress(
        base / "00_project_registry" / "progress_m2_features.json",
        {"last_completed_index": limit or -1, "stage": "M2_features"},
    )
    register_artifact(base, "M2_features", feature_csv, "ligand_feature_csv", owner="Student 4")
    register_artifact(base, "M2_features", graph_pt, "se3_graph_tensor", owner="Student 3")
    for split, path in split_paths.items():
        register_artifact(base, "M2_features", path, f"{split}_split_ids", owner="Student 3")
    register_run(
        base,
        stage="M2_features",
        status="completed",
        input_path=str(curated_csv),
        output_path=str(feature_csv),
        notes="Tiny Debug mode" if args.tiny_debug else "Main feature build",
    )
    print(f"Wrote features: {feature_csv}")
    print(f"Wrote graphs: {graph_pt}")


if __name__ == "__main__":
    main()
