from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress
from emd_v5_2_hybrid.train_se3 import debug_train_step, load_graphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 3 SE(3) debug train-step gate.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--atom-feature-dim", type=int, default=64)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    graph_path = base / "03_features" / "ligand_graphs" / "se3_graphs.pt"
    checkpoint_path = base / "04_models_checkpoints" / "se3_flow" / "se3_latest_checkpoint.pt"
    result_path = base / "04_models_checkpoints" / "se3_flow" / "se3_dataloader_test_passed.json"

    graphs = load_graphs(graph_path)
    result = debug_train_step(
        graphs,
        checkpoint_path,
        atom_feature_dim=args.atom_feature_dim,
        hidden_dim=args.hidden_dim,
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_progress(
        base / "00_project_registry" / "progress_m3_se3_debug.json",
        {"last_completed_index": 1, "stage": "M3_se3_debug", "status": "passed"},
    )
    register_artifact(base, "M3_se3_debug", checkpoint_path, "se3_checkpoint", owner="Student 3")
    register_artifact(base, "M3_se3_debug", result_path, "se3_debug_result", owner="Student 3")
    register_run(
        base,
        stage="M3_se3_debug",
        status="completed",
        input_path=str(graph_path),
        output_path=str(checkpoint_path),
        molecules_in=len(graphs),
        molecules_out=len(graphs),
        notes=f"Debug loss {result['loss']:.6f}",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
