from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress
from emd_v5_2_hybrid.train_se3 import load_graphs, train_epochs


def split_graphs(graphs: list[dict], graph_index_csv: Path, seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    import pandas as pd

    graph_by_id = {str(graph.get("mol_id", "")): graph for graph in graphs}
    if graph_index_csv.exists():
        index = pd.read_csv(graph_index_csv)
        split_values = index["split"].astype(str).str.lower()
        train_ids = index.loc[split_values.isin(["train", "pretrain_train"]), "mol_id"].astype(str).tolist()
        val_ids = index.loc[split_values.isin(["val", "pretrain_val"]), "mol_id"].astype(str).tolist()
        test_ids = index.loc[split_values.isin(["test", "pretrain_test"]), "mol_id"].astype(str).tolist()
        train_graphs = [graph_by_id[mol_id] for mol_id in train_ids if mol_id in graph_by_id]
        val_graphs = [graph_by_id[mol_id] for mol_id in val_ids if mol_id in graph_by_id]
        test_graphs = [graph_by_id[mol_id] for mol_id in test_ids if mol_id in graph_by_id]
        if train_graphs and val_graphs:
            return train_graphs, val_graphs, test_graphs

    shuffled = list(graphs)
    random.Random(seed).shuffle(shuffled)
    train_cut = int(len(shuffled) * 0.7)
    val_cut = int(len(shuffled) * 0.85)
    return shuffled[:train_cut], shuffled[train_cut:val_cut], shuffled[val_cut:]


def limit_graphs(graphs: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(graphs) <= limit:
        return graphs
    sampled = list(graphs)
    random.Random(seed).shuffle(sampled)
    return sampled[:limit]


def write_training_curve(log_csv: Path, output_png: Path) -> str:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        return f"skipped: {exc.__class__.__name__}"

    if not log_csv.exists() or log_csv.stat().st_size == 0:
        return "skipped: missing training log"
    log = pd.read_csv(log_csv)
    if log.empty:
        return "skipped: empty training log"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(log["epoch"], log["train_loss"], label="train")
    plt.plot(log["epoch"], log["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Flow matching loss")
    plt.title("SE(3) Flow Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main SE(3) flow training for Colab/GPU.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--graph-index", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from se3_latest_checkpoint.pt in the checkpoint directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--atom-feature-dim", type=int, default=64)
    parser.add_argument("--max-train-graphs", type=int, default=0)
    parser.add_argument("--max-val-graphs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or leave empty for auto.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    graph_path = Path(args.graph_path).resolve() if args.graph_path else base / "03_features" / "ligand_graphs" / "se3_graphs.pt"
    graph_index = Path(args.graph_index).resolve() if args.graph_index else base / "03_features" / "ligand_graphs" / "se3_graph_index.csv"
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else base / "04_models_checkpoints" / "se3_flow"
    init_checkpoint = Path(args.init_checkpoint).resolve() if args.init_checkpoint else None
    if args.resume and init_checkpoint is None:
        latest = checkpoint_dir / "se3_latest_checkpoint.pt"
        best = checkpoint_dir / "se3_best_checkpoint.pt"
        if latest.exists():
            init_checkpoint = latest
        elif best.exists():
            init_checkpoint = best
        else:
            raise FileNotFoundError(f"--resume requested, but no SE(3) checkpoint exists in {checkpoint_dir}")
    summary_path = checkpoint_dir / "training_summary.json"
    log_path = checkpoint_dir / "training_log.csv"
    curve_path = checkpoint_dir / "training_curve.png"

    graphs = load_graphs(graph_path)
    train_graphs, val_graphs, test_graphs = split_graphs(graphs, graph_index, seed=args.seed)
    train_graphs = limit_graphs(train_graphs, args.max_train_graphs, args.seed)
    val_graphs = limit_graphs(val_graphs, args.max_val_graphs, args.seed)

    best_path = train_epochs(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        checkpoint_dir=checkpoint_dir,
        atom_feature_dim=args.atom_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        init_checkpoint=init_checkpoint,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    curve_status = write_training_curve(log_path, curve_path)

    import pandas as pd

    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    best_val_loss = float(log["val_loss"].min()) if not log.empty and "val_loss" in log else None
    final_train_loss = float(log.iloc[-1]["train_loss"]) if not log.empty else None
    final_val_loss = float(log.iloc[-1]["val_loss"]) if not log.empty else None
    summary = {
        "status": "completed",
        "device": args.device or "auto",
        "num_graphs_total": len(graphs),
        "num_train_graphs": len(train_graphs),
        "num_val_graphs": len(val_graphs),
        "num_test_graphs_held_out": len(test_graphs),
        "epochs": args.epochs,
        "epochs_completed": int(log.iloc[-1]["epoch"]) if not log.empty and "epoch" in log else 0,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "graph_path": str(graph_path),
        "graph_index": str(graph_index),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
        "resume": bool(args.resume),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(checkpoint_dir / "se3_latest_checkpoint.pt"),
        "training_log": str(log_path),
        "training_curve": str(curve_path),
        "training_curve_status": curve_status,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_progress(
        base / "00_project_registry" / "progress_m3_se3_training.json",
        {
            "stage": "M3_se3_main_training",
            "status": "completed",
            "epochs": args.epochs,
            "best_val_loss": best_val_loss,
        },
    )
    register_artifact(base, "M3_se3_main_training", best_path, "se3_best_checkpoint", owner="Student 3")
    register_artifact(base, "M3_se3_main_training", checkpoint_dir / "se3_latest_checkpoint.pt", "se3_latest_checkpoint", owner="Student 3")
    register_artifact(base, "M3_se3_main_training", log_path, "se3_training_log", owner="Student 3")
    register_artifact(base, "M3_se3_main_training", summary_path, "se3_training_summary", owner="Student 3")
    if curve_path.exists():
        register_artifact(base, "M3_se3_main_training", curve_path, "se3_training_curve", owner="Student 3")
    register_run(
        base,
        stage="M3_se3_main_training",
        status="completed",
        input_path=str(graph_path),
        output_path=str(best_path),
        molecules_in=len(train_graphs) + len(val_graphs),
        molecules_out=len(train_graphs) + len(val_graphs),
        notes=f"epochs={args.epochs}; best_val_loss={best_val_loss}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
