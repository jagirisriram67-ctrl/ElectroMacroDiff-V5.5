"""Continue SE(3) training for the V5.5 staircase.

This remains auxiliary geometry evidence. The generated molecules are not
claimed to come from the SE(3) model unless a future generation script actually
uses SE(3) coordinates.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue V5.5 SE(3) flow matching training.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default="v5_5_se3_continuation")
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.checkpoint_every = 1
        args.max_graphs = args.max_graphs or 32

    import pandas as pd
    import torch

    from emd_v5_2_hybrid.device_helper import device_summary, resolve_device, xla_step
    from emd_v5_2_hybrid.se3_flow import SE3FlowConfig, SE3FlowMatching, flow_matching_loss
    from emd_v5_2_hybrid.train_se3 import evaluate_loss, load_graphs, make_dataloader, move_batch, save_checkpoint

    base = Path(args.base).resolve()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(json.dumps(device_summary(device), indent=2))

    output_dir_name = "v5_5_se3_continuation_smoke" if args.smoke_test else "v5_5_se3_continuation"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else base / "04_models_checkpoints" / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    init_checkpoint = base / "04_models_checkpoints" / "se3_flow" / "se3_best_checkpoint.pt"
    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Missing SE(3) checkpoint: {init_checkpoint}")
    checkpoint = torch.load(init_checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    atom_feature_dim = int(config.get("atom_feature_dim", 64))
    hidden_dim = int(config.get("hidden_dim", 128))
    num_layers = int(config.get("num_layers", 4))
    init_epoch = int(checkpoint.get("epoch", 0))
    init_val_loss = checkpoint.get("val_loss", float("inf"))

    graph_path = base / "03_features" / "ligand_graphs" / "se3_graphs.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing SE(3) graph tensor: {graph_path}")
    graphs = load_graphs(graph_path)
    if args.max_graphs:
        graphs = graphs[: args.max_graphs]
    if not graphs:
        raise RuntimeError("No SE(3) graphs available for continuation training.")
    train_graphs, val_graphs, test_graphs = split_graphs_by_mol_id(graphs, seed=42)
    if not val_graphs:
        val_graphs = train_graphs[: min(8, len(train_graphs))]
    if not test_graphs:
        test_graphs = val_graphs
    print(f"Graphs: {len(graphs)} | train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")

    train_loader = make_dataloader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = make_dataloader(test_graphs, batch_size=args.batch_size, shuffle=False)

    model = SE3FlowMatching(SE3FlowConfig(atom_feature_dim=atom_feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_val_loss = float(init_val_loss) if init_val_loss is not None else float("inf")
    best_epoch = init_epoch
    no_improve = 0
    global_step = int(checkpoint.get("global_step", 0))
    log_rows = []
    start = time.time()

    for local_epoch in range(1, args.epochs + 1):
        epoch = init_epoch + local_epoch
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            xla_step()
            global_step += 1
            losses.append(float(loss.item()))

        scheduler.step()
        train_loss = sum(losses) / max(len(losses), 1)
        val_loss = evaluate_loss(model, val_loader, device)
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "global_step": global_step})
        pd.DataFrame(log_rows).to_csv(output_dir / "se3_v5_5_training_log.csv", index=False)

        if local_epoch == 1 or local_epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                output_dir / "se3_v5_5_best_checkpoint.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                train_loss=train_loss,
                val_loss=val_loss,
                config={"atom_feature_dim": atom_feature_dim, "hidden_dim": hidden_dim, "num_layers": num_layers},
            )
        else:
            no_improve += 1

        save_checkpoint(
            output_dir / "se3_v5_5_latest_checkpoint.pt",
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            val_loss=val_loss,
            config={"atom_feature_dim": atom_feature_dim, "hidden_dim": hidden_dim, "num_layers": num_layers},
        )
        if args.checkpoint_every and local_epoch % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"se3_v5_5_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                train_loss=train_loss,
                val_loss=val_loss,
                config={"atom_feature_dim": atom_feature_dim, "hidden_dim": hidden_dim, "num_layers": num_layers},
            )
        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}; best val={best_val_loss:.4f}")
            break

    test_loss = evaluate_loss(model, test_loader, device)
    summary = {
        "status": "completed",
        "run_id": args.run_id,
        "smoke_test": bool(args.smoke_test),
        "model_type": "se3_flow_continuation",
        "device": device,
        "init_epoch": init_epoch,
        "init_val_loss": None if init_val_loss is None else float(init_val_loss),
        "final_epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 6),
        "test_loss": round(float(test_loss), 6),
        "total_time_seconds": round(time.time() - start, 1),
        **device_summary(device),
    }
    (output_dir / "se3_v5_5_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def split_graphs_by_mol_id(graphs, seed: int = 42):
    mol_ids = sorted({str(graph.get("mol_id", "")) for graph in graphs})
    random.Random(seed).shuffle(mol_ids)
    train_cut = int(len(mol_ids) * 0.70)
    val_cut = int(len(mol_ids) * 0.85)
    train_ids = set(mol_ids[:train_cut])
    val_ids = set(mol_ids[train_cut:val_cut])
    test_ids = set(mol_ids[val_cut:])
    return (
        [graph for graph in graphs if str(graph.get("mol_id", "")) in train_ids],
        [graph for graph in graphs if str(graph.get("mol_id", "")) in val_ids],
        [graph for graph in graphs if str(graph.get("mol_id", "")) in test_ids],
    )


if __name__ == "__main__":
    main()
