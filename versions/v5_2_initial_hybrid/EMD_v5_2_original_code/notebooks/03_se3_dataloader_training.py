# ============================================================
# NOTEBOOK 03: SE(3) DataLoader and Training
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# Purpose: Test DataLoader, train SE(3) flow matching model
# Owner: Student 3
# Day: 3-4
# Go/No-Go: Must complete one train step by end of Day 3
# ============================================================

# ── CELL 1: Install & Mount ───────────────────────────────
# !pip -q install torch e3nn torchdiffeq pyyaml tqdm
# from google.colab import drive
# drive.mount("/content/drive")

# ── CELL 2: Config ─────────────────────────────────────────
import os, sys, time
import numpy as np
import pandas as pd
import torch

BASE = "."
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))

TINY_DEBUG = True
RANDOM_SEED = 42

from emd_pipeline.config_registry import (
    load_config, set_all_seeds, load_progress, save_progress, update_registry
)

config = load_config(os.path.join(BASE, "campaign_config.yaml"))
set_all_seeds(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CELL 3: Load Graph Tensors ────────────────────────────
graphs_path = os.path.join(BASE, "03_features", "ligand_graphs", "se3_graphs.pt")
graphs = torch.load(graphs_path, weights_only=False)
print(f"Loaded {len(graphs)} graph tensors")

# Load split IDs
splits_dir = os.path.join(BASE, "03_features", "splits")
train_ids, val_ids = [], []

train_ids_path = os.path.join(splits_dir, "train_ids.txt")
val_ids_path = os.path.join(splits_dir, "val_ids.txt")

if os.path.exists(train_ids_path):
    with open(train_ids_path) as f:
        train_ids = [line.strip() for line in f if line.strip()]
if os.path.exists(val_ids_path):
    with open(val_ids_path) as f:
        val_ids = [line.strip() for line in f if line.strip()]

print(f"Split IDs: train={len(train_ids)}, val={len(val_ids)}")

# Fallback: if no splits, use all for training
if not train_ids:
    train_ids = [g["mol_id"] for g in graphs[:max(1, int(len(graphs) * 0.8))]]
    val_ids = [g["mol_id"] for g in graphs[int(len(graphs) * 0.8):]]
    print(f"Fallback split: train={len(train_ids)}, val={len(val_ids)}")

# ── CELL 4: Create Datasets ───────────────────────────────
from emd_pipeline.se3_flow_model import (
    MolecularGraphDataset, collate_fn,
    SE3FlowMatchingModel, flow_matching_loss,
    train_se3_model, plot_training_curve
)

max_atoms = config["model"]["max_atoms"]

train_dataset = MolecularGraphDataset(graphs, max_atoms=max_atoms, split_ids=train_ids)
val_dataset = MolecularGraphDataset(graphs, max_atoms=max_atoms, split_ids=val_ids)

# If val is empty, use train as val (for tiny debug)
if len(val_dataset) == 0:
    val_dataset = train_dataset
    print("Warning: using train set as validation (tiny debug)")

print(f"Train dataset: {len(train_dataset)} molecules")
print(f"Val dataset: {len(val_dataset)} molecules")

# ── CELL 5: DataLoader Debug Test ─────────────────────────
print("\n--- DataLoader Debug Test ---")

from torch.utils.data import DataLoader

test_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

for batch in test_loader:
    print(f"Batch keys: {list(batch.keys())}")
    print(f"  atom_features: {batch['atom_features'].shape}")
    print(f"  coords: {batch['coords'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    print(f"  num_atoms: {batch['num_atoms']}")
    print(f"  mol_ids: {batch['mol_id']}")
    break

print("✅ DataLoader test passed")

# ── CELL 6: Model Initialization ──────────────────────────
print("\n--- Model Initialization ---")

# Get atom feature dimension from data
sample = train_dataset[0]
atom_feature_dim = sample["atom_features"].shape[1]
print(f"Atom feature dim: {atom_feature_dim}")

model = SE3FlowMatchingModel(
    atom_feature_dim=atom_feature_dim,
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    num_heads=config["model"]["num_heads"],
    max_atoms=max_atoms,
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

# ── CELL 7: Single Forward/Backward Test ──────────────────
print("\n--- Single Step Debug Test ---")

model = model.to(device)
model.train()

batch = next(iter(test_loader))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

optimizer.zero_grad()
loss = flow_matching_loss(model, batch, device)
print(f"Forward pass loss: {loss.item():.6f}")
assert torch.isfinite(loss), "Loss is not finite!"

loss.backward()
print(f"Backward pass: ✅")

# Check gradients
grad_norms = []
for name, p in model.named_parameters():
    if p.grad is not None:
        grad_norms.append(p.grad.norm().item())
print(f"Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={np.mean(grad_norms):.6f}")

optimizer.step()
print(f"Optimizer step: ✅")

# Save and reload checkpoint
ckpt_dir = os.path.join(BASE, "04_models_checkpoints", "se3_flow")
os.makedirs(ckpt_dir, exist_ok=True)
test_ckpt_path = os.path.join(ckpt_dir, "se3_debug_checkpoint.pt")

ckpt = {
    "epoch": 0,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": loss.item(),
    "config": config["model"],
}
torch.save(ckpt, test_ckpt_path)

# Reload
model2 = SE3FlowMatchingModel(
    atom_feature_dim=atom_feature_dim,
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    num_heads=config["model"]["num_heads"],
    max_atoms=max_atoms,
).to(device)

loaded = torch.load(test_ckpt_path, map_location=device, weights_only=False)
model2.load_state_dict(loaded["model_state_dict"])
print(f"Checkpoint save/reload: ✅")

# Quick inference test
model2.eval()
with torch.no_grad():
    test_loss = flow_matching_loss(model2, batch, device)
print(f"Inference test loss: {test_loss.item():.6f}")

# Save debug test result
debug_result = {
    "test": "se3_dataloader_training",
    "status": "passed",
    "forward_loss": loss.item(),
    "inference_loss": test_loss.item(),
    "model_params": total_params,
    "atom_feature_dim": atom_feature_dim,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

import json
debug_path = os.path.join(ckpt_dir, "se3_dataloader_test_passed.json")
with open(debug_path, "w") as f:
    json.dump(debug_result, f, indent=2)
print(f"\n✅ ALL DEBUG TESTS PASSED — saved to {debug_path}")
print("   Go/No-Go Gate: PASS — proceed to training")

# ── CELL 8: Full Training ────────────────────────────────
print("\n" + "=" * 60)
print("STARTING FULL TRAINING")
print("=" * 60)

# Reset model for clean training
model = SE3FlowMatchingModel(
    atom_feature_dim=atom_feature_dim,
    hidden_dim=config["model"]["hidden_dim"],
    num_layers=config["model"]["num_layers"],
    num_heads=config["model"]["num_heads"],
    max_atoms=max_atoms,
)

train_config = {
    "epochs_target": 20 if TINY_DEBUG else config["model"]["epochs_target"],
    "batch_size": config["model"]["batch_size"],
    "learning_rate": config["model"]["learning_rate"],
    "weight_decay": config["model"]["weight_decay"],
    "gradient_clip": config["model"]["gradient_clip"],
    "checkpoint_every": 5 if TINY_DEBUG else config["model"]["checkpoint_every"],
    "validate_every": 5 if TINY_DEBUG else config["model"]["validate_every"],
    "random_seed": RANDOM_SEED,
}

progress_path = os.path.join(BASE, "00_project_registry", "progress_03_training.json")

trained_model, log_df = train_se3_model(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    config=train_config,
    checkpoint_dir=ckpt_dir,
    device=device,
    progress_path=progress_path,
)

# ── CELL 9: Plot Training Curve ───────────────────────────
curve_path = os.path.join(ckpt_dir, "training_curve.png")
plot_training_curve(log_df, curve_path)

# ── CELL 10: Summary ──────────────────────────────────────
final_train_loss = log_df["train_loss"].iloc[-1]
final_val_loss = log_df["val_loss"].dropna().iloc[-1] if log_df["val_loss"].notna().any() else float("nan")

update_registry(
    os.path.join(BASE, "00_project_registry", "run_registry.csv"),
    {
        "notebook": "03_se3_training",
        "step": "complete",
        "status": "complete",
        "output_path": ckpt_dir,
        "notes": f"{len(log_df)} epochs, final_train={final_train_loss:.6f}, final_val={final_val_loss:.6f}",
    }
)

print("\n" + "=" * 60)
print("NOTEBOOK 03 COMPLETE — SE(3) Model Training")
print("=" * 60)
print(f"  Epochs completed: {len(log_df)}")
print(f"  Final train loss: {final_train_loss:.6f}")
print(f"  Final val loss: {final_val_loss:.6f}")
print(f"  Checkpoints: {ckpt_dir}")
print(f"  Next: Run Notebook 04 (Hybrid Candidate Generation)")
