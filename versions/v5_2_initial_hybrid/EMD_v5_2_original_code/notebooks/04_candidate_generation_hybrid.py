# ============================================================
# NOTEBOOK 04: Hybrid Candidate Generation
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# Purpose: Generate from SE(3) model + SELFIES/RDKit baselines, merge
# Owner: Student 3 + Student 4
# Day: 4-5
# Success gate: 100+ filtered candidates total
# ============================================================

# ── CELL 1: Install & Mount ───────────────────────────────
# !pip -q install rdkit-pypi selfies torch pyyaml tqdm
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

from emd_pipeline.config_registry import load_config, set_all_seeds, update_registry
config = load_config(os.path.join(BASE, "campaign_config.yaml"))
set_all_seeds(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CELL 3: Load Training Data ────────────────────────────
curated_df = pd.read_csv(os.path.join(BASE, "02_curated_data", "jak2_curated_ligands.csv"))
training_smiles = set(curated_df["canonical_smiles"].dropna().tolist())
print(f"Training set: {len(training_smiles)} molecules")

if TINY_DEBUG:
    curated_df = curated_df.head(config["data"]["tiny_debug_size"])

# Select potent seeds for baseline generation
seeds_df = curated_df.copy()
if "p_activity" in seeds_df.columns:
    seeds_df = seeds_df.sort_values("p_activity", ascending=False).head(
        50 if not TINY_DEBUG else 10
    )
print(f"Seed molecules for baseline generation: {len(seeds_df)}")

# ── CELL 4: SE(3) Model Generation ────────────────────────
print("\n--- SE(3) Model Generation ---")

from emd_pipeline.se3_flow_model import (
    SE3FlowMatchingModel, generate_se3_candidates, coords_to_smiles
)

ckpt_dir = os.path.join(BASE, "04_models_checkpoints", "se3_flow")
best_ckpt = os.path.join(ckpt_dir, "se3_best_checkpoint.pt")
latest_ckpt = os.path.join(ckpt_dir, "se3_latest_checkpoint.pt")

se3_raw_df = pd.DataFrame()

ckpt_path = best_ckpt if os.path.exists(best_ckpt) else latest_ckpt

if os.path.exists(ckpt_path):
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load graphs for seed templates
    graphs = torch.load(
        os.path.join(BASE, "03_features", "ligand_graphs", "se3_graphs.pt"),
        weights_only=False
    )
    
    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_config = ckpt.get("config", config["model"])
    
    atom_feature_dim = graphs[0]["atom_features"].shape[1]
    model = SE3FlowMatchingModel(
        atom_feature_dim=atom_feature_dim,
        hidden_dim=model_config.get("hidden_dim", 128),
        num_layers=model_config.get("num_layers", 4),
        num_heads=model_config.get("num_heads", 4),
        max_atoms=model_config.get("max_atoms", 80),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Model loaded from epoch {ckpt.get('epoch', 'unknown')}")
    
    # Generate 3D coordinate candidates
    n_seeds = min(len(graphs), 20 if TINY_DEBUG else 100)
    generated = generate_se3_candidates(
        model, graphs[:n_seeds],
        num_steps=20 if TINY_DEBUG else 50,
        device=device
    )
    
    # Reconstruct SMILES from generated coordinates
    se3_smiles = coords_to_smiles(generated, graphs)
    
    if se3_smiles:
        se3_raw_df = pd.DataFrame(se3_smiles)
        se3_raw_path = os.path.join(BASE, "05_generated_candidates", "se3", "generated_se3_raw.csv")
        os.makedirs(os.path.dirname(se3_raw_path), exist_ok=True)
        se3_raw_df.to_csv(se3_raw_path, index=False)
        print(f"SE(3) candidates with SMILES: {len(se3_raw_df)}")
    else:
        print("SE(3): No valid SMILES reconstructed (model may need more training)")
else:
    print("No SE(3) checkpoint found -- skipping SE(3) generation")
    print("   Pipeline continues with baseline generation only")

# ── CELL 5: SELFIES Baseline Generation ───────────────────
print("\n--- SELFIES Baseline Generation ---")

from emd_pipeline.candidate_generation import generate_selfies_candidates

selfies_path = os.path.join(BASE, "05_generated_candidates", "selfies", "generated_selfies_filtered.csv")
selfies_df = generate_selfies_candidates(
    seeds_df,
    mutations_per_seed=5 if TINY_DEBUG else config["generation"]["selfies_mutations_per_seed"],
    seed=RANDOM_SEED,
    output_path=selfies_path,
)

# ── CELL 6: RDKit Baseline Generation ─────────────────────
print("\n--- RDKit Baseline Generation ---")

from emd_pipeline.candidate_generation import generate_rdkit_candidates

rdkit_path = os.path.join(BASE, "05_generated_candidates", "rdkit", "generated_rdkit_filtered.csv")
rdkit_df = generate_rdkit_candidates(
    seeds_df,
    subs_per_seed=3 if TINY_DEBUG else config["generation"]["rdkit_substitutions_per_seed"],
    seed=RANDOM_SEED,
    output_path=rdkit_path,
)

# ── CELL 7: Filter All Candidates ─────────────────────────
print("\n--- Filtering ---")

from emd_pipeline.candidate_generation import filter_candidates

# Filter each source
dfs_to_merge = []

if len(se3_raw_df) > 0:
    se3_filtered = filter_candidates(se3_raw_df, training_smiles=training_smiles)
    se3_filtered_path = os.path.join(BASE, "05_generated_candidates", "se3", "generated_se3_filtered.csv")
    se3_filtered.to_csv(se3_filtered_path, index=False)
    dfs_to_merge.append(se3_filtered)

if len(selfies_df) > 0:
    selfies_filtered = filter_candidates(selfies_df, training_smiles=training_smiles)
    dfs_to_merge.append(selfies_filtered)

if len(rdkit_df) > 0:
    rdkit_filtered = filter_candidates(rdkit_df, training_smiles=training_smiles)
    dfs_to_merge.append(rdkit_filtered)

# ── CELL 8: Merge Candidates ──────────────────────────────
from emd_pipeline.candidate_generation import merge_candidates, compute_generation_metrics

merged_path = os.path.join(BASE, "05_generated_candidates", "merged", "generated_merged_filtered.csv")

if dfs_to_merge:
    merged_df = merge_candidates(*dfs_to_merge, output_path=merged_path)
else:
    merged_df = pd.DataFrame()
    print("⚠️ No candidates generated!")

# ── CELL 9: Generation Metrics ────────────────────────────
metrics_path = os.path.join(BASE, "05_generated_candidates", "merged", "generation_comparison_metrics.csv")

if len(merged_df) > 0:
    metrics = compute_generation_metrics(merged_df, training_smiles=training_smiles, output_path=metrics_path)

# ── CELL 10: Validation & Summary ─────────────────────────
total_candidates = len(merged_df) if len(merged_df) > 0 else 0
gate_pass = total_candidates >= (10 if TINY_DEBUG else 100)

update_registry(
    os.path.join(BASE, "00_project_registry", "run_registry.csv"),
    {
        "notebook": "04_generation",
        "step": "complete",
        "status": "complete",
        "output_path": merged_path,
        "notes": f"{total_candidates} merged candidates",
    }
)

print("\n" + "=" * 60)
print("NOTEBOOK 04 COMPLETE — Hybrid Candidate Generation")
print("=" * 60)
print(f"  SE(3) candidates: {len(se3_raw_df)}")
print(f"  SELFIES candidates: {len(selfies_df)}")
print(f"  RDKit candidates: {len(rdkit_df)}")
print(f"  Merged filtered: {total_candidates}")
print(f"  {'✅' if gate_pass else '⚠️'} Success gate: {'PASS' if gate_pass else 'NEEDS ATTENTION'}")
print(f"  Next: Run Notebook 05 (Docking)")
