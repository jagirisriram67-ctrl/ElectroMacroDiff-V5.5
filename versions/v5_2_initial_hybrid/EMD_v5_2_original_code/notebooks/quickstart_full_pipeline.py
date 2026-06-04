# ============================================================
# QUICKSTART: Run Entire Pipeline in One Notebook
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# This is a single-file Colab notebook that runs the FULL
# pipeline end-to-end in Tiny Debug mode (~25 molecules).
# Use this to validate the pipeline works before scaling up.
# ============================================================

# ── CELL 1: Install All Dependencies ──────────────────────
# !pip -q install rdkit-pypi selfies chembl-webresource-client biopython
# !pip -q install torch e3nn torchdiffeq
# !pip -q install matplotlib seaborn py3Dmol openpyxl python-docx pyyaml tqdm tabulate meeko

# ── CELL 2: Mount Drive & Setup ───────────────────────────
# from google.colab import drive
# drive.mount("/content/drive")

import os, sys, time, json, shutil
import numpy as np
import pandas as pd
import torch

# For local testing, use current dir. In Colab, change to Drive path.
BASE = "."
sys.path.insert(0, BASE)

# ── CELL 3: Initialize Project ────────────────────────────
from emd_pipeline.config_registry import (
    load_config, ensure_dirs, set_all_seeds,
    save_progress, update_registry, save_environment_versions
)

config = load_config(os.path.join(BASE, "campaign_config.yaml"))
ensure_dirs(BASE)
set_all_seeds(42)

# Save environment snapshot (blueprint M0 requirement)
save_environment_versions(os.path.join(BASE, "00_project_registry", "environment_versions.txt"))

TINY_DEBUG = True
TINY_SIZE = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CELL 4: M1 — Data Collection ──────────────────────────
print("\n" + "="*60 + "\n  M1: DATA COLLECTION\n" + "="*60)

from emd_pipeline.data_collection import (
    fetch_chembl_jak2_activities, fetch_pdb_structure,
    curate_activities, scaffold_split, analyze_macrocycles
)

# Download PDB
pdb_dir = os.path.join(BASE, "01_raw_data", "pdb")
pdb_path = os.path.join(pdb_dir, "5AEP.pdb")
if not os.path.exists(pdb_path):
    try:
        pdb_path = fetch_pdb_structure("5AEP", pdb_dir)
    except:
        print("PDB download failed -- will continue without")

# Fetch ChEMBL data
raw_path = os.path.join(BASE, "01_raw_data", "chembl", "jak2_activities_raw.csv")
if not os.path.exists(raw_path):
    try:
        raw_df = fetch_chembl_jak2_activities(raw_path, max_records=200 if TINY_DEBUG else 5000)
    except Exception as e:
        print(f"ChEMBL fetch failed: {e}")
        # Create minimal synthetic dataset for testing
        from rdkit import Chem
        test_smiles = [
            "c1ccc2c(c1)cc1ccccc1n2", "CC(=O)Oc1ccccc1C(=O)O",
            "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",
            "c1ccc(-c2ccccn2)cc1", "O=C(O)c1ccccc1O",
            "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1", "c1ccc2[nH]ccc2c1",
            "c1ccc(-c2cccc3ccccc23)cc1", "O=c1ccoc2ccccc12",
            "c1ccc2nc(-c3ccccc3)ccc2c1", "CC(=O)c1ccccc1",
            "OC(=O)c1cc(O)c(O)c(O)c1", "c1ccc(-n2cccc2)cc1",
            "c1ccc2c(c1)ccc1ccccc12", "O=C(O)/C=C/c1ccccc1",
            "c1ccc(NC(=O)c2ccccc2)cc1", "Cc1ccc(S(N)(=O)=O)cc1",
            "c1ccc(-c2nc3ccccc3s2)cc1", "O=C1CCc2ccccc2N1",
            "c1ccc2c(c1)oc1ccccc12", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "c1ccc(-c2ccc3[nH]ccc3c2)cc1", "COc1ccc2c(c1)cc(=O)oc2",
            "c1ccc(-c2ccncc2)cc1", "c1ccc(-c2cccnc2)cc1",
        ]
        raw_df = pd.DataFrame({
            "canonical_smiles": test_smiles,
            "activity_value": np.random.uniform(1, 10000, len(test_smiles)),
            "activity_units": ["nM"] * len(test_smiles),
            "activity_type": ["IC50"] * len(test_smiles),
            "source": ["synthetic_test"] * len(test_smiles),
        })
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        raw_df.to_csv(raw_path, index=False)
        print(f"Created synthetic dataset: {len(raw_df)} molecules")
else:
    raw_df = pd.read_csv(raw_path)

# Curate
curated_path = os.path.join(BASE, "02_curated_data", "jak2_curated_ligands.csv")
curated_df = curate_activities(raw_df, mw_min=150, mw_max=900, output_path=curated_path)

if TINY_DEBUG:
    curated_df = curated_df.head(TINY_SIZE)

# Split
curated_df = scaffold_split(curated_df, seed=42)
curated_df.to_csv(curated_path, index=False)

# Macrocycle analysis
macro_stats = analyze_macrocycles(curated_df)
print(f"M1 complete: {len(curated_df)} curated ligands")

# ── CELL 5: M2 — Feature Engineering ──────────────────────
print("\n" + "="*60 + "\n  M2: FEATURE ENGINEERING\n" + "="*60)

from emd_pipeline.feature_engineering import (
    compute_descriptors, generate_conformers, build_se3_graph_tensors, save_splits
)

# Descriptors
features_path = os.path.join(BASE, "03_features", "ligand_features.csv")
features_df = compute_descriptors(curated_df, output_path=features_path)

# Conformers
conformer_path = os.path.join(BASE, "03_features", "conformers", "curated_conformers.sdf")
mols_3d = generate_conformers(curated_df, output_sdf=conformer_path)

# SE(3) graphs
graphs_pt = os.path.join(BASE, "03_features", "ligand_graphs", "se3_graphs.pt")
graphs_idx = os.path.join(BASE, "03_features", "ligand_graphs", "se3_graph_index.csv")
graphs = build_se3_graph_tensors(mols_3d, output_pt=graphs_pt, output_index=graphs_idx)

# Splits
if "split" in curated_df.columns:
    save_splits(curated_df, os.path.join(BASE, "03_features", "splits"))

print(f"M2 complete: {len(features_df)} features, {len(mols_3d)} conformers, {len(graphs)} graphs")

# ── CELL 6: M3 — SE(3) Model Training ─────────────────────
print("\n" + "="*60 + "\n  M3: SE(3) FLOW MATCHING MODEL\n" + "="*60)

from emd_pipeline.se3_flow_model import (
    MolecularGraphDataset, collate_fn, SE3FlowMatchingModel,
    train_se3_model, plot_training_curve
)

max_atoms = config["model"]["max_atoms"]
train_ids = [g["mol_id"] for g in graphs[:max(1, int(len(graphs)*0.8))]]
val_ids = [g["mol_id"] for g in graphs[int(len(graphs)*0.8):]]

train_dataset = MolecularGraphDataset(graphs, max_atoms=max_atoms, split_ids=train_ids)
val_dataset = MolecularGraphDataset(graphs, max_atoms=max_atoms, split_ids=val_ids)
if len(val_dataset) == 0:
    val_dataset = train_dataset

atom_dim = train_dataset[0]["atom_features"].shape[1]
model = SE3FlowMatchingModel(
    atom_feature_dim=atom_dim,
    hidden_dim=64,  # Smaller for quickstart
    num_layers=2,
    num_heads=2,
    max_atoms=max_atoms,
)

ckpt_dir = os.path.join(BASE, "04_models_checkpoints", "se3_flow")
train_config = {
    "epochs_target": 10 if TINY_DEBUG else 100,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "checkpoint_every": 5,
    "validate_every": 5,
    "random_seed": 42,
}

trained_model, log_df = train_se3_model(
    model, train_dataset, val_dataset, train_config, ckpt_dir, device
)

curve_path = os.path.join(ckpt_dir, "training_curve.png")
plot_training_curve(log_df, curve_path)
print(f"M3 complete: {len(log_df)} epochs trained")

# ── CELL 7: M4 — Candidate Generation ─────────────────────
print("\n" + "="*60 + "\n  M4: CANDIDATE GENERATION\n" + "="*60)

from emd_pipeline.candidate_generation import (
    generate_selfies_candidates, generate_rdkit_candidates,
    filter_candidates, merge_candidates, compute_generation_metrics
)
from emd_pipeline.se3_flow_model import generate_se3_candidates, coords_to_smiles

training_smiles = set(curated_df["canonical_smiles"].dropna().tolist())
seeds_df = curated_df.head(10)

# SELFIES
selfies_path = os.path.join(BASE, "05_generated_candidates", "selfies", "generated_selfies_filtered.csv")
selfies_df = generate_selfies_candidates(seeds_df, mutations_per_seed=5, output_path=selfies_path)

# RDKit
rdkit_path = os.path.join(BASE, "05_generated_candidates", "rdkit", "generated_rdkit_filtered.csv")
rdkit_df = generate_rdkit_candidates(seeds_df, subs_per_seed=3, output_path=rdkit_path)

# SE(3) Flow Matching Generation
print("\n--- SE(3) Flow Matching Generation ---")
se3_candidates = generate_se3_candidates(
    trained_model, graphs[:min(10, len(graphs))],
    num_steps=20 if TINY_DEBUG else 50, device=device
)
se3_smiles = coords_to_smiles(se3_candidates, graphs)
if se3_smiles:
    se3_df = pd.DataFrame(se3_smiles)
    se3_gen_path = os.path.join(BASE, "05_generated_candidates", "se3", "generated_se3_filtered.csv")
    os.makedirs(os.path.dirname(se3_gen_path), exist_ok=True)
    se3_df.to_csv(se3_gen_path, index=False)
    print(f"SE3: {len(se3_df)} molecules with SMILES")
else:
    se3_df = pd.DataFrame()
    print("SE3: No valid SMILES generated (model needs more training)")

# Filter & merge all sources
dfs = []
if len(selfies_df) > 0:
    dfs.append(filter_candidates(selfies_df, training_smiles=training_smiles, mw_range=(50, 900)))
if len(rdkit_df) > 0:
    dfs.append(filter_candidates(rdkit_df, training_smiles=training_smiles, mw_range=(50, 900)))
if len(se3_df) > 0:
    dfs.append(filter_candidates(se3_df, training_smiles=training_smiles, mw_range=(50, 900)))

merged_path = os.path.join(BASE, "05_generated_candidates", "merged", "generated_merged_filtered.csv")
merged_df = merge_candidates(*dfs, output_path=merged_path) if dfs else pd.DataFrame()

metrics_path = os.path.join(BASE, "05_generated_candidates", "merged", "generation_comparison_metrics.csv")
if len(merged_df) > 0:
    compute_generation_metrics(merged_df, output_path=metrics_path)

# Source summary
if len(merged_df) > 0 and "source_generator" in merged_df.columns:
    for src, count in merged_df["source_generator"].value_counts().items():
        print(f"  {src}: {count} candidates")

print(f"M4 complete: {len(merged_df)} merged candidates")

# ── CELL 8: M5 — Docking (Placeholder) ────────────────────
print("\n" + "="*60 + "\n  M5: DOCKING\n" + "="*60)

# For quickstart, generate placeholder docking scores
# Real docking requires Vina installation
scores_records = []
dock_smiles = merged_df["canonical_smiles"].tolist() if len(merged_df) > 0 else []
np.random.seed(42)

for i, smi in enumerate(dock_smiles[:50]):
    scores_records.append({
        "candidate_id": merged_df.iloc[i].get("candidate_id", f"D{i}") if i < len(merged_df) else f"D{i}",
        "smiles": smi,
        "docking_engine": "placeholder",
        "receptor_pdb": "5AEP",
        "grid_center_x": 25.0, "grid_center_y": 10.0, "grid_center_z": 15.0,
        "grid_size_x": 22, "grid_size_y": 22, "grid_size_z": 22,
        "best_score": np.random.uniform(-10, -4),
        "pose_rank": 1,
        "control_or_generated": "generated",
        "notes": "placeholder for quickstart",
    })

scores_df = pd.DataFrame(scores_records)
scores_path = os.path.join(BASE, "06_docking", "scores", "docking_scores.csv")
os.makedirs(os.path.dirname(scores_path), exist_ok=True)
scores_df.to_csv(scores_path, index=False)
print(f"M5 complete: {len(scores_df)} docking scores (placeholder)")

# ── CELL 9: M6 — ADMET & Synthesis ────────────────────────
print("\n" + "="*60 + "\n  M6: ADMET & SYNTHESIS\n" + "="*60)

from emd_pipeline.admet_synthesis import (
    compute_admet_scores, compute_synthesis_scores,
    compute_safety_proxy, apply_filter_flags
)

admet_path = os.path.join(BASE, "07_admet_synthesis", "admet_scores.csv")
admet_df = compute_admet_scores(scores_df, smiles_col="smiles", output_path=admet_path)

synth_path = os.path.join(BASE, "07_admet_synthesis", "synthesis_scores.csv")
synth_df = compute_synthesis_scores(scores_df, smiles_col="smiles", output_path=synth_path)

safety_path = os.path.join(BASE, "07_admet_synthesis", "safety_proxy_notes.csv")
safety_df = compute_safety_proxy(scores_df, smiles_col="smiles", output_path=safety_path)

filter_path = os.path.join(BASE, "07_admet_synthesis", "filter_flags.csv")
apply_filter_flags(admet_df, output_path=filter_path)

print(f"M6 complete: {len(admet_df)} ADMET, {len(synth_df)} synth, {len(safety_df)} safety")

# ── CELL 10: M7 — Final Ranking & TPP ─────────────────────
print("\n" + "="*60 + "\n  M7: FINAL RANKING & TPP\n" + "="*60)

from emd_pipeline.ranking_report import (
    consensus_ranking, generate_molecule_cards,
    save_top_sdf, generate_tpp_report
)

ranked_path = os.path.join(BASE, "08_final_ranking", "final_ranked_candidates.csv")
ranked_df = consensus_ranking(
    scores_df, admet_df, synth_df, safety_df,
    candidates_df=merged_df, output_path=ranked_path
)

# Molecule cards
cards_path = os.path.join(BASE, "08_final_ranking", "top10_molecule_cards.xlsx")
generate_molecule_cards(ranked_df, cards_path, top_n=10)

# Top 5 SDF
sdf_path = os.path.join(BASE, "08_final_ranking", "final_top5.sdf")
save_top_sdf(ranked_df, sdf_path, top_n=5)

# TPP Report
reports_dir = os.path.join(BASE, "09_reports")
generate_tpp_report(ranked_df, config, reports_dir)

# ── CELL 11: Campaign Dashboard & Log ─────────────────────
from emd_pipeline.campaign_dashboard import generate_daily_log, check_campaign_status

# Generate daily progress log
generate_daily_log(BASE)

# Quick status check
status = check_campaign_status(BASE)

# ── CELL 12: Final Summary ────────────────────────────────
print("\n" + "="*70)
print("  ElectroMacroDiff V5.2 Hybrid — QUICKSTART COMPLETE!")
print("="*70)
print(f"  M1 Curated ligands:    {len(curated_df)}")
print(f"  M2 Features/graphs:    {len(features_df)} / {len(graphs)}")
print(f"  M3 Training epochs:    {len(log_df)}")
print(f"  M4 Merged candidates:  {len(merged_df)}")
print(f"  M5 Docked:             {len(scores_df)}")
print(f"  M6 ADMET scored:       {len(admet_df)}")
print(f"  M7 Final ranked:       {len(ranked_df)}")
print(f"  Campaign progress:     {status['overall_pct']:.0f}%")
print(f"\n  Top 3 candidates:")
for _, row in ranked_df.head(3).iterrows():
    print(f"    #{row.get('rank','?')}: {row.get('candidate_id','?')} "
          f"score={row.get('final_weighted_score',0):.4f} "
          f"dock={row.get('best_score','N/A')}")
print(f"\n  All outputs saved under: {BASE}")
print(f"  Daily log: {BASE}/11_logs/daily_log.md")
print(f"  Pipeline is ready for full-scale Colab execution!")
