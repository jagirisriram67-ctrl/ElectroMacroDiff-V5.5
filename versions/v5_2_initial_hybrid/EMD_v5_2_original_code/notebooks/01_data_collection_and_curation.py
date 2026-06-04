# ============================================================
# NOTEBOOK 01: Data Collection and Curation
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# Purpose: Collect JAK2 data from ChEMBL/BindingDB/PDB, curate, split
# Owner: Student 2
# Day: 2
# Success gate: 300+ usable JAK2 ligands, PDB 5AEP downloaded
# ============================================================

# ── CELL 1: Install & Mount ───────────────────────────────
# !pip -q install rdkit-pypi selfies chembl-webresource-client biopython pyyaml tqdm
# from google.colab import drive
# drive.mount("/content/drive")

# ── CELL 2: Config ─────────────────────────────────────────
import os, sys, time
import numpy as np
import pandas as pd

BASE = "."  # Change to Drive path in Colab
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))

TINY_DEBUG = True  # Set False for full run
RANDOM_SEED = 42

from emd_pipeline.config_registry import (
    load_config, set_all_seeds, load_progress, save_progress, update_registry
)

config = load_config(os.path.join(BASE, "campaign_config.yaml"))
set_all_seeds(RANDOM_SEED)

# ── CELL 3: Resume Check ──────────────────────────────────
progress_path = os.path.join(BASE, "00_project_registry", "progress_01_data.json")
progress = load_progress(progress_path)
print(f"Resume state: {progress}")

# ── CELL 4: Download PDB Structure ────────────────────────
from emd_pipeline.data_collection import fetch_pdb_structure

pdb_dir = os.path.join(BASE, "01_raw_data", "pdb")

if not os.path.exists(os.path.join(pdb_dir, "5AEP.pdb")):
    pdb_path = fetch_pdb_structure("5AEP", pdb_dir)
    save_progress(progress_path, {"step": "pdb_download", "status": "complete"})
else:
    pdb_path = os.path.join(pdb_dir, "5AEP.pdb")
    print(f"PDB already downloaded: {pdb_path}")

# ── CELL 5: Fetch ChEMBL JAK2 Activities ──────────────────
from emd_pipeline.data_collection import fetch_chembl_jak2_activities

raw_chembl_path = os.path.join(BASE, "01_raw_data", "chembl", "jak2_activities_raw.csv")

if not os.path.exists(raw_chembl_path):
    max_records = 100 if TINY_DEBUG else 10000
    raw_df = fetch_chembl_jak2_activities(
        output_path=raw_chembl_path,
        max_records=max_records
    )
    save_progress(progress_path, {"step": "chembl_fetch", "status": "complete", "records": len(raw_df)})
else:
    raw_df = pd.read_csv(raw_chembl_path)
    print(f"ChEMBL data loaded: {len(raw_df)} records")

print(f"\nRaw data shape: {raw_df.shape}")
print(f"Columns: {list(raw_df.columns)}")
print(f"\nSample:")
print(raw_df.head())

# ── CELL 6: Curate Activities ─────────────────────────────
from emd_pipeline.data_collection import curate_activities

curated_path = os.path.join(BASE, "02_curated_data", "jak2_curated_ligands.csv")

curated_df = curate_activities(
    raw_df,
    mw_min=config["data"]["mw_min"],
    mw_max=config["data"]["mw_max"],
    output_path=curated_path,
)

print(f"\nCurated: {len(curated_df)} ligands")
print(curated_df.describe())

save_progress(progress_path, {"step": "curation", "status": "complete", "ligands": len(curated_df)})

# ── CELL 7: Macrocycle Analysis ────────────────────────────
from emd_pipeline.data_collection import analyze_macrocycles

macro_stats = analyze_macrocycles(curated_df)

# Save macrocycle subset
macro_df = curated_df[
    curated_df["has_macrocycle_12_20"] | curated_df["has_constrained_ring_8_11"]
].copy()
macro_path = os.path.join(BASE, "02_curated_data", "jak2_macrocycle_constrained.csv")
macro_df.to_csv(macro_path, index=False)
print(f"Macrocycle/constrained subset: {len(macro_df)} molecules saved")

# ── CELL 8: Scaffold Split ────────────────────────────────
from emd_pipeline.data_collection import scaffold_split

curated_df = scaffold_split(
    curated_df,
    train_frac=config["data"]["train_fraction"],
    val_frac=config["data"]["val_fraction"],
    seed=RANDOM_SEED,
)

split_path = os.path.join(BASE, "02_curated_data", "jak2_train_val_test_split.csv")
curated_df.to_csv(split_path, index=False)
print(f"Split data saved to {split_path}")

# ── CELL 9: Data Validation ───────────────────────────────
print("\n--- Data Validation ---")
print(f"Total curated: {len(curated_df)}")
print(f"Splits: {dict(curated_df['split'].value_counts())}")
print(f"Has SMILES: {curated_df['canonical_smiles'].notna().sum()}")
print(f"Has activity: {curated_df.get('p_activity', pd.Series()).notna().sum()}")
print(f"Macrocycles: {macro_stats.get('macrocycles_12_20', 0)}")
print(f"Constrained: {macro_stats.get('constrained_8_11', 0)}")

gate_pass = len(curated_df) >= 20 if TINY_DEBUG else len(curated_df) >= 300
print(f"\n{'✅' if gate_pass else '⚠️'} Success gate: {'PASS' if gate_pass else 'NEEDS ATTENTION'}")

# ── CELL 10: Visualization ────────────────────────────────
from emd_pipeline.visualization import plot_dataset_overview

plot_path = os.path.join(BASE, "02_curated_data", "data_overview.png")
plot_dataset_overview(curated_df, output_path=plot_path)

# ── CELL 11: Summary & Registry ───────────────────────────
update_registry(
    os.path.join(BASE, "00_project_registry", "run_registry.csv"),
    {
        "notebook": "01_data_collection",
        "step": "complete",
        "status": "complete",
        "output_path": curated_path,
        "notes": f"{len(curated_df)} curated ligands, {macro_stats.get('macrocycles_12_20', 0)} macrocycles",
    }
)

save_progress(progress_path, {"step": "complete", "status": "complete", "ligands": len(curated_df)})

print("\n" + "=" * 60)
print("NOTEBOOK 01 COMPLETE — Data Collection & Curation")
print("=" * 60)
print(f"  Curated ligands: {len(curated_df)}")
print(f"  PDB structure: 5AEP ✅")
print(f"  Next: Run Notebook 02 (Features & SE(3) Dataset)")
