# ============================================================
# NOTEBOOK 02: Ligand Features and Splits
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# Purpose: Compute descriptors, generate conformers, build SE(3) graphs
# Owner: Student 3 + Student 4
# Day: 3
# Success gate: 90%+ pass RDKit, 100+ graph tensors
# ============================================================

# ── CELL 1: Install & Mount ───────────────────────────────
# !pip -q install rdkit-pypi torch e3nn pyyaml tqdm
# from google.colab import drive
# drive.mount("/content/drive")

# ── CELL 2: Config ─────────────────────────────────────────
import os, sys, time
import numpy as np
import pandas as pd

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

# ── CELL 3: Load Curated Data ─────────────────────────────
curated_path = os.path.join(BASE, "02_curated_data", "jak2_curated_ligands.csv")
curated_df = pd.read_csv(curated_path)
print(f"Loaded {len(curated_df)} curated ligands")

if TINY_DEBUG:
    curated_df = curated_df.head(config["data"]["tiny_debug_size"])
    print(f"Tiny Debug mode: using {len(curated_df)} molecules")

# ── CELL 4: Compute Descriptors ───────────────────────────
from emd_pipeline.feature_engineering import compute_descriptors

features_path = os.path.join(BASE, "03_features", "ligand_features.csv")
features_df = compute_descriptors(
    curated_df,
    smiles_col="canonical_smiles",
    output_path=features_path,
)
print(f"\nFeature summary:\n{features_df.describe()}")

# ── CELL 5: Generate 3D Conformers ────────────────────────
from emd_pipeline.feature_engineering import generate_conformers

conformer_path = os.path.join(BASE, "03_features", "conformers", "curated_conformers.sdf")
mols_3d = generate_conformers(
    curated_df,
    smiles_col="canonical_smiles",
    output_sdf=conformer_path,
    optimize=True,
)

success_rate = len(mols_3d) / max(len(curated_df), 1) * 100
print(f"Conformer success rate: {success_rate:.1f}%")

# ── CELL 6: Build SE(3) Graph Tensors ─────────────────────
from emd_pipeline.feature_engineering import build_se3_graph_tensors

graphs_pt_path = os.path.join(BASE, "03_features", "ligand_graphs", "se3_graphs.pt")
graphs_index_path = os.path.join(BASE, "03_features", "ligand_graphs", "se3_graph_index.csv")

graphs = build_se3_graph_tensors(
    mols_3d,
    output_pt=graphs_pt_path,
    output_index=graphs_index_path,
)

# Verify graph structure
if graphs:
    g = graphs[0]
    print(f"\nSample graph structure:")
    print(f"  atom_features: {g['atom_features'].shape}")
    print(f"  coords: {g['coords'].shape}")
    print(f"  edge_index: {g['edge_index'].shape}")
    print(f"  edge_features: {g['edge_features'].shape}")
    print(f"  num_atoms: {g['num_atoms']}")
    print(f"  mol_id: {g['mol_id']}")

# ── CELL 7: Save Split ID Files ───────────────────────────
from emd_pipeline.feature_engineering import save_splits

if "split" in curated_df.columns:
    splits_dir = os.path.join(BASE, "03_features", "splits")
    save_splits(curated_df, splits_dir)
else:
    # Create simple split
    n = len(curated_df)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    splits_dir = os.path.join(BASE, "03_features", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    for split_name, split_indices in [
        ("train", indices[:n_train]),
        ("val", indices[n_train:n_train + n_val]),
        ("test", indices[n_train + n_val:]),
    ]:
        ids = curated_df.iloc[split_indices]["mol_id"].tolist()
        with open(os.path.join(splits_dir, f"{split_name}_ids.txt"), "w") as f:
            f.write("\n".join(str(i) for i in ids))
        print(f"  {split_name}: {len(ids)} molecules")

# ── CELL 8: Validation ────────────────────────────────────
print("\n--- Feature Validation ---")
rdkit_pass_rate = features_df["passes_rdkit"].mean() * 100
print(f"RDKit pass rate: {rdkit_pass_rate:.1f}%")
print(f"Graph tensors: {len(graphs)}")
print(f"Conformers: {len(mols_3d)}")

gate_pass = rdkit_pass_rate >= 90 and len(graphs) >= (10 if TINY_DEBUG else 100)
print(f"\n{'✅' if gate_pass else '⚠️'} Success gate: {'PASS' if gate_pass else 'NEEDS ATTENTION'}")

# ── CELL 9: Feature Distribution Plots ────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, col, title in zip(
    axes.flat,
    ["mw", "logp", "tpsa", "qed", "rotatable_bonds", "max_ring_size"],
    ["Molecular Weight", "LogP", "TPSA", "QED", "Rotatable Bonds", "Max Ring Size"]
):
    if col in features_df.columns:
        ax.hist(features_df[col].dropna(), bins=20, color="#673AB7", alpha=0.7, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(col)

plt.suptitle("Ligand Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plot_path = os.path.join(BASE, "03_features", "feature_distributions.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Feature distribution plot saved: {plot_path}")

# ── CELL 10: Summary ──────────────────────────────────────
update_registry(
    os.path.join(BASE, "00_project_registry", "run_registry.csv"),
    {
        "notebook": "02_features",
        "step": "complete",
        "status": "complete",
        "output_path": features_path,
        "notes": f"{len(features_df)} features, {len(graphs)} graphs, {len(mols_3d)} conformers",
    }
)

print("\n" + "=" * 60)
print("NOTEBOOK 02 COMPLETE — Features & SE(3) Dataset")
print("=" * 60)
print(f"  Features: {len(features_df)} molecules")
print(f"  Conformers: {len(mols_3d)} molecules")
print(f"  SE(3) graphs: {len(graphs)} tensors")
print(f"  Next: Run Notebook 03 (SE(3) Model Training)")
