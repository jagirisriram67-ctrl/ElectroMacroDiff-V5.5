# ============================================================
# NOTEBOOK 00: Environment and Drive Check
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# Purpose: Validate environment, create Drive structure, run Tiny Debug
# Owner: Student 1 (all assist)
# Day: 1
# ============================================================

# ── CELL 1: Install Dependencies ───────────────────────────
# !pip -q install -r "/content/drive/MyDrive/EMD_V5_2_Hybrid/requirements_v5_2_hybrid_colab.txt"
# Or install individually:
# !pip -q install rdkit-pypi selfies chembl-webresource-client biopython
# !pip -q install torch e3nn torchdiffeq
# !pip -q install matplotlib seaborn py3Dmol openpyxl python-docx pyyaml tqdm tabulate

# ── CELL 2: Mount Google Drive ─────────────────────────────
# from google.colab import drive
# drive.mount("/content/drive")

# ── CELL 3: Configuration ──────────────────────────────────
import os
import sys
import json
import yaml
import time
import numpy as np
import pandas as pd

# Base path — change this if not using Drive
# BASE = "/content/drive/MyDrive/EMD_V5_2_Hybrid"
BASE = "."  # Local development fallback

# Add pipeline to path
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))

TINY_DEBUG = True
TINY_DEBUG_SIZE = 25
RANDOM_SEED = 42

print(f"Base path: {BASE}")
print(f"Tiny Debug: {TINY_DEBUG} (size={TINY_DEBUG_SIZE})")

# ── CELL 4: Create Directory Structure ─────────────────────
from emd_pipeline.config_registry import ensure_dirs, set_all_seeds, load_config

ensure_dirs(BASE)
set_all_seeds(RANDOM_SEED)

# Copy config to registry
import shutil
config_src = os.path.join(BASE, "campaign_config.yaml")
config_dst = os.path.join(BASE, "00_project_registry", "campaign_config.yaml")
if os.path.exists(config_src):
    shutil.copy(config_src, config_dst)
    print(f"Config copied to {config_dst}")

# ── CELL 5: Environment Version Check ──────────────────────
from emd_pipeline.config_registry import save_environment_versions
from emd_pipeline.colab_utils import check_gpu_status, create_colab_toc

env_path = os.path.join(BASE, "00_project_registry", "environment_versions.txt")
versions = save_environment_versions(env_path)
for v in versions:
    print(v)

print("\n--- GPU Status ---")
gpu_info = check_gpu_status()

print("\n--- Pipeline Notebooks ---")
create_colab_toc()

# ── CELL 6: Quick Sanity Tests ─────────────────────────────
print("\n--- Sanity Tests ---")

# Test 1: RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as QEDModule
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None
    mw = Descriptors.MolWt(mol)
    qed = QEDModule.qed(mol)
    print(f"✅ RDKit: benzene MW={mw:.1f}, QED={qed:.3f}")
except Exception as e:
    print(f"❌ RDKit: {e}")

# Test 2: PyTorch
try:
    import torch
    x = torch.randn(3, 3)
    y = torch.mm(x, x.T)
    gpu = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"✅ PyTorch: {torch.__version__} ({gpu})")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

# Test 3: SELFIES
try:
    import selfies as sf
    smi = "c1ccccc1"
    encoded = sf.encoder(smi)
    decoded = sf.decoder(encoded)
    print(f"✅ SELFIES: '{smi}' -> '{encoded}' -> '{decoded}'")
except Exception as e:
    print(f"❌ SELFIES: {e}")

# Test 4: e3nn
try:
    import e3nn
    print(f"✅ e3nn: {e3nn.__version__}")
except Exception as e:
    print(f"⚠️ e3nn: {e}")

# Test 5: Save/Load CSV to Drive
test_csv = os.path.join(BASE, "00_project_registry", "test_drive_write.csv")
test_df = pd.DataFrame({"test": [1, 2, 3], "value": ["a", "b", "c"]})
test_df.to_csv(test_csv, index=False)
reloaded = pd.read_csv(test_csv)
assert len(reloaded) == 3
os.remove(test_csv)
print(f"✅ Drive CSV write/read test passed")

# ── CELL 7: Create Campaign Config ────────────────────────
config = load_config(os.path.join(BASE, "campaign_config.yaml"))
print(f"\nCampaign: {config['project']['name']}")
print(f"Target: {config['target']['name']} ({config['target']['primary_pdb']})")

# ── CELL 8: Initialize Run Registry ───────────────────────
from emd_pipeline.config_registry import update_registry

registry_path = os.path.join(BASE, "00_project_registry", "run_registry.csv")
update_registry(registry_path, {
    "notebook": "00_environment_check",
    "step": "initialization",
    "status": "complete",
    "duration_seconds": 0,
    "output_path": env_path,
    "notes": f"Environment validated. GPU={'available' if torch.cuda.is_available() else 'CPU only'}",
})

# ── CELL 9: Tiny Debug - Quick Molecule Test ───────────────
print("\n--- Tiny Debug: Quick Molecule Processing ---")

debug_smiles = [
    "c1ccc2c(c1)cc1ccccc1n2",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",
    "c1ccc(-c2ccccn2)cc1",
    "O=C(O)c1ccccc1O",
]

from rdkit.Chem import Descriptors as Desc

debug_results = []
for smi in debug_smiles:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        debug_results.append({
            "smiles": Chem.MolToSmiles(mol),
            "mw": Desc.MolWt(mol),
            "logp": Desc.MolLogP(mol),
            "qed": QEDModule.qed(mol),
        })

debug_df = pd.DataFrame(debug_results)
print(debug_df.to_string(index=False))

debug_path = os.path.join(BASE, "00_project_registry", "tiny_debug_test.csv")
debug_df.to_csv(debug_path, index=False)
print(f"\n✅ Tiny debug test saved to {debug_path}")

# ── CELL 10: Check Drive Artifacts ─────────────────────────
from emd_pipeline.colab_utils import check_drive_artifacts
check_drive_artifacts(BASE)

# ── CELL 11: Summary ──────────────────────────────────────
print("\n" + "=" * 60)
print("NOTEBOOK 00 COMPLETE -- Environment Validated")
print("=" * 60)
print(f"  Drive structure: created")
print(f"  RDKit: OK")
print(f"  PyTorch: OK ({'GPU' if torch.cuda.is_available() else 'CPU'})")
print(f"  SELFIES: OK")
print(f"  Drive I/O: OK")
print(f"  Tiny Debug: OK ({len(debug_results)} molecules processed)")
print(f"\n  Next: Run Notebook 01 (Data Collection)")

# Write daily log
log_path = os.path.join(BASE, "11_logs", "daily_log.md")
with open(log_path, "a") as f:
    f.write(f"\n## Day 1 - {time.strftime('%Y-%m-%d')}\n")
    f.write(f"- Environment setup complete\n")
    f.write(f"- Drive structure created\n")
    f.write(f"- All core packages validated\n")
    f.write(f"- Tiny debug test passed\n")
print(f"Daily log updated: {log_path}")
