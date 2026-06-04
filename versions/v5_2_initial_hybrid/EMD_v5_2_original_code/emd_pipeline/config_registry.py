"""
EMD V5.2 Hybrid — Configuration and Registry Utilities
======================================================
Handles campaign configuration loading, progress tracking, and
the resume/checkpoint system that protects against Colab disconnects.
"""

import os
import json
import time
import yaml
import csv
from datetime import datetime
from pathlib import Path


# ── Configuration ───────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    """Load campaign configuration from YAML file.
    
    Args:
        config_path: Path to campaign_config.yaml.
                     If None, looks in standard locations.
    
    Returns:
        dict: Full configuration dictionary.
    """
    if config_path is None:
        # Try standard locations
        candidates = [
            "/content/drive/MyDrive/EMD_V5_2_Hybrid/campaign_config.yaml",
            "/content/drive/MyDrive/EMD_V5_2_Hybrid/00_project_registry/campaign_config.yaml",
            "campaign_config.yaml",
        ]
        for c in candidates:
            if os.path.exists(c):
                config_path = c
                break
        if config_path is None:
            raise FileNotFoundError(
                "campaign_config.yaml not found. Upload it to Drive or provide the path."
            )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_base_path(config: dict) -> str:
    """Get the Drive base path from config.
    
    Falls back to local path if Drive is not mounted.
    """
    drive_base = config.get("paths", {}).get("drive_base", ".")
    if os.path.exists(drive_base):
        return drive_base
    local_base = config.get("paths", {}).get("local_base", ".")
    return local_base


def ensure_dirs(base: str):
    """Create the full EMD V5.2 Hybrid directory tree under base."""
    dirs = [
        "00_project_registry",
        "01_raw_data/chembl",
        "01_raw_data/bindingdb",
        "01_raw_data/pubchem",
        "01_raw_data/pdb",
        "01_raw_data/literature",
        "02_curated_data",
        "03_features/ligand_graphs",
        "03_features/conformers",
        "03_features/splits",
        "04_models_checkpoints/se3_flow",
        "04_models_checkpoints/baseline_models",
        "05_generated_candidates/se3",
        "05_generated_candidates/selfies",
        "05_generated_candidates/rdkit",
        "05_generated_candidates/merged",
        "06_docking/receptor",
        "06_docking/ligands_pdbqt",
        "06_docking/poses",
        "06_docking/scores",
        "06_docking/images",
        "07_admet_synthesis",
        "08_final_ranking",
        "09_reports",
        "10_notebooks",
        "11_logs",
    ]
    for d in dirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)
    print(f"[OK] Created {len(dirs)} directories under {base}")


# ── Progress / Resume System ───────────────────────────────

def load_progress(path: str) -> dict:
    """Load progress from JSON file for resumable execution.
    
    Args:
        path: Path to progress JSON file.
    
    Returns:
        dict: Progress state. Returns default if file doesn't exist.
    """
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"last_completed_index": -1, "status": "not_started"}


def save_progress(path: str, progress: dict):
    """Save progress to JSON file.
    
    Args:
        path: Path to write progress JSON.
        progress: Dictionary with progress state.
    """
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def update_registry(registry_path: str, entry: dict):
    """Append an entry to the run registry CSV.
    
    Args:
        registry_path: Path to run_registry.csv
        entry: Dictionary with run metadata.
    """
    file_exists = os.path.exists(registry_path)
    fieldnames = [
        "timestamp", "notebook", "step", "status",
        "duration_seconds", "output_path", "notes"
    ]
    # Add timestamp if not present
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)


# ── Environment Snapshot ────────────────────────────────────

def save_environment_versions(output_path: str):
    """Save installed package versions to a text file.
    
    Args:
        output_path: Path to write environment_versions.txt
    """
    import importlib
    
    packages = [
        "torch", "e3nn", "torchdiffeq", "rdkit",
        "selfies", "numpy", "pandas", "scipy",
        "sklearn", "matplotlib", "seaborn", "biopython",
        "yaml", "requests", "tqdm", "openpyxl",
    ]
    
    lines = [
        f"EMD V5.2 Hybrid — Environment Snapshot",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
    ]
    
    for pkg in packages:
        try:
            # Handle package name differences
            pkg_import = pkg
            if pkg == "sklearn":
                pkg_import = "sklearn"
            elif pkg == "biopython":
                pkg_import = "Bio"
            elif pkg == "yaml":
                pkg_import = "yaml"
            
            mod = importlib.import_module(pkg_import)
            version = getattr(mod, "__version__", "installed (version unknown)")
            lines.append(f"{pkg}: {version}")
        except ImportError:
            lines.append(f"{pkg}: NOT INSTALLED")
    
    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f"\nGPU: {torch.cuda.get_device_name(0)}")
            lines.append(f"CUDA: {torch.version.cuda}")
        else:
            lines.append(f"\nGPU: Not available (CPU mode)")
    except Exception:
        lines.append(f"\nGPU: Check failed")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"[OK] Environment versions saved to {output_path}")
    return lines


# ── Seed Management ─────────────────────────────────────────

def set_all_seeds(seed: int = 42):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed value.
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"[OK] All seeds set to {seed}")


# ── Tiny Debug Mode ─────────────────────────────────────────

def get_tiny_debug_flag(config: dict) -> bool:
    """Check if we're in Tiny Debug mode."""
    return config.get("data", {}).get("tiny_debug", True)


def get_tiny_debug_size(config: dict) -> int:
    """Get the Tiny Debug set size."""
    return config.get("data", {}).get("tiny_debug_size", 25)


# ── Drive Mount Helper ──────────────────────────────────────

def mount_drive():
    """Mount Google Drive in Colab. No-op if not in Colab."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("[OK] Google Drive mounted")
        return True
    except ImportError:
        print("[INFO] Not in Colab -- skipping Drive mount")
        return False
    except Exception as e:
        print(f"[WARN] Drive mount failed: {e}")
        return False
