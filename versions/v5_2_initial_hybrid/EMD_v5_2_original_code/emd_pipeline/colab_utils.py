"""
EMD V5.2 Hybrid — Colab Utilities
==================================
GPU monitoring, memory management, session protection,
and Colab-specific helpers for reliable pipeline execution.
"""

import os
import sys
import time
import json


def check_gpu_status():
    """Check and report GPU availability and memory."""
    info = {"gpu_available": False, "device_name": "CPU", "memory_total_gb": 0,
            "memory_used_gb": 0, "memory_free_gb": 0, "cuda_version": "N/A"}
    
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            
            info["memory_total_gb"] = round(total, 2)
            info["memory_used_gb"] = round(allocated, 2)
            info["memory_free_gb"] = round(total - reserved, 2)
    except:
        pass
    
    print(f"GPU: {info['device_name']}")
    if info["gpu_available"]:
        print(f"  CUDA: {info['cuda_version']}")
        print(f"  Memory: {info['memory_used_gb']:.1f} / {info['memory_total_gb']:.1f} GB "
              f"({info['memory_free_gb']:.1f} GB free)")
    
    return info


def clear_gpu_memory():
    """Clear GPU memory cache. Call between heavy operations."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("GPU memory cache cleared")
    except:
        pass


def get_optimal_batch_size(num_atoms, hidden_dim=128, gpu_memory_gb=None):
    """Estimate optimal batch size for SE(3) model based on available GPU memory.
    
    Conservative estimates for T4 (16GB) GPU.
    """
    if gpu_memory_gb is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            else:
                gpu_memory_gb = 4  # CPU - use small batches
        except:
            gpu_memory_gb = 4
    
    # Rough memory estimate: each molecule ~= num_atoms * hidden_dim * 4 bytes * 3 (fwd/bwd/opt)
    mem_per_mol_mb = num_atoms * hidden_dim * 4 * 3 / (1024**2)
    available_mb = gpu_memory_gb * 1024 * 0.7  # Use 70% of GPU memory
    
    batch_size = max(1, int(available_mb / mem_per_mol_mb))
    batch_size = min(batch_size, 32)  # Cap at 32
    
    print(f"Recommended batch size: {batch_size} "
          f"(GPU={gpu_memory_gb:.0f}GB, atoms={num_atoms}, hidden={hidden_dim})")
    return batch_size


def session_keepalive():
    """Print a keepalive message to prevent Colab from disconnecting.
    
    Call this inside long training loops:
        if epoch % 10 == 0: session_keepalive()
    """
    import datetime
    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] Session alive - pipeline running...")


class TimedSection:
    """Context manager for timing pipeline sections.
    
    Usage:
        with TimedSection("M3: Training"):
            train_model(...)
    """
    def __init__(self, name):
        self.name = name
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"{'='*60}")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}min"
        else:
            time_str = f"{elapsed/3600:.1f}hr"
        print(f"  [{self.name}] completed in {time_str}")


def safe_save(data, path, format="csv"):
    """Save data to Drive with retry logic for unstable connections.
    
    Retries up to 3 times with exponential backoff.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    for attempt in range(3):
        try:
            if format == "csv":
                data.to_csv(path, index=False)
            elif format == "json":
                with open(path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            elif format == "pt":
                import torch
                torch.save(data, path)
            print(f"Saved: {path}")
            return True
        except Exception as e:
            wait = 2 ** attempt
            print(f"  Save attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    
    print(f"  FAILED to save {path} after 3 attempts!")
    return False


def estimate_remaining_time(current_step, total_steps, elapsed_seconds):
    """Estimate remaining time for a loop operation."""
    if current_step <= 0:
        return "estimating..."
    
    rate = elapsed_seconds / current_step
    remaining = rate * (total_steps - current_step)
    
    if remaining < 60:
        return f"{remaining:.0f}s"
    elif remaining < 3600:
        return f"{remaining/60:.1f}min"
    else:
        return f"{remaining/3600:.1f}hr"


def create_colab_toc():
    """Print a table of contents for the pipeline notebooks."""
    toc = """
    ElectroMacroDiff V5.2 Hybrid - Pipeline Notebooks
    ==================================================
    
    NB 00: Environment & Drive Check      [Day 1]
    NB 01: Data Collection & Curation      [Day 2]
    NB 02: Ligand Features & Splits        [Day 3]
    NB 03: SE(3) DataLoader & Training     [Day 3-4]
    NB 04: Candidate Generation (Hybrid)   [Day 4-5]
    NB 05: Docking & Pose Filtering        [Day 6-7]
    NB 06: ADMET, Synthesis & Safety       [Day 8]
    NB 07: Final Ranking & TPP Report      [Day 9-10]
    QS:    Quickstart (Full Pipeline)      [Any time]
    
    Run in order: 00 -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07
    Each notebook resumes from Drive checkpoints if disconnected.
    """
    print(toc)


def check_drive_artifacts(base_path):
    """Check which pipeline artifacts exist on Drive and report status."""
    artifacts = {
        "Config": "campaign_config.yaml",
        "Raw ChEMBL": "01_raw_data/chembl/jak2_activities_raw.csv",
        "PDB 5AEP": "01_raw_data/pdb/5AEP.pdb",
        "Curated data": "02_curated_data/jak2_curated_ligands.csv",
        "Features": "03_features/ligand_features.csv",
        "SE(3) graphs": "03_features/ligand_graphs/se3_graphs.pt",
        "SE(3) checkpoint": "04_models_checkpoints/se3_flow/se3_best_checkpoint.pt",
        "Training curve": "04_models_checkpoints/se3_flow/training_curve.png",
        "Merged candidates": "05_generated_candidates/merged/generated_merged_filtered.csv",
        "Docking scores": "06_docking/scores/docking_scores.csv",
        "ADMET scores": "07_admet_synthesis/admet_scores.csv",
        "Final ranking": "08_final_ranking/final_ranked_candidates.csv",
        "Molecule cards": "08_final_ranking/top10_molecule_cards.xlsx",
        "TPP report": "09_reports/EMD_V5_2_Hybrid_TPP.txt",
    }
    
    print("Drive Artifact Status:")
    print("-" * 50)
    found = 0
    for name, rel_path in artifacts.items():
        full = os.path.join(base_path, rel_path)
        exists = os.path.exists(full)
        if exists:
            size_kb = os.path.getsize(full) / 1024
            print(f"  [OK]   {name:25s} ({size_kb:.0f} KB)")
            found += 1
        else:
            print(f"  [----] {name}")
    
    print(f"\n  {found}/{len(artifacts)} artifacts found")
    return found
