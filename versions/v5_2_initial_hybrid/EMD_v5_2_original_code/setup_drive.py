"""
EMD V5.2 Hybrid — Google Drive Setup Script
============================================
Creates the complete Drive folder structure and copies
project files for use in Google Colab.

Run this INSIDE a Colab notebook to set up Drive:

    %run setup_drive.py

Or import and call:

    from setup_drive import setup_drive_project
    setup_drive_project()
"""

import os
import shutil


def setup_drive_project(
    drive_base: str = "/content/drive/MyDrive/EMD_V5_2_Hybrid",
    local_project: str = None,
):
    """Create complete Drive project structure and copy files.
    
    Args:
        drive_base: Google Drive base directory path.
        local_project: Local project directory to copy files from.
                       If None, copies only from current directory.
    """
    # Create all directories
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
    
    print(f"Creating project structure at: {drive_base}")
    for d in dirs:
        os.makedirs(os.path.join(drive_base, d), exist_ok=True)
    print(f"  ✅ Created {len(dirs)} directories")
    
    # Files to copy to Drive root
    root_files = [
        "campaign_config.yaml",
        "requirements_v5_2_hybrid_colab.txt",
        "EMD_V5_2_Hybrid_Master_Plan.md",
    ]
    
    # Files to copy to specific directories
    targeted_copies = {
        "00_project_registry": [
            "campaign_config.yaml",
            "requirements_v5_2_hybrid_colab.txt",
        ],
    }
    
    # Copy pipeline modules
    pipeline_files = [
        "emd_pipeline/__init__.py",
        "emd_pipeline/config_registry.py",
        "emd_pipeline/data_collection.py",
        "emd_pipeline/feature_engineering.py",
        "emd_pipeline/se3_flow_model.py",
        "emd_pipeline/candidate_generation.py",
        "emd_pipeline/docking.py",
        "emd_pipeline/admet_synthesis.py",
        "emd_pipeline/ranking_report.py",
    ]
    
    source_dir = local_project or "."
    
    # Copy root files
    for fname in root_files:
        src = os.path.join(source_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(drive_base, fname)
            shutil.copy2(src, dst)
            print(f"  📄 {fname} -> Drive root")
    
    # Copy targeted files
    for target_dir, files in targeted_copies.items():
        for fname in files:
            src = os.path.join(source_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(drive_base, target_dir, fname)
                shutil.copy2(src, dst)
                print(f"  📄 {fname} -> {target_dir}/")
    
    # Copy pipeline modules
    pipeline_dest = os.path.join(drive_base, "emd_pipeline")
    os.makedirs(pipeline_dest, exist_ok=True)
    
    for pfile in pipeline_files:
        src = os.path.join(source_dir, pfile)
        if os.path.exists(src):
            dst = os.path.join(drive_base, pfile)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  🐍 {pfile}")
    
    # Copy notebooks (both .py and .ipynb)
    nb_dir = os.path.join(source_dir, "notebooks")
    if os.path.exists(nb_dir):
        nb_dest = os.path.join(drive_base, "10_notebooks")
        for fname in os.listdir(nb_dir):
            if fname.endswith((".py", ".ipynb")):
                src = os.path.join(nb_dir, fname)
                dst = os.path.join(nb_dest, fname)
                shutil.copy2(src, dst)
                print(f"  📓 {fname}")
    
    # Create initial daily log
    log_path = os.path.join(drive_base, "11_logs", "daily_log.md")
    if not os.path.exists(log_path):
        import time
        with open(log_path, "w") as f:
            f.write("# ElectroMacroDiff V5.2 Hybrid — Daily Log\n\n")
            f.write(f"## Setup — {time.strftime('%Y-%m-%d')}\n")
            f.write("- Drive project structure created\n")
            f.write("- Files copied from local workspace\n")
        print(f"  📝 Daily log initialized")
    
    # Create AI assistance notes
    ai_notes_path = os.path.join(drive_base, "11_logs", "ai_assistance_notes.md")
    if not os.path.exists(ai_notes_path):
        with open(ai_notes_path, "w") as f:
            f.write("# AI Assistance Notes\n\n")
            f.write("Log prompts and useful AI-generated insights here.\n\n")
        print(f"  📝 AI assistance notes initialized")
    
    print(f"\n✅ Drive project setup complete!")
    print(f"   Base path: {drive_base}")
    print(f"\n   Next step: Open 00_environment_and_drive_check.ipynb in Colab")
    
    return drive_base


if __name__ == "__main__":
    # When run directly, try to set up Drive
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        setup_drive_project()
    except ImportError:
        print("Not in Colab — running local setup")
        local_base = os.path.dirname(os.path.abspath(__file__))
        setup_drive_project(
            drive_base=os.path.join(local_base, "EMD_V5_2_Hybrid_Local"),
            local_project=local_base,
        )
