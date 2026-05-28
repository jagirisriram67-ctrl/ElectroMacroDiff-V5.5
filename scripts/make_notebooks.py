from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def notebook(
    slug: str,
    title: str,
    purpose: str,
    main_code: str,
    validation_code: str,
    summary_code: str,
) -> dict:
    progress_code = f"""
# Resume/progress cell
NOTEBOOK_SLUG = "{slug}"
PROGRESS_PATH = BASE_PATH / "00_project_registry" / f"progress_{{NOTEBOOK_SLUG}}.json"
save_progress(PROGRESS_PATH, {{"last_completed_index": -1, "status": "started"}})
print(f"Progress file: {{PROGRESS_PATH}}")
"""
    cells = [
        markdown_cell(f"# {title}\n\n{purpose}"),
        code_cell(
            """
# Install cell
from google.colab import drive
drive.mount("/content/drive")

BASE = "/content/drive/MyDrive/EMD_V5_2_Hybrid"
REQ = f"{BASE}/requirements_v5_2_hybrid_colab.txt"
!pip -q install -r "{REQ}"
"""
        ),
        code_cell(
            """
# Config cell
import os
import sys
from pathlib import Path

BASE_PATH = Path(BASE)
sys.path.insert(0, str(BASE_PATH / "src"))

from emd_v5_2_hybrid.registry import ensure_project_tree, register_run, save_progress

ensure_project_tree(BASE_PATH)
print(f"Project base: {BASE_PATH}")
"""
        ),
        code_cell(
            """
# Tiny Debug mode
TINY_DEBUG = True
TINY_SIZE = 25
RANDOM_SEED = 42
print({"tiny_debug": TINY_DEBUG, "tiny_size": TINY_SIZE, "seed": RANDOM_SEED})
"""
        ),
        code_cell(progress_code),
        code_cell(main_code),
        code_cell(validation_code),
        code_cell(summary_code),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "00_environment_and_drive_check.ipynb": notebook(
        "00_environment_and_drive_check",
        "00 Environment And Drive Check",
        "Validate the Drive folder, requirements, package versions, and registry.",
        """
from emd_v5_2_hybrid.registry import write_environment_report, register_artifact

env_path = write_environment_report(BASE_PATH)
register_artifact(BASE_PATH, "M0_registry", env_path, "environment_report", owner="Student 1")
register_run(BASE_PATH, "M0_registry", "completed", output_path=str(env_path), notes="Notebook environment check")
print(env_path.read_text()[:2000])
""",
        """
required = [
    "00_project_registry/run_registry.csv",
    "00_project_registry/artifact_registry.csv",
    "00_project_registry/environment_versions.txt",
]
for rel in required:
    path = BASE_PATH / rel
    assert path.exists(), f"Missing {path}"
print("Environment validation passed.")
""",
        """
print("Summary: Drive mounted, tree validated, environment report saved.")
""",
    ),
    "01_data_collection_and_curation.ipynb": notebook(
        "01_data_collection_and_curation",
        "01 Data Collection And Curation",
        "Collect ChEMBL JAK2 activities, curate ligand records, and download PDB 5AEP.",
        """
script = BASE_PATH / "scripts" / "01_collect_data.py"
flag = "--tiny-debug" if TINY_DEBUG else ""
!python "{script}" --base "{BASE}" --limit 1000 {flag}
""",
        """
import pandas as pd
curated = BASE_PATH / "02_curated_data" / "jak2_curated_ligands.csv"
pdb = BASE_PATH / "01_raw_data" / "pdb" / "5AEP.pdb"
df = pd.read_csv(curated)
assert len(df) >= (5 if TINY_DEBUG else 100), len(df)
assert pdb.exists() and pdb.stat().st_size > 1000
display(df.head())
print("Data validation passed.")
""",
        """
print("Summary: curated ligand table and receptor PDB are ready for feature generation.")
""",
    ),
    "02_ligand_features_and_splits.ipynb": notebook(
        "02_ligand_features_and_splits",
        "02 Ligand Features And Splits",
        "Compute descriptors, conformers, graph tensors, and split files.",
        """
script = BASE_PATH / "scripts" / "02_build_features.py"
flag = "--tiny-debug" if TINY_DEBUG else ""
!python "{script}" --base "{BASE}" {flag}
""",
        """
import pandas as pd
features = BASE_PATH / "03_features" / "ligand_features.csv"
graphs = BASE_PATH / "03_features" / "ligand_graphs" / "se3_graphs.pt"
df = pd.read_csv(features)
assert len(df) >= (5 if TINY_DEBUG else 100), len(df)
assert graphs.exists() and graphs.stat().st_size > 0
display(df.head())
print("Feature validation passed.")
""",
        """
print("Summary: descriptors, split IDs, and SE(3) graph tensors are saved.")
""",
    ),
    "03_se3_dataloader_training.ipynb": notebook(
        "03_se3_dataloader_training",
        "03 SE3 Dataloader Training",
        "Run the make-or-break debug gate: batch, forward pass, loss, backward pass, optimizer step, save/reload checkpoint.",
        """
script = BASE_PATH / "scripts" / "03_debug_se3_training.py"
!python "{script}" --base "{BASE}" --hidden-dim 64
""",
        """
import json
result_path = BASE_PATH / "04_models_checkpoints" / "se3_flow" / "se3_dataloader_test_passed.json"
checkpoint = BASE_PATH / "04_models_checkpoints" / "se3_flow" / "se3_latest_checkpoint.pt"
result = json.loads(result_path.read_text())
assert result["status"] == "passed", result
assert checkpoint.exists() and checkpoint.stat().st_size > 0
print(result)
print("SE(3) debug validation passed.")
""",
        """
print("Summary: SE(3) DataLoader/training gate passed. Continue to longer training if time allows.")
""",
    ),
    "04_candidate_generation_hybrid.ipynb": notebook(
        "04_candidate_generation_hybrid",
        "04 Candidate Generation Hybrid",
        "Generate baseline RDKit/SELFIES candidates and reserve SE(3) output slots for model inference.",
        """
script = BASE_PATH / "scripts" / "04_generate_baselines.py"
!python "{script}" --base "{BASE}" --seed-count 50 --selfies-per-seed 10 --rdkit-per-seed 5
""",
        """
import pandas as pd
merged = BASE_PATH / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
df = pd.read_csv(merged)
assert len(df) >= (5 if TINY_DEBUG else 50), len(df)
assert df["inchikey"].is_unique
display(df.head())
print("Generation validation passed.")
""",
        """
print("Summary: baseline candidates are generated, filtered, deduplicated, and ready for docking prep.")
""",
    ),
    "05_docking_and_pose_filtering.ipynb": notebook(
        "05_docking_and_pose_filtering",
        "05 Docking And Pose Filtering",
        "Infer the binding grid from PDB 5AEP, prepare top candidates as 3D SDF, then create a Vina manifest once PDBQT files exist.",
        """
# Step 1: infer grid from the co-crystallized QUP ligand and prepare candidate SDF files.
prep_script = BASE_PATH / "scripts" / "06_prepare_docking_inputs.py"
!python "{prep_script}" --base "{BASE}" --top-n 150

# Step 2: prepare receptor/ligand PDBQT files and build a Vina manifest.
pdbqt_script = BASE_PATH / "scripts" / "06_prepare_pdbqt.py"
manifest_script = BASE_PATH / "scripts" / "06_make_vina_manifest.py"
!python "{pdbqt_script}" --base "{BASE}"
!python "{manifest_script}" --base "{BASE}"

# Step 3: run Vina where the vina executable is available, then parse logs.
# run_script = BASE_PATH / "scripts" / "06_run_vina_manifest.py"
# parse_script = BASE_PATH / "scripts" / "06_parse_vina_results.py"
# !python "{run_script}" --base "{BASE}"
# !python "{parse_script}" --base "{BASE}"
""",
        """
grid = BASE_PATH / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json"
sdf = BASE_PATH / "06_docking" / "ligands_sdf" / "candidates_for_docking.sdf"
receptor = BASE_PATH / "06_docking" / "receptor" / "jak2_prepared.pdbqt"
manifest = BASE_PATH / "06_docking" / "scores" / "vina_command_manifest.csv"
assert grid.exists() and grid.stat().st_size > 0
assert sdf.exists() and sdf.stat().st_size > 0
assert receptor.exists() and receptor.stat().st_size > 0
assert manifest.exists() and manifest.stat().st_size > 0
print(grid.read_text())
print("Docking prep validation passed. Vina scores remain the next gate.")
""",
        """
print("Summary: docking grid and ligand SDF files are ready. Vina scores remain the next gate.")
""",
    ),
    "06_admet_synthesis_safety.ipynb": notebook(
        "06_admet_synthesis_safety",
        "06 ADMET Synthesis Safety",
        "Score generated candidates with descriptor-based ADMET and synthesis proxies.",
        """
script = BASE_PATH / "scripts" / "05_score_admet_synthesis.py"
!python "{script}" --base "{BASE}"
""",
        """
import pandas as pd
admet = BASE_PATH / "07_admet_synthesis" / "admet_scores.csv"
df = pd.read_csv(admet)
assert len(df) > 0
display(df.head())
print("ADMET validation passed.")
""",
        """
print("Summary: ADMET, synthesis, and safety proxy files are ready for consensus ranking.")
""",
    ),
    "07_final_ranking_and_tpp.ipynb": notebook(
        "07_final_ranking_and_tpp",
        "07 Final Ranking And TPP",
        "Merge generation, docking, ADMET, synthesis, novelty, and safety proxy signals into final candidate ranking.",
        """
script = BASE_PATH / "scripts" / "07_rank_candidates.py"
!python "{script}" --base "{BASE}"
""",
        """
import pandas as pd
ranking = BASE_PATH / "08_final_ranking" / "final_ranked_candidates.csv"
df = pd.read_csv(ranking)
assert len(df) > 0
assert "final_weighted_score" in df.columns
display(df.head(10))
print("Ranking validation passed.")
""",
        """
print("Summary: final ranking is complete. Use the top 3-5 for TPP molecule cards and final report.")
""",
    ),
}


def main() -> None:
    out_dir = ROOT / "10_notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        path = out_dir / name
        path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
