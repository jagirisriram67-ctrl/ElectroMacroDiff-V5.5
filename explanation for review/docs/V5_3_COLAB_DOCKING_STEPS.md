# V5.3 Clean Colab Docking Steps

Use this after the conflicted local partial docking outputs have been deleted.

Important: standard AutoDock Vina is CPU-based, not GPU-based. Colab GPU helps model training, but Vina itself runs on CPU. Colab can still be faster or more stable than the local laptop workflow, and this clean manifest prevents the overlap/conflict problem.

## Local Upload

Zip this cleaned folder:

`C:\Users\srira\Desktop\new_plan\EMD_V5_2_Hybrid`

Upload to Google Drive as:

`MyDrive/EMD_V5_2_Hybrid_clean_colab_docking.zip`

## Colab Cell 1 - Mount And Extract

```python
from google.colab import drive
drive.mount("/content/drive")

import os, shutil, zipfile

ZIP_PATH = "/content/drive/MyDrive/EMD_V5_2_Hybrid_clean_colab_docking.zip"
PROJECT_DIR = "/content/EMD_V5_2_Hybrid"

if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)

os.makedirs(PROJECT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(PROJECT_DIR)

os.chdir(PROJECT_DIR)
print("Project:", os.getcwd())
!ls
```

## Colab Cell 2 - Install Dependencies

```python
!python -m pip install -q --upgrade pip
!python -m pip install -q pandas numpy pyyaml requests rdkit meeko gemmi
```

## Colab Cell 3 - Download Linux Vina

```python
!python scripts/06_download_vina.py --base .
!chmod +x tools/vina/vina
!tools/vina/vina --help | head
```

## Colab Cell 4 - Confirm Inputs

```python
import os, glob, pandas as pd

candidates = pd.read_csv("05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv")
ligands = glob.glob("06_docking/v5_3_model_guided/ligands_pdbqt/CAND_*.pdbqt")

print("V5.3 candidates:", len(candidates))
print("Prepared ligand PDBQT:", len(ligands))
print("Receptor exists:", os.path.exists("06_docking/receptor/jak2_prepared.pdbqt"))
print("Grid exists:", os.path.exists("06_docking/v5_3_model_guided/receptor/docking_grid_5AEP_QUP.json"))
```

Expected:

- `V5.3 candidates: 114`
- `Prepared ligand PDBQT: 114`
- receptor and grid both `True`

## Colab Cell 5 - Rebuild A Clean Linux Manifest

```python
!rm -f 06_docking/v5_3_model_guided/scores/vina_command_manifest.csv
!rm -f 06_docking/v5_3_model_guided/scores/CAND_*_vina.log
!rm -f 06_docking/v5_3_model_guided/poses/CAND_*_vina_out.pdbqt

!python scripts/06_make_vina_manifest.py \
  --base . \
  --grid-json 06_docking/v5_3_model_guided/receptor/docking_grid_5AEP_QUP.json \
  --ligand-pdbqt-dir 06_docking/v5_3_model_guided/ligands_pdbqt \
  --receptor-pdbqt 06_docking/receptor/jak2_prepared.pdbqt \
  --manifest-output 06_docking/v5_3_model_guided/scores/vina_command_manifest.csv \
  --pose-dir 06_docking/v5_3_model_guided/poses \
  --log-dir 06_docking/v5_3_model_guided/scores \
  --exhaustiveness 16 \
  --stage V5_3_model_guided_colab_manifest

import pandas as pd
manifest = pd.read_csv("06_docking/v5_3_model_guided/scores/vina_command_manifest.csv")
print(manifest["status"].value_counts(dropna=False))
display(manifest.head())
```

## Colab Cell 6 - Run Vina In Resume-Safe Chunks

Run this cell. If Colab disconnects, reconnect and run it again; completed rows stay completed.

```python
import pandas as pd, time

MANIFEST = "06_docking/v5_3_model_guided/scores/vina_command_manifest.csv"

for chunk in range(20):
    print(f"\n=== Vina chunk {chunk + 1} ===")
    !python scripts/06_run_vina_manifest.py \
      --base . \
      --manifest 06_docking/v5_3_model_guided/scores/vina_command_manifest.csv \
      --limit 10 \
      --vina-exe tools/vina/vina \
      --stage V5_3_model_guided_colab_vina_run

    manifest = pd.read_csv(MANIFEST)
    counts = manifest["status"].value_counts(dropna=False)
    print(counts)
    if int((manifest["status"] == "completed").sum()) == len(manifest):
        print("All V5.3 docking jobs completed.")
        break
    time.sleep(2)
```

## Colab Cell 7 - Parse, Rank, Pose-Check, Benchmark

```python
!python scripts/06_parse_vina_results.py \
  --base . \
  --manifest 06_docking/v5_3_model_guided/scores/vina_command_manifest.csv \
  --candidates-csv 05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv \
  --grid-json 06_docking/v5_3_model_guided/receptor/docking_grid_5AEP_QUP.json \
  --output-csv 06_docking/v5_3_model_guided/scores/docking_scores_full_exh16.csv \
  --stage V5_3_model_guided_colab_docking_parse

!python scripts/05_score_admet_synthesis.py \
  --base . \
  --candidates-csv 05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv \
  --output-csv 07_admet_synthesis/v5_3_model_guided_admet_scores.csv \
  --flags-csv 07_admet_synthesis/v5_3_model_guided_filter_flags.csv \
  --notes-csv 07_admet_synthesis/v5_3_model_guided_safety_proxy_notes.csv \
  --stage V5_3_model_guided_colab_admet

!python scripts/07_rank_candidates.py \
  --base . \
  --candidates-csv 05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv \
  --docking-csv 06_docking/v5_3_model_guided/scores/docking_scores_full_exh16.csv \
  --admet-csv 07_admet_synthesis/v5_3_model_guided_admet_scores.csv \
  --output-csv 08_final_ranking/v5_3_model_guided_ranked_candidates.csv \
  --stage V5_3_model_guided_colab_ranking

!python scripts/06_pose_sanity.py \
  --base . \
  --top-n 10 \
  --ranking-csv 08_final_ranking/v5_3_model_guided_ranked_candidates.csv \
  --receptor-pdb 06_docking/receptor/5AEP_receptor_clean.pdb \
  --grid-json 06_docking/v5_3_model_guided/receptor/docking_grid_5AEP_QUP.json \
  --pose-dir 06_docking/v5_3_model_guided/poses \
  --output-csv 06_docking/v5_3_model_guided/scores/pose_sanity_scores.csv \
  --image-dir 06_docking/v5_3_model_guided/images/pose_sanity \
  --stage V5_3_model_guided_colab_pose_sanity

!python scripts/07_rank_candidates.py \
  --base . \
  --candidates-csv 05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv \
  --docking-csv 06_docking/v5_3_model_guided/scores/docking_scores_full_exh16.csv \
  --pose-sanity-csv 06_docking/v5_3_model_guided/scores/pose_sanity_scores.csv \
  --admet-csv 07_admet_synthesis/v5_3_model_guided_admet_scores.csv \
  --output-csv 08_final_ranking/v5_3_model_guided_ranked_candidates.csv \
  --stage V5_3_model_guided_colab_ranking_with_pose

!python scripts/18_build_v5_3_benchmark_report.py --base .
```

## Colab Cell 8 - Quick Result Check

```python
import pandas as pd

docking = pd.read_csv("06_docking/v5_3_model_guided/scores/docking_scores_full_exh16.csv")
docking["best_score"] = pd.to_numeric(docking["best_score"], errors="coerce")
print("Parsed scores:", docking["best_score"].notna().sum(), "/", len(docking))
print("Best:", docking["best_score"].min())
print("Median:", docking["best_score"].median())

ranked = pd.read_csv("08_final_ranking/v5_3_model_guided_ranked_candidates.csv")
display(ranked[["rank", "candidate_id", "best_score", "pose_decision", "final_weighted_score", "decision"]].head(10))

pose = pd.read_csv("06_docking/v5_3_model_guided/scores/pose_sanity_scores.csv")
display(pose[["candidate_id", "pose_decision", "contacts_within_4A", "hard_clashes_lt_1_8A"]])
```

## Colab Cell 9 - Package Results Back To Drive

```python
import os, zipfile

ARTIFACT_ZIP = "/content/drive/MyDrive/EMD_V5_3_colab_docking_results.zip"
files = [
    "06_docking/v5_3_model_guided/scores/vina_command_manifest.csv",
    "06_docking/v5_3_model_guided/scores/docking_scores_full_exh16.csv",
    "06_docking/v5_3_model_guided/scores/pose_sanity_scores.csv",
    "08_final_ranking/v5_3_model_guided_ranked_candidates.csv",
    "07_admet_synthesis/v5_3_model_guided_admet_scores.csv",
    "07_admet_synthesis/v5_3_model_guided_filter_flags.csv",
    "07_admet_synthesis/v5_3_model_guided_safety_proxy_notes.csv",
    "09_reports/v5_3_benchmark/EMD_V5_3_Benchmark_Report.md",
    "09_reports/v5_3_benchmark/emd_candidate_benchmark_metrics.csv",
    "09_reports/v5_3_benchmark/emd_docking_benchmark_metrics.csv",
    "09_reports/v5_3_benchmark/emd_pose_benchmark_metrics.csv",
    "09_reports/v5_3_benchmark/emd_v5_3_benchmark_summary.json",
]

with zipfile.ZipFile(ARTIFACT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in files:
        if os.path.exists(path):
            z.write(path, path)
    for folder in [
        "06_docking/v5_3_model_guided/poses",
        "06_docking/v5_3_model_guided/images/pose_sanity",
    ]:
        if os.path.exists(folder):
            for root, _dirs, names in os.walk(folder):
                for name in names:
                    path = os.path.join(root, name)
                    z.write(path, path)

print("Saved:", ARTIFACT_ZIP)
!ls -lh "$ARTIFACT_ZIP"
```

Download `EMD_V5_3_colab_docking_results.zip`, put it in:

`C:\Users\srira\Desktop\new_plan`

Then tell Codex:

`I uploaded EMD_V5_3_colab_docking_results.zip. Merge the Colab docking results and update the benchmark report.`
