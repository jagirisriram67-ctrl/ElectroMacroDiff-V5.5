# V5.3 Colab Benchmark Plan

This is the next training path. It is larger than the V5.2 SE(3) run and closer to MED because it adds macrocycle fragmentation, anchor-site learning, and linker-size supervision.

## Important Benchmark Truth

Do not claim EMD crosses MED accuracy until it is evaluated on the same benchmark style.

MED reports metrics such as validity, uniqueness, macrocyclization percentage, and linker novelty on large fragment-linker data. The current EMD project is JAK2-focused and low-resource. The correct claim is:

> EMD V5.3 is being upgraded toward MED-style macrocycle generation with JAK2-specific fragment-linker training, macrocycle-only docking, and pose sanity validation.

The target metrics for our project are:

- Valid RDKit molecules: aim `>90%`.
- Unique molecules: aim `>98%`.
- Macrocycle fraction in macrocycle generation mode: aim `>90%`.
- Pose sanity pass among top 10: aim `10 / 10`.
- Docking coverage: aim `100+` parsed Vina scores per campaign.
- Linker/anchor model quality: improve validation/test F1 across V5.3 iterations.

## Current V5.3 Baseline

Already built locally:

- Fragment-linker pairs: `2,714`.
- Real JAK2 macrocycles used: `97`.
- Anchor atom rows: `87,760`.
- Positive anchor labels: `5,428`.
- Baseline anchor model: `04_models_checkpoints/v5_3_anchor/anchor_site_model.pt`.
- Baseline test F1: about `0.207`.

This is useful, but not final. The model is still feature-light and imbalanced.

## Colab Step 1 - Upload Updated Project

Zip:

`C:\Users\srira\Desktop\new_plan\EMD_V5_2_Hybrid`

Upload as:

`MyDrive/EMD_V5_2_Hybrid.zip`

Set Colab:

`Runtime -> Change runtime type -> T4 GPU`

## Colab Step 2 - Mount And Extract

```python
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile

ZIP_PATH = "/content/drive/MyDrive/EMD_V5_2_Hybrid.zip"
PROJECT_DIR = "/content/drive/MyDrive/EMD_V5_2_Hybrid"

if not os.path.exists(PROJECT_DIR):
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall("/content/drive/MyDrive")

os.chdir(PROJECT_DIR)
print("Project:", os.getcwd())
```

## Colab Step 3 - Install

```python
!python -m pip install -q --upgrade pip
!python -m pip install -q pandas numpy matplotlib pyyaml requests selfies rdkit torch
```

Verify GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Colab Step 4 - Rebuild V5.3 Dataset

```python
!python scripts/13_build_v5_3_fragment_dataset.py \
  --base . \
  --max-pairs-per-molecule 40
```

Expected output should be roughly thousands of fragment-linker rows and tens of thousands of atom rows.

## Colab Step 5 - Train Anchor Model Longer

Main V5.3 anchor training:

```python
!python scripts/14_train_v5_3_anchor_model.py \
  --base . \
  --epochs 1000 \
  --hidden-dim 192 \
  --learning-rate 0.0005 \
  --device cuda
```

If Colab disconnects, use:

```python
!python scripts/14_train_v5_3_anchor_model.py \
  --base . \
  --epochs 500 \
  --hidden-dim 128 \
  --learning-rate 0.0007 \
  --device cuda
```

## Colab Step 6 - Train Linker-Size Model

Main V5.3 linker-size training:

```python
!python scripts/15_train_v5_3_linker_size_model.py \
  --base . \
  --epochs 1000 \
  --hidden-dim 192 \
  --learning-rate 0.0005 \
  --device cuda
```

If Colab is unstable, use:

```python
!python scripts/15_train_v5_3_linker_size_model.py \
  --base . \
  --epochs 500 \
  --hidden-dim 128 \
  --learning-rate 0.0007 \
  --device cuda
```

Track `within_one_accuracy`; this model is useful if it predicts linker size within one heavy atom.

## Colab Step 7 - Train SE(3) Longer

Main longer SE(3) run:

```python
!python scripts/03_train_se3_main.py \
  --base . \
  --epochs 120 \
  --batch-size 8 \
  --hidden-dim 160 \
  --num-layers 5 \
  --learning-rate 0.00015 \
  --checkpoint-every 10 \
  --device cuda
```

Stretch run:

```python
!python scripts/03_train_se3_main.py \
  --base . \
  --epochs 300 \
  --batch-size 8 \
  --hidden-dim 192 \
  --num-layers 6 \
  --learning-rate 0.0001 \
  --checkpoint-every 10 \
  --device cuda
```

Free-tier Colab may disconnect. A completed 120-epoch run is more useful than a failed 300-epoch run.

## Colab Step 8 - Inspect Metrics

```python
import pandas as pd

se3 = pd.read_csv("04_models_checkpoints/se3_flow/training_log.csv")
anchor = pd.read_csv("04_models_checkpoints/v5_3_anchor/anchor_training_log.csv")
linker = pd.read_csv("04_models_checkpoints/v5_3_linker_size/linker_size_training_log.csv")

print("SE3 best val:", se3["val_loss"].min())
print("Anchor best val F1:", anchor["val_f1"].max())
print("Linker best within-one:", linker["val_within_one_accuracy"].max())
display(se3.tail())
display(anchor.tail())
display(linker.tail())
```

Also inspect:

```python
!cat 04_models_checkpoints/se3_flow/training_summary.json
!cat 04_models_checkpoints/v5_3_anchor/anchor_training_summary.json
!cat 04_models_checkpoints/v5_3_linker_size/linker_size_training_summary.json
```

## Colab Step 9 - Package V5.3 Artifacts

```python
import zipfile, os

ARTIFACT_ZIP = "/content/drive/MyDrive/EMD_V5_3_trained_artifacts.zip"
files = [
    "02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv",
    "03_features/v5_3_anchor_atom_training.csv",
    "04_models_checkpoints/se3_flow/se3_best_checkpoint.pt",
    "04_models_checkpoints/se3_flow/se3_latest_checkpoint.pt",
    "04_models_checkpoints/se3_flow/training_log.csv",
    "04_models_checkpoints/se3_flow/training_curve.png",
    "04_models_checkpoints/se3_flow/training_summary.json",
    "04_models_checkpoints/v5_3_anchor/anchor_site_model.pt",
    "04_models_checkpoints/v5_3_anchor/anchor_training_log.csv",
    "04_models_checkpoints/v5_3_anchor/anchor_training_summary.json",
    "04_models_checkpoints/v5_3_linker_size/linker_size_model.pt",
    "04_models_checkpoints/v5_3_linker_size/linker_size_training_features.csv",
    "04_models_checkpoints/v5_3_linker_size/linker_size_training_log.csv",
    "04_models_checkpoints/v5_3_linker_size/linker_size_training_summary.json",
    "00_project_registry/run_registry.csv",
    "00_project_registry/artifact_registry.csv",
    "00_project_registry/progress_v5_3_fragment_dataset.json",
    "00_project_registry/progress_v5_3_anchor_training.json",
    "00_project_registry/progress_v5_3_linker_size_training.json",
]

with zipfile.ZipFile(ARTIFACT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in files:
        if os.path.exists(path):
            z.write(path, path)

print("Saved:", ARTIFACT_ZIP)
!ls -lh "$ARTIFACT_ZIP"
```

## Bring It Back

Download:

`EMD_V5_3_trained_artifacts.zip`

Place it in:

`C:\Users\srira\Desktop\new_plan`

Then ask:

`I uploaded EMD_V5_3_trained_artifacts.zip. Merge it and run V5.3 model-guided macrocycle generation.`

## What We Build After This

After V5.3 training returns:

1. Use anchor model to score attachment sites on high-potency JAK2 seeds.
2. Use fragment-linker statistics to choose linker length and chemistry.
3. Use trained SE(3) checkpoint to guide 3D geometry.
4. Generate a new macrocycle-only batch.
5. Dock at least 150 candidates.
6. Compare:
   - V5.2 macrocycle candidates,
   - V5.3 model-guided macrocycles,
   - MED paper reported metrics where comparable.
