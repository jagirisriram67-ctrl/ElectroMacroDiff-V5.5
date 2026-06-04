# Google Colab Training Workflow

Use Colab for real SE(3) training. Local CPU is enough for debug, but Colab GPU is better for 50-120 epoch training.

## What Colab Training Does

- Trains the SE(3) coordinate flow model on `03_features/ligand_graphs/se3_graphs.pt`.
- Saves checkpoints in `04_models_checkpoints/se3_flow`.
- Produces:
  - `se3_best_checkpoint.pt`
  - `se3_latest_checkpoint.pt`
  - `training_log.csv`
  - `training_curve.png`
  - `training_summary.json`

Important: in V5.2 Hybrid, this trained model is model evidence and a foundation for V5.3. The current final candidate generation still comes from SELFIES, RDKit, and macrocycle-linker generation. To make training directly improve molecules, the next upgrade is to connect trained SE(3) checkpoints to anchor/linker inference.

## Step 1 - Upload Project To Drive

1. Zip the whole local folder:

   `C:\Users\srira\Desktop\new_plan\EMD_V5_2_Hybrid`

2. Upload it to Google Drive:

   `MyDrive/EMD_V5_2_Hybrid.zip`

3. Open Google Colab and choose:

   `Runtime -> Change runtime type -> T4 GPU`

## Step 2 - Colab Setup Cell

Run this first cell in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile, pathlib, shutil

ZIP_PATH = "/content/drive/MyDrive/EMD_V5_2_Hybrid.zip"
PROJECT_DIR = "/content/drive/MyDrive/EMD_V5_2_Hybrid"

if not os.path.exists(PROJECT_DIR):
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall("/content/drive/MyDrive")

os.chdir(PROJECT_DIR)
print("Project:", os.getcwd())
!ls
```

If your zip extracts into a nested folder like `EMD_V5_2_Hybrid/EMD_V5_2_Hybrid`, move into the inner folder before continuing.

## Step 3 - Install Dependencies

Run:

```python
!python -m pip install -q --upgrade pip
!python -m pip install -q pandas numpy matplotlib pyyaml requests selfies rdkit torch
```

Then verify:

```python
import torch, rdkit, pandas, selfies
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

If CUDA is false, stop and set Colab runtime to GPU.

## Step 4 - Quick Smoke Training

Run a small 2-epoch training first:

```python
!python scripts/03_train_se3_main.py \
  --base . \
  --epochs 2 \
  --batch-size 8 \
  --hidden-dim 128 \
  --num-layers 4 \
  --checkpoint-every 1 \
  --device cuda
```

Check outputs:

```python
!ls -lh 04_models_checkpoints/se3_flow
!tail -n 5 04_models_checkpoints/se3_flow/training_log.csv
```

## Step 5 - Main Training

If smoke training works, run:

```python
!python scripts/03_train_se3_main.py \
  --base . \
  --epochs 50 \
  --batch-size 8 \
  --hidden-dim 128 \
  --num-layers 4 \
  --learning-rate 0.0002 \
  --checkpoint-every 10 \
  --device cuda
```

If Colab disconnects or is slow, use this lighter run:

```python
!python scripts/03_train_se3_main.py \
  --base . \
  --epochs 25 \
  --batch-size 4 \
  --hidden-dim 96 \
  --num-layers 3 \
  --learning-rate 0.0002 \
  --checkpoint-every 5 \
  --device cuda
```

Stretch run for stronger evidence:

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

## Step 6 - Inspect Training

Run:

```python
import pandas as pd
log = pd.read_csv("04_models_checkpoints/se3_flow/training_log.csv")
display(log.tail())
print("Best val loss:", log["val_loss"].min())
```

Show the curve:

```python
from IPython.display import Image, display
display(Image("04_models_checkpoints/se3_flow/training_curve.png"))
```

A useful run should show validation loss moving downward or stabilizing. If validation loss explodes, reduce learning rate to `0.0001`.

## Step 7 - Package Trained Artifacts

Run:

```python
import zipfile, os

ARTIFACT_ZIP = "/content/drive/MyDrive/EMD_V5_2_Hybrid_trained_artifacts.zip"
files = [
    "04_models_checkpoints/se3_flow/se3_best_checkpoint.pt",
    "04_models_checkpoints/se3_flow/se3_latest_checkpoint.pt",
    "04_models_checkpoints/se3_flow/training_log.csv",
    "04_models_checkpoints/se3_flow/training_curve.png",
    "04_models_checkpoints/se3_flow/training_summary.json",
    "00_project_registry/run_registry.csv",
    "00_project_registry/artifact_registry.csv",
    "00_project_registry/progress_m3_se3_training.json",
]

with zipfile.ZipFile(ARTIFACT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in files:
        if os.path.exists(path):
            z.write(path, path)

print("Saved:", ARTIFACT_ZIP)
!ls -lh /content/drive/MyDrive/EMD_V5_2_Hybrid_trained_artifacts.zip
```

## Step 8 - Bring Artifacts Back Here

Download this file from Drive:

`EMD_V5_2_Hybrid_trained_artifacts.zip`

Put it in:

`C:\Users\srira\Desktop\new_plan`

Then ask Codex:

`I uploaded EMD_V5_2_Hybrid_trained_artifacts.zip. Please merge the trained Colab checkpoints into the project, validate them, and plan V5.3 model-guided macrocycle generation.`

## Step 9 - What To Do After Training

After Colab training is back in the project:

1. Validate checkpoint/log integrity.
2. Compare smoke, 50-epoch, and stretch training curves.
3. Build V5.3 data:
   - Fragment the 97 real JAK2 macrocycles.
   - Create acyclic core plus linker pairs.
   - Learn anchor-site priors.
4. Connect trained SE(3) geometry to macrocycle linker inference.
5. Generate a second macrocycle batch biased by:
   - parent JAK2 pActivity,
   - 12-16 atom ring size,
   - lower SA score,
   - docking pose sanity,
   - hinge-region pharmacophore retention.
