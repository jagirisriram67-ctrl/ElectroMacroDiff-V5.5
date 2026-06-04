# Next Colab Steps: More Training And Testing Data

This is the exact next workflow. It increases training data by adding broad ChEMBL macrocycles for pretraining, then fine-tunes on the JAK2 macrocycle dataset.

## Goal

Train with more data in this order:

1. Broad macrocycle pretraining data from ChEMBL.
2. JAK2 macrocycle fine-tuning data.
3. Three model tracks:
   - anchor-site model,
   - linker-size model,
   - SE(3) geometry model.

This is the right path for better accuracy. More random molecules are not enough; the project needs more **macrocycle fragment-linker pairs**.

## Step 1 - Zip And Upload

Zip:

`C:\Users\srira\Desktop\new_plan\EMD_V5_2_Hybrid`

Upload to Google Drive as:

`MyDrive/EMD_V5_2_Hybrid.zip`

Open Google Colab:

`Runtime -> Change runtime type -> T4 GPU`

## Step 2 - Mount Drive And Enter Project

Run this cell:

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
!ls
```

## Step 3 - Install Dependencies

```python
!python -m pip install -q --upgrade pip
!python -m pip install -q pandas numpy matplotlib pyyaml requests selfies rdkit torch
```

Verify GPU:

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

If CUDA is false, change runtime to GPU before training.

## Step 4 - Collect Bigger Macrocycle Pretraining Data

Main run:

```python
!python scripts/16_collect_macrocycle_pretraining_data.py \
  --base . \
  --scan-limit 100000 \
  --page-size 1000 \
  --max-macrocycles 5000
```

If this is too slow on free Colab, use:

```python
!python scripts/16_collect_macrocycle_pretraining_data.py \
  --base . \
  --scan-limit 30000 \
  --page-size 1000 \
  --max-macrocycles 1500
```

Stretch run if Colab is stable:

```python
!python scripts/16_collect_macrocycle_pretraining_data.py \
  --base . \
  --scan-limit 250000 \
  --page-size 1000 \
  --max-macrocycles 10000
```

Check count:

```python
import pandas as pd
pre = pd.read_csv("02_curated_data/v5_3_pretrain_macrocycles.csv")
print(pre.shape)
print(pre["split"].value_counts())
```

## Step 5 - Build Fragment-Linker Datasets

Build broad pretraining pairs:

```python
!python scripts/13_build_v5_3_fragment_dataset.py \
  --base . \
  --input-csv 02_curated_data/v5_3_pretrain_macrocycles.csv \
  --fragment-output 02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv \
  --anchor-output 03_features/v5_3_pretrain_anchor_atom_training.csv \
  --max-pairs-per-molecule 40
```

Build JAK2 fine-tuning pairs:

```python
!python scripts/13_build_v5_3_fragment_dataset.py \
  --base . \
  --input-csv 02_curated_data/jak2_curated_ligands.csv \
  --fragment-output 02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv \
  --anchor-output 03_features/v5_3_anchor_atom_training.csv \
  --max-pairs-per-molecule 40
```

Check dataset sizes:

```python
import pandas as pd
for path in [
    "02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv",
    "03_features/v5_3_pretrain_anchor_atom_training.csv",
    "02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv",
    "03_features/v5_3_anchor_atom_training.csv",
]:
    df = pd.read_csv(path)
    print(path, df.shape)
```

## Step 6 - Build Bigger SE(3) Graph Data

Build graphs for broad pretraining macrocycles:

```python
!python scripts/02_build_features.py \
  --base . \
  --input-csv 02_curated_data/v5_3_pretrain_macrocycles.csv \
  --feature-output 03_features/v5_3_pretrain_ligand_features.csv \
  --graph-output 03_features/ligand_graphs/v5_3_pretrain_se3_graphs.pt \
  --graph-index-output 03_features/ligand_graphs/v5_3_pretrain_se3_graph_index.csv \
  --skip-split-files
```

## Step 7 - Train Anchor Model: Pretrain Then Fine-Tune

Pretrain on broad macrocycles:

```python
!python scripts/14_train_v5_3_anchor_model.py \
  --base . \
  --input-csv 03_features/v5_3_pretrain_anchor_atom_training.csv \
  --output-dir 04_models_checkpoints/v5_3_anchor_pretrain \
  --epochs 1000 \
  --hidden-dim 192 \
  --learning-rate 0.0005 \
  --device cuda
```

Fine-tune on JAK2 macrocycles:

```python
!python scripts/14_train_v5_3_anchor_model.py \
  --base . \
  --input-csv 03_features/v5_3_anchor_atom_training.csv \
  --output-dir 04_models_checkpoints/v5_3_anchor \
  --init-checkpoint 04_models_checkpoints/v5_3_anchor_pretrain/anchor_site_model.pt \
  --epochs 500 \
  --hidden-dim 192 \
  --learning-rate 0.0002 \
  --device cuda
```

## Step 8 - Train Linker-Size Model: Pretrain Then Fine-Tune

Pretrain on broad macrocycles:

```python
!python scripts/15_train_v5_3_linker_size_model.py \
  --base . \
  --input-csv 02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv \
  --output-dir 04_models_checkpoints/v5_3_linker_size_pretrain \
  --epochs 1000 \
  --hidden-dim 192 \
  --learning-rate 0.0005 \
  --device cuda
```

Fine-tune on JAK2 macrocycles:

```python
!python scripts/15_train_v5_3_linker_size_model.py \
  --base . \
  --input-csv 02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv \
  --output-dir 04_models_checkpoints/v5_3_linker_size \
  --init-checkpoint 04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_model.pt \
  --epochs 500 \
  --hidden-dim 192 \
  --learning-rate 0.0002 \
  --device cuda
```

## Step 9 - Train SE(3): Pretrain Then Fine-Tune

Pretrain SE(3) on broad macrocycles:

```python
!python scripts/03_train_se3_main.py \
  --base . \
  --graph-path 03_features/ligand_graphs/v5_3_pretrain_se3_graphs.pt \
  --graph-index 03_features/ligand_graphs/v5_3_pretrain_se3_graph_index.csv \
  --checkpoint-dir 04_models_checkpoints/se3_flow_pretrain \
  --epochs 120 \
  --batch-size 8 \
  --hidden-dim 160 \
  --num-layers 5 \
  --learning-rate 0.00015 \
  --checkpoint-every 10 \
  --device cuda
```

Fine-tune SE(3) on JAK2:

```python
!python scripts/03_train_se3_main.py \
  --base . \
  --graph-path 03_features/ligand_graphs/se3_graphs.pt \
  --graph-index 03_features/ligand_graphs/se3_graph_index.csv \
  --checkpoint-dir 04_models_checkpoints/se3_flow \
  --init-checkpoint 04_models_checkpoints/se3_flow_pretrain/se3_best_checkpoint.pt \
  --epochs 80 \
  --batch-size 8 \
  --hidden-dim 160 \
  --num-layers 5 \
  --learning-rate 0.0001 \
  --checkpoint-every 10 \
  --device cuda
```

## Step 10 - Inspect Metrics

```python
import pandas as pd, json

files = {
    "anchor_pretrain": "04_models_checkpoints/v5_3_anchor_pretrain/anchor_training_summary.json",
    "anchor_finetune": "04_models_checkpoints/v5_3_anchor/anchor_training_summary.json",
    "linker_pretrain": "04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_training_summary.json",
    "linker_finetune": "04_models_checkpoints/v5_3_linker_size/linker_size_training_summary.json",
    "se3_pretrain": "04_models_checkpoints/se3_flow_pretrain/training_summary.json",
    "se3_finetune": "04_models_checkpoints/se3_flow/training_summary.json",
}

for name, path in files.items():
    print("\n==", name, "==")
    with open(path) as f:
        print(json.dumps(json.load(f), indent=2)[:2000])
```

## Step 11 - Package Everything For Codex

```python
import zipfile, os

ARTIFACT_ZIP = "/content/drive/MyDrive/EMD_V5_3_moredata_trained_artifacts.zip"
files = [
    "02_curated_data/v5_3_pretrain_macrocycles.csv",
    "02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv",
    "02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv",
    "03_features/v5_3_pretrain_anchor_atom_training.csv",
    "03_features/v5_3_anchor_atom_training.csv",
    "03_features/v5_3_pretrain_ligand_features.csv",
    "03_features/ligand_graphs/v5_3_pretrain_se3_graphs.pt",
    "03_features/ligand_graphs/v5_3_pretrain_se3_graph_index.csv",
    "04_models_checkpoints/v5_3_anchor_pretrain/anchor_site_model.pt",
    "04_models_checkpoints/v5_3_anchor_pretrain/anchor_training_log.csv",
    "04_models_checkpoints/v5_3_anchor_pretrain/anchor_training_summary.json",
    "04_models_checkpoints/v5_3_anchor/anchor_site_model.pt",
    "04_models_checkpoints/v5_3_anchor/anchor_training_log.csv",
    "04_models_checkpoints/v5_3_anchor/anchor_training_summary.json",
    "04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_model.pt",
    "04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_training_log.csv",
    "04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_training_summary.json",
    "04_models_checkpoints/v5_3_linker_size/linker_size_model.pt",
    "04_models_checkpoints/v5_3_linker_size/linker_size_training_log.csv",
    "04_models_checkpoints/v5_3_linker_size/linker_size_training_summary.json",
    "04_models_checkpoints/se3_flow_pretrain/se3_best_checkpoint.pt",
    "04_models_checkpoints/se3_flow_pretrain/se3_latest_checkpoint.pt",
    "04_models_checkpoints/se3_flow_pretrain/training_log.csv",
    "04_models_checkpoints/se3_flow_pretrain/training_curve.png",
    "04_models_checkpoints/se3_flow_pretrain/training_summary.json",
    "04_models_checkpoints/se3_flow/se3_best_checkpoint.pt",
    "04_models_checkpoints/se3_flow/se3_latest_checkpoint.pt",
    "04_models_checkpoints/se3_flow/training_log.csv",
    "04_models_checkpoints/se3_flow/training_curve.png",
    "04_models_checkpoints/se3_flow/training_summary.json",
    "00_project_registry/run_registry.csv",
    "00_project_registry/artifact_registry.csv",
]

with zipfile.ZipFile(ARTIFACT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for path in files:
        if os.path.exists(path):
            z.write(path, path)

print("Saved:", ARTIFACT_ZIP)
!ls -lh "$ARTIFACT_ZIP"
```

Download:

`EMD_V5_3_moredata_trained_artifacts.zip`

Put it in:

`C:\Users\srira\Desktop\new_plan`

Then tell Codex:

`I uploaded EMD_V5_3_moredata_trained_artifacts.zip. Merge it and run model-guided macrocycle generation.`

## Benchmark Rule

Do not claim MED is beaten until we compare:

- validity,
- uniqueness,
- macrocycle fraction,
- linker novelty,
- docking score distribution,
- pose sanity pass rate,
- JAK2 fine-tune performance.

This workflow gives the larger training/testing base needed for that benchmark.
