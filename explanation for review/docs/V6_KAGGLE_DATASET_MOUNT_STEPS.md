# V6 Kaggle Dataset Mount Steps

These cells prepare V6 without touching the frozen V5.5 branch.

Use one dataset lane at a time. Replace only the `/kaggle/input/...` paths that
exist in your Kaggle account.

## Cell 1: Copy Project

```python
import os
import shutil
from pathlib import Path

INPUT_BASE = Path("/kaggle/input/YOUR_PROJECT_DATASET/EMD_V5_2_Hybrid")
WORK = Path("/kaggle/working")
BASE = WORK / "EMD_V5_2_Hybrid"

if not INPUT_BASE.exists():
    raise FileNotFoundError(f"Project dataset not found: {INPUT_BASE}")

shutil.rmtree(BASE, ignore_errors=True)
shutil.copytree(INPUT_BASE, BASE)
os.chdir(BASE)

print("BASE:", BASE)
print("cwd:", os.getcwd())
```

## Cell 2: Lightweight Dependencies

```python
!python -m pip install -q pandas numpy pyyaml pyarrow

import pandas as pd
print("pandas:", pd.__version__)
```

## Cell 3: Scan Mounted Premium Datasets

```python
!python scripts/31_prepare_v6_premium_manifest.py \
  --base . \
  --activity-root /kaggle/input/YOUR_JAK_ACTIVITY_DATASET \
  --pdbbind-root /kaggle/input/YOUR_PDBBIND_DATASET \
  --plinder-root /kaggle/input/YOUR_PLINDER_DATASET \
  --biolip2-root /kaggle/input/YOUR_BIOLIP2_DATASET \
  --crossdocked-root /kaggle/input/YOUR_CROSSDOCKED_DATASET
```

If a dataset is not mounted yet, remove that line. Missing datasets are reported
as incomplete, not treated as a crash.

## Cell 4: Generate The Current V6 Training Plan

```python
!python scripts/32_write_v6_training_plan.py --base .

from pathlib import Path
plan = Path("docs/V6_KAGGLE_SEQUENTIAL_TRAINING_PLAN.md")
print(plan.read_text(encoding="utf-8")[:4000])
```

## Cell 5: Zip Preparation Outputs

```python
import shutil
from pathlib import Path

OUT = Path("/kaggle/working/v6_preparation_outputs")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)

for rel in [
    "03_features/v6_premium_dataset_manifest.csv",
    "03_features/v6_premium_dataset_summary.json",
    "docs/V6_KAGGLE_SEQUENTIAL_TRAINING_PLAN.md",
    "docs/V6_PREMIUM_DATASET_UPGRADE_PLAN.md",
]:
    src = Path(rel)
    if not src.exists():
        print("Missing optional:", rel)
        continue
    dst = OUT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

zip_path = shutil.make_archive("/kaggle/working/v6_preparation_outputs", "zip", OUT)
print(zip_path)
```

## What To Send Back

Download and send:

- `/kaggle/working/v6_preparation_outputs.zip`

After this file is merged, we pick the first ready lane and write the exact
training notebook for that lane.
