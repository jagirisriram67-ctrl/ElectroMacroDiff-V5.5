# V5.3 Colab Vina-GPU 2.1 One-Cell Run

Use this after selecting `Runtime > Change runtime type > T4 GPU` in Google Colab.

This version replaces the older many-cell guide. Run only the single cell below from a fresh Colab runtime. It mounts Drive, extracts the zip, installs dependencies, rebuilds Vina-GPU 2.1, sanitizes all PDBQT ligands, smoke-tests 2 ligands, docks all 114 ligands, parses scores, ranks, benchmarks, and writes the result zip back to Drive.

References:

- Vina-GPU 2.1 repository: https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
- AutoDock Vina documentation: https://autodock-vina.readthedocs.io/

## Before Running

Upload this file to Google Drive:

`C:\Users\srira\Desktop\new_plan\EMD_V5_3_vina_gpu_colab_ready.zip`

It should appear in Drive as:

`MyDrive/EMD_V5_3_vina_gpu_colab_ready.zip`

## Run This One Cell

```python
from google.colab import drive
drive.mount("/content/drive")

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path("/content/drive/MyDrive/EMD_V5_3_vina_gpu_colab_ready.zip")
EXTRACT_DIR = Path("/content/EMD_V5_3_vina_gpu_extract")
RESULT_ZIP = Path("/content/drive/MyDrive/EMD_V5_3_vina_gpu_2_1_results.zip")
FAILURE_ZIP = Path("/content/drive/MyDrive/EMD_V5_3_vina_gpu_failure_debug.zip")

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Zip not found: {ZIP_PATH}\n"
        "Upload EMD_V5_3_vina_gpu_colab_ready.zip to MyDrive first."
    )

print("Removing old extracted folder and old Vina-GPU build...")
shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
shutil.rmtree(Path("/content/Vina-GPU-2.1"), ignore_errors=True)
for old_file in [RESULT_ZIP, FAILURE_ZIP]:
    if old_file.exists():
        old_file.unlink()

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
print("Extracting:", ZIP_PATH)
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

marker = Path("05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv")
matches = [p.parent.parent.parent for p in EXTRACT_DIR.rglob(str(marker))]
if not matches:
    print("Zip top-level entries:")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for name in z.namelist()[:60]:
            print(name)
    raise RuntimeError("Could not locate extracted EMD project root.")

PROJECT_DIR = matches[0]
RUNNER = PROJECT_DIR / "scripts" / "run_v5_3_vina_gpu_colab.py"
if not RUNNER.exists():
    raise FileNotFoundError(f"Runner script missing from zip: {RUNNER}")

print("Project directory:", PROJECT_DIR)
print("Runner:", RUNNER)
print("This will take time. Leave the Colab tab open.")

cmd = [
    sys.executable,
    str(RUNNER),
    "--base",
    str(PROJECT_DIR),
    "--threads",
    "4000,2048,1024",
    "--output-zip",
    str(RESULT_ZIP),
    "--failure-debug-zip",
    str(FAILURE_ZIP),
]
print("$", " ".join(cmd))
result = subprocess.run(cmd, cwd=PROJECT_DIR, text=True)

if result.returncode != 0:
    print("\nFAILED.")
    print("A debug zip should be here:", FAILURE_ZIP)
    print("Download/upload that debug zip so the exact failing command and logs can be checked.")
    raise RuntimeError(f"Vina-GPU pipeline failed with exit code {result.returncode}")

print("\nSUCCESS.")
print("Result zip:", RESULT_ZIP)
print("Size MB:", round(RESULT_ZIP.stat().st_size / 1024 / 1024, 2))
```

## Expected Output

The successful output file is:

`MyDrive/EMD_V5_3_vina_gpu_2_1_results.zip`

Bring that zip back to:

`C:\Users\srira\Desktop\new_plan`

Then say:

`I uploaded EMD_V5_3_vina_gpu_2_1_results.zip. Merge the Vina-GPU 2.1 results and update the benchmark report.`

If it fails, do not keep rerunning random cells. Download this debug file instead:

`MyDrive/EMD_V5_3_vina_gpu_failure_debug.zip`

Then send that zip here. It contains the exact driver log, Vina-GPU config files, and Vina-GPU output logs needed to fix the next real error.
