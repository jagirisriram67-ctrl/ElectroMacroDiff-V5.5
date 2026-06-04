# V5.5 Kaggle Docking Cells

This notebook flow is for the Kaggle-mounted project folder:

`/kaggle/input/datasets/sriramnetwork/emd-v5-2-hybrid-v5-5-4doc/EMD_V5_2_Hybrid`

Kaggle input folders are read-only, so Cell 1 copies the project into `/kaggle/working/EMD_V5_2_Hybrid` before running anything.

## Cell 1: Copy Mounted Dataset To Working Directory

```python
import os
import shutil
from pathlib import Path

INPUT_BASE = Path("/kaggle/input/datasets/sriramnetwork/emd-v5-2-hybrid-v5-5-4doc/EMD_V5_2_Hybrid")
WORK = Path("/kaggle/working")
BASE = WORK / "EMD_V5_2_Hybrid"

print("INPUT_BASE:", INPUT_BASE)
print("INPUT_BASE exists:", INPUT_BASE.exists())
if not INPUT_BASE.exists():
    raise FileNotFoundError(INPUT_BASE)

shutil.rmtree(BASE, ignore_errors=True)
shutil.copytree(INPUT_BASE, BASE)

os.chdir(BASE)
print("BASE:", BASE)
print("cwd:", os.getcwd())
print("candidate csv exists:", Path("05_generated_candidates/v5_5_pocket_guided/generated_v5_5_pocket_guided.csv").exists())
```

## Cell 2: Install And Verify Dependencies

```python
!apt-get update -qq
!apt-get install -y -qq git build-essential libboost-all-dev ocl-icd-opencl-dev opencl-headers clinfo
!python -m pip install -q --upgrade pip setuptools wheel
!python -m pip install -q pandas numpy pyyaml matplotlib pytest
!python -m pip install -q rdkit
!python -m pip install -q meeko
```

```python
from rdkit import Chem
import rdkit
import meeko

print("RDKit OK:", rdkit.__version__)
print("Meeko OK:", getattr(meeko, "__version__", "installed"))
```

```python
!nvidia-smi
!clinfo | head -n 60
```

## Cell 3: Define Branch Paths And Verify Inputs

```python
from pathlib import Path
import pandas as pd

BASE = Path("/kaggle/working/EMD_V5_2_Hybrid").resolve()
BRANCH = "v5_5_pocket_guided"

GEN_CSV = BASE / "05_generated_candidates" / BRANCH / "generated_v5_5_pocket_guided.csv"
ATTEMPT_LOG = BASE / "05_generated_candidates" / BRANCH / "v5_5_pocket_guided_attempt_log.csv"

DOCK_DIR = BASE / "06_docking" / BRANCH
RECEPTOR_DIR = DOCK_DIR / "receptor"
SCORES_DIR = DOCK_DIR / "scores"
POSE_DIR = DOCK_DIR / "poses"
RAW_POSE_DIR = DOCK_DIR / "poses_gpu_raw"
IMG_DIR = DOCK_DIR / "images" / "pose_sanity"
SDF_DIR = DOCK_DIR / "ligands_sdf"
PDBQT_DIR = DOCK_DIR / "ligands_pdbqt"
CLEAN_PDBQT_DIR = DOCK_DIR / "ligands_pdbqt_vina_gpu_clean"
MEEKO_LOG_DIR = SCORES_DIR / "meeko_ligand_logs"

GRID_JSON = RECEPTOR_DIR / "docking_grid_5AEP_QUP.json"
DOCKING_CSV = SCORES_DIR / "docking_scores_full_vina_gpu_2_1.csv"
POSE_SANITY_CSV = SCORES_DIR / "pose_sanity_scores.csv"
POCKET_SCORES_CSV = SCORES_DIR / "pocket_electronic_fit_scores.csv"
POCKET_PROFILE_JSON = SCORES_DIR / "jak2_pocket_electronic_profile.json"
SE3_CSV = SCORES_DIR / "se3_geometry_scores.csv"
SANITIZE_SUMMARY_CSV = SCORES_DIR / "vina_gpu_ligand_sanitization_summary.csv"
VINA_STDOUT_LOG = SCORES_DIR / "vina_gpu_2_1_stdout.log"

ADMET_CSV = BASE / "07_admet_synthesis" / "v5_5_pocket_guided_admet_scores.csv"
FLAGS_CSV = BASE / "07_admet_synthesis" / "v5_5_pocket_guided_filter_flags.csv"
NOTES_CSV = BASE / "07_admet_synthesis" / "v5_5_pocket_guided_safety_proxy_notes.csv"

RANK_CSV = BASE / "08_final_ranking" / "v5_5_pocket_guided_ranked_candidates.csv"
POCKET_RANK_CSV = BASE / "08_final_ranking" / "v5_5_pocket_guided_pocket_electronic_ranked_candidates.csv"
SE3_RANK_CSV = BASE / "08_final_ranking" / "v5_5_pocket_guided_pocket_electronic_ranked_candidates_with_se3.csv"

for path in [RECEPTOR_DIR, SCORES_DIR, POSE_DIR, RAW_POSE_DIR, IMG_DIR, SDF_DIR, PDBQT_DIR, CLEAN_PDBQT_DIR, MEEKO_LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(GEN_CSV)
print("Generated candidates:", len(df))
print(df.head(3)[["candidate_id", "canonical_smiles"]])

required = [
    GEN_CSV,
    ATTEMPT_LOG,
    BASE / "06_docking" / "receptor" / "jak2_prepared.pdbqt",
    BASE / "06_docking" / "receptor" / "5AEP_receptor_clean.pdb",
    BASE / "04_models_checkpoints" / "v5_5_se3_continuation" / "se3_v5_5_best_checkpoint.pt",
]
for path in required:
    print(path, "OK" if path.exists() else "MISSING")

print("Expected before Cell 4:")
print(GRID_JSON, "WILL_BE_CREATED_IN_CELL_4")
```

## Cell 4: Prepare V5.5 Docking SDFs

This step is RDKit conformer generation, so it runs on CPU. The cell below uses multiple Kaggle CPU workers. GPU usage begins in Cell 8 when Vina-GPU performs docking.

```python
!python scripts/30_prepare_v5_5_docking_inputs_parallel.py \
  --base . \
  --input-csv 05_generated_candidates/v5_5_pocket_guided/generated_v5_5_pocket_guided.csv \
  --top-n 2000 \
  --clean \
  --ligand-sdf-dir 06_docking/v5_5_pocket_guided/ligands_sdf \
  --combined-sdf 06_docking/v5_5_pocket_guided/ligands_sdf/candidates_for_docking.sdf \
  --grid-json 06_docking/v5_5_pocket_guided/receptor/docking_grid_5AEP_QUP.json \
  --num-workers 4 \
  --stage V5_5_docking_prep
```

## Cell 4B: Verify Docking Prep Outputs

```python
checks = [
    BASE / "06_docking" / "v5_5_pocket_guided" / "receptor" / "docking_grid_5AEP_QUP.json",
    BASE / "06_docking" / "v5_5_pocket_guided" / "ligands_sdf" / "candidates_for_docking.sdf",
    BASE / "06_docking" / "v5_5_pocket_guided" / "ligands_sdf" / "docking_input_manifest.csv",
]

for path in checks:
    print(path, "OK" if path.exists() else "MISSING")
```

## Cell 5: Prepare Receptor And Ligand PDBQT

```python
!python scripts/06_prepare_pdbqt.py \
  --base . \
  --workers 4 \
  --clean-ligands \
  --grid-json 06_docking/v5_5_pocket_guided/receptor/docking_grid_5AEP_QUP.json \
  --ligand-sdf-dir 06_docking/v5_5_pocket_guided/ligands_sdf \
  --ligand-pdbqt-dir 06_docking/v5_5_pocket_guided/ligands_pdbqt \
  --ligand-log-dir 06_docking/v5_5_pocket_guided/scores/meeko_ligand_logs \
  --summary-json 06_docking/v5_5_pocket_guided/scores/pdbqt_preparation_summary.json \
  --receptor-pdbqt 06_docking/receptor/jak2_prepared.pdbqt \
  --receptor-log 06_docking/receptor/mk_prepare_receptor.log \
  --stage V5_5_pdbqt_prep
```

## Cell 5B: Count Prepared Ligands

```python
pdbqt_count = len(list(PDBQT_DIR.glob("CAND_*.pdbqt")))
sdf_count = len(list(SDF_DIR.glob("CAND_*.sdf")))
print("SDF count:", sdf_count)
print("Ligand PDBQT count:", pdbqt_count)
if pdbqt_count == 0:
    raise RuntimeError("No ligand PDBQT files were prepared. Check Cell 5 output/logs.")
```

## Cell 6: Sanitize Ligands For Vina-GPU

```python
!python scripts/06_prepare_vina_gpu_ligands.py \
  --base . \
  --input-dir 06_docking/v5_5_pocket_guided/ligands_pdbqt \
  --output-dir 06_docking/v5_5_pocket_guided/ligands_pdbqt_vina_gpu_clean \
  --summary-csv 06_docking/v5_5_pocket_guided/scores/vina_gpu_ligand_sanitization_summary.csv \
  --clean \
  --expected-count 0 \
  --workers 8 \
  --stage V5_5_vina_gpu_ligand_sanitize
```

```python
clean_count = len(list(CLEAN_PDBQT_DIR.glob("CAND_*.pdbqt")))
print("Clean Vina-GPU ligand count:", clean_count)
if clean_count == 0:
    raise RuntimeError("No clean Vina-GPU ligands were created.")
```

## Cell 7: Clone And Build Vina-GPU 2.1

```python
import shutil
import subprocess
from pathlib import Path

VINA_REPO = Path("/kaggle/working/Vina-GPU-2.1")
VINA_DIR = VINA_REPO / "AutoDock-Vina-GPU-2.1"
BINARY = VINA_DIR / "AutoDock-Vina-GPU-2-1"
BINARY_CACHE = Path("/kaggle/working/AutoDock-Vina-GPU-2-1-cached")

makefile_template = """WORK_DIR=$(shell pwd)
OPENCL_VERSION=-DOPENCL_3_0
GPU_PLATFORM=-DNVIDIA_PLATFORM
DOCKING_BOX_SIZE=-DSMALL_BOX

BOOST_INC_PATH=-I/usr/include
VINA_GPU_INC_PATH=-I$(WORK_DIR)/lib -I$(WORK_DIR)/OpenCL/inc
OPENCL_INC_PATH=-I/usr/include
LIB1=-lboost_program_options -lboost_system -lboost_filesystem -lboost_thread -lOpenCL
LIB2={lib2}
LIB3=-lm -lpthread
LIB_PATH=-L/usr/lib/x86_64-linux-gnu
SRC=./lib/*.cpp ./OpenCL/src/wrapcl.cpp
MACRO=$(OPENCL_VERSION) $(GPU_PLATFORM) $(DOCKING_BOX_SIZE) -DBOOST_TIMER_ENABLE_DEPRECATED
CXX=g++
CXXFLAGS=-O3 -std=c++17

all: out

out: ./main/main.cpp
\t$(CXX) -o AutoDock-Vina-GPU-2-1 $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp $(CXXFLAGS) $(SRC) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION) -DNDEBUG

source: ./main/main.cpp
\t$(CXX) -o AutoDock-Vina-GPU-2-1 $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp $(CXXFLAGS) $(SRC) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION) -DNDEBUG -DBUILD_KERNEL_FROM_SOURCE

clean:
\trm -f AutoDock-Vina-GPU-2-1
"""

def write_makefile(include_stdcppfs: bool) -> None:
    lib2 = "-lstdc++fs" if include_stdcppfs else ""
    (VINA_DIR / "Makefile").write_text(makefile_template.format(lib2=lib2), encoding="utf-8")

if BINARY_CACHE.exists():
    BINARY = BINARY_CACHE
    subprocess.run(["chmod", "+x", str(BINARY)], check=True)
    print("Using cached Vina-GPU binary:", BINARY)
else:
    if VINA_REPO.exists():
        shutil.rmtree(VINA_REPO)

    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git", str(VINA_REPO)],
        check=True,
    )

    for kernel in VINA_DIR.glob("Kernel*_Opt.bin"):
        kernel.unlink()

    write_makefile(True)
    build = subprocess.run(["bash", "-lc", "ulimit -s 8192 && make clean && make source -j2"], cwd=VINA_DIR)
    if build.returncode != 0:
        write_makefile(False)
        subprocess.run(["bash", "-lc", "ulimit -s 8192 && make clean && make source -j2"], cwd=VINA_DIR, check=True)

    subprocess.run(["chmod", "+x", str(BINARY)], check=True)
    shutil.copy2(BINARY, BINARY_CACHE)
    BINARY = BINARY_CACHE
    subprocess.run(["chmod", "+x", str(BINARY)], check=True)
    print("Built and cached Vina-GPU binary:", BINARY)

subprocess.run(["ldd", str(BINARY)], check=True)
```

## Cell 8: Smoke Test Then Full Vina-GPU Run

```python
import json
import re
import shutil
import subprocess
from pathlib import Path

# Smoke test tries aggressive values first and falls back if the GPU/runtime rejects them.
THREADS = [16384, 8192, 4000, 2048, 1024]
SCORE_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)", re.I)

clean_dir = CLEAN_PDBQT_DIR
raw_out = RAW_POSE_DIR
grid = json.loads(GRID_JSON.read_text(encoding="utf-8"))
receptor = BASE / "06_docking" / "receptor" / "jak2_prepared.pdbqt"

def dir_arg(path: Path) -> str:
    text = str(path)
    return text if text.endswith("/") else text + "/"

def write_config(config_path: Path, ligand_dir: Path, output_dir: Path, thread: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = f"""receptor = {receptor}
ligand_directory = {dir_arg(ligand_dir)}
output_directory = {dir_arg(output_dir)}
opencl_binary_path = {dir_arg(VINA_DIR)}
center_x = {grid['center']['x']}
center_y = {grid['center']['y']}
center_z = {grid['center']['z']}
size_x = {grid['size']['x']}
size_y = {grid['size']['y']}
size_z = {grid['size']['z']}
thread = {thread}
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path

def run_vina(config_path: Path, log_path: Path) -> int:
    cmd = f"ulimit -s 8192 && {BINARY} --config {config_path}"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=VINA_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
    return process.wait()

def parse_score(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = SCORE_RE.search(text)
    return float(match.group(1)) if match else None

smoke_ligands = Path("/kaggle/working/vina_gpu_smoke_ligands")
smoke_out = Path("/kaggle/working/vina_gpu_smoke_out")
shutil.rmtree(smoke_ligands, ignore_errors=True)
shutil.rmtree(smoke_out, ignore_errors=True)
smoke_ligands.mkdir(parents=True)
smoke_out.mkdir(parents=True)

for ligand in sorted(clean_dir.glob("CAND_*.pdbqt"))[:2]:
    shutil.copy2(ligand, smoke_ligands / ligand.name)

preferred_thread = None
for thread in THREADS:
    shutil.rmtree(smoke_out, ignore_errors=True)
    smoke_out.mkdir(parents=True)
    cfg = write_config(Path(f"/kaggle/working/vina_gpu_smoke_{thread}.txt"), smoke_ligands, smoke_out, thread)
    log_path = SCORES_DIR / f"vina_gpu_2_1_smoke_thread_{thread}.log"
    rc = run_vina(cfg, log_path)
    outputs = sorted(smoke_out.rglob("*.pdbqt"))
    parsed = sum(1 for path in outputs if parse_score(path) is not None)
    print("SMOKE", thread, "rc=", rc, "outputs=", len(outputs), "parsed=", parsed)
    if rc == 0 and len(outputs) == 2 and parsed == 2:
        preferred_thread = thread
        break

if preferred_thread is None:
    raise RuntimeError("Smoke test failed for all thread settings")

print("Chosen thread =", preferred_thread)

shutil.rmtree(raw_out, ignore_errors=True)
raw_out.mkdir(parents=True, exist_ok=True)

cfg = write_config(Path(f"/kaggle/working/vina_gpu_{BRANCH}_{preferred_thread}.txt"), clean_dir, raw_out, preferred_thread)
log_path = SCORES_DIR / f"vina_gpu_2_1_thread_{preferred_thread}.log"
rc = run_vina(cfg, log_path)
outputs = sorted(raw_out.rglob("*.pdbqt"))
parsed = sum(1 for path in outputs if parse_score(path) is not None)
print("FULL rc =", rc, "outputs =", len(outputs), "parsed =", parsed)

if rc != 0:
    raise RuntimeError(f"Vina-GPU failed with exit code {rc}")

shutil.copy2(log_path, VINA_STDOUT_LOG)
```

## Cell 9: Parse Raw GPU Outputs Into Branch CSV

```python
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

SCORE_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)", re.I)

def parse_score(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = SCORE_RE.search(text)
    return float(match.group(1)) if match else None

def find_raw_pose(candidate_id: str, all_raw: list[Path]):
    exact_names = {
        f"{candidate_id}.pdbqt",
        f"{candidate_id}_out.pdbqt",
        f"{candidate_id}_vina_out.pdbqt",
    }
    for path in all_raw:
        if path.name in exact_names:
            return path
    for path in all_raw:
        if path.stem.startswith(candidate_id):
            return path
    return None

grid = json.loads(GRID_JSON.read_text(encoding="utf-8"))
candidates = pd.read_csv(GEN_CSV)
smiles_by_id = dict(zip(candidates["candidate_id"].astype(str), candidates["canonical_smiles"].astype(str)))
all_raw = sorted(RAW_POSE_DIR.rglob("*.pdbqt"))

def process_ligand(ligand: Path) -> dict:
    candidate_id = ligand.stem
    raw_pose = find_raw_pose(candidate_id, all_raw)
    final_pose = POSE_DIR / f"{candidate_id}_vina_out.pdbqt"
    best_score = None
    note = "missing_gpu_pose"
    if raw_pose is not None:
        shutil.copy2(raw_pose, final_pose)
        best_score = parse_score(raw_pose)
        note = "parsed_vina_gpu_2_1" if best_score is not None else "gpu_pose_no_score_found"
    return {
        "candidate_id": candidate_id,
        "smiles": smiles_by_id.get(candidate_id, ""),
        "docking_engine": "vina_gpu_2_1",
        "receptor_pdb": "5AEP",
        "grid_center_x": grid["center"]["x"],
        "grid_center_y": grid["center"]["y"],
        "grid_center_z": grid["center"]["z"],
        "grid_size_x": grid["size"]["x"],
        "grid_size_y": grid["size"]["y"],
        "grid_size_z": grid["size"]["z"],
        "best_score": best_score,
        "pose_rank": 1 if best_score is not None else "",
        "pose_file": str(final_pose) if final_pose.exists() else "",
        "pose_image": "",
        "control_or_generated": "generated",
        "notes": note,
    }

ligands = sorted(CLEAN_PDBQT_DIR.glob("CAND_*.pdbqt"))
rows = []
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_ligand, ligand): ligand for ligand in ligands}
    for index, future in enumerate(as_completed(futures), start=1):
        rows.append(future.result())
        if index % 100 == 0 or index == len(futures):
            print("Parsed/collated poses:", index, "/", len(futures))

docking = pd.DataFrame(rows)
docking.to_csv(DOCKING_CSV, index=False)
print("Parsed scores:", int(docking["best_score"].notna().sum()), "/", len(docking))
print(docking.sort_values("best_score", na_position="last").head(10)[["candidate_id", "best_score", "notes"]])
```

## Cell 10: ADMET, Ranking, Pose Sanity, Pocket Scoring, SE3, Benchmark

```python
!python scripts/05_score_admet_synthesis.py \
  --base . \
  --candidates-csv 05_generated_candidates/v5_5_pocket_guided/generated_v5_5_pocket_guided.csv \
  --output-csv 07_admet_synthesis/v5_5_pocket_guided_admet_scores.csv \
  --flags-csv 07_admet_synthesis/v5_5_pocket_guided_filter_flags.csv \
  --notes-csv 07_admet_synthesis/v5_5_pocket_guided_safety_proxy_notes.csv \
  --stage V5_5_vina_gpu_admet

!python scripts/07_rank_candidates.py \
  --base . \
  --candidates-csv 05_generated_candidates/v5_5_pocket_guided/generated_v5_5_pocket_guided.csv \
  --docking-csv 06_docking/v5_5_pocket_guided/scores/docking_scores_full_vina_gpu_2_1.csv \
  --admet-csv 07_admet_synthesis/v5_5_pocket_guided_admet_scores.csv \
  --output-csv 08_final_ranking/v5_5_pocket_guided_ranked_candidates.csv \
  --stage V5_5_vina_gpu_ranking

!python scripts/06_pose_sanity.py \
  --base . \
  --top-n 20 \
  --ranking-csv 08_final_ranking/v5_5_pocket_guided_ranked_candidates.csv \
  --receptor-pdb 06_docking/receptor/5AEP_receptor_clean.pdb \
  --grid-json 06_docking/v5_5_pocket_guided/receptor/docking_grid_5AEP_QUP.json \
  --pose-dir 06_docking/v5_5_pocket_guided/poses \
  --output-csv 06_docking/v5_5_pocket_guided/scores/pose_sanity_scores.csv \
  --image-dir 06_docking/v5_5_pocket_guided/images/pose_sanity \
  --stage V5_5_vina_gpu_pose_sanity

!python scripts/07_rank_candidates.py \
  --base . \
  --candidates-csv 05_generated_candidates/v5_5_pocket_guided/generated_v5_5_pocket_guided.csv \
  --docking-csv 06_docking/v5_5_pocket_guided/scores/docking_scores_full_vina_gpu_2_1.csv \
  --pose-sanity-csv 06_docking/v5_5_pocket_guided/scores/pose_sanity_scores.csv \
  --admet-csv 07_admet_synthesis/v5_5_pocket_guided_admet_scores.csv \
  --output-csv 08_final_ranking/v5_5_pocket_guided_ranked_candidates.csv \
  --stage V5_5_vina_gpu_ranking_with_pose

!python scripts/21_score_pocket_electronics.py \
  --base . \
  --ranking-csv 08_final_ranking/v5_5_pocket_guided_ranked_candidates.csv \
  --docking-csv 06_docking/v5_5_pocket_guided/scores/docking_scores_full_vina_gpu_2_1.csv \
  --pose-dir 06_docking/v5_5_pocket_guided/poses \
  --receptor 06_docking/receptor/jak2_prepared.pdbqt \
  --output-csv 06_docking/v5_5_pocket_guided/scores/pocket_electronic_fit_scores.csv \
  --output-ranking-csv 08_final_ranking/v5_5_pocket_guided_pocket_electronic_ranked_candidates.csv \
  --profile-json 06_docking/v5_5_pocket_guided/scores/jak2_pocket_electronic_profile.json \
  --top-n 0 \
  --stage V5_5_pocket_electronic_fit

!python scripts/23_score_se3_geometry.py \
  --base . \
  --ranking-csv 08_final_ranking/v5_5_pocket_guided_pocket_electronic_ranked_candidates.csv \
  --checkpoint 04_models_checkpoints/v5_5_se3_continuation/se3_v5_5_best_checkpoint.pt \
  --output-csv 06_docking/v5_5_pocket_guided/scores/se3_geometry_scores.csv \
  --output-ranking-csv 08_final_ranking/v5_5_pocket_guided_pocket_electronic_ranked_candidates_with_se3.csv \
  --device cuda \
  --num-repeats 3 \
  --stage V5_5_se3_geometry_scoring

!python scripts/18_build_v5_3_benchmark_report.py --base .
!python interface/extract_data.py
```

## Cell 11: Quick Result Check

```python
import json
import pandas as pd

docking = pd.read_csv(DOCKING_CSV)
ranked_path = SE3_RANK_CSV if SE3_RANK_CSV.exists() else POCKET_RANK_CSV
ranked = pd.read_csv(ranked_path)
benchmark = json.loads((BASE / "09_reports" / "v5_3_benchmark" / "emd_v5_3_benchmark_summary.json").read_text(encoding="utf-8"))

print("Docked rows:", len(docking))
print("Parsed scores:", int(docking["best_score"].notna().sum()))
print("Best Vina:", docking["best_score"].min())
print("Median Vina:", docking["best_score"].median())
print("Ranking file:", ranked_path)
print(ranked.head(10)[["candidate_id", "best_score", "pocket_electronic_fit_score", "pocket_guided_final_score"]])
print("Active benchmark branch:", benchmark.get("active_branch_for_med_comparison"))
```

## Cell 12: Zip The V5.5 Docking Outputs

```python
import shutil
from pathlib import Path

OUT_ROOT = Path("/kaggle/working/v5_5_docking_outputs")
shutil.rmtree(OUT_ROOT, ignore_errors=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

paths_to_copy = [
    BASE / "06_docking" / "v5_5_pocket_guided",
    BASE / "07_admet_synthesis" / "v5_5_pocket_guided_admet_scores.csv",
    BASE / "07_admet_synthesis" / "v5_5_pocket_guided_filter_flags.csv",
    BASE / "07_admet_synthesis" / "v5_5_pocket_guided_safety_proxy_notes.csv",
    BASE / "08_final_ranking" / "v5_5_pocket_guided_ranked_candidates.csv",
    BASE / "08_final_ranking" / "v5_5_pocket_guided_pocket_electronic_ranked_candidates.csv",
    BASE / "08_final_ranking" / "v5_5_pocket_guided_pocket_electronic_ranked_candidates_with_se3.csv",
    BASE / "09_reports" / "v5_3_benchmark",
    BASE / "interface" / "data.js",
]

for path in paths_to_copy:
    if not path.exists():
        print("Missing optional output:", path)
        continue
    target = OUT_ROOT / path.relative_to(BASE)
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, target, dirs_exist_ok=True)
    else:
        shutil.copy2(path, target)

zip_path = shutil.make_archive("/kaggle/working/v5_5_docking_outputs", "zip", OUT_ROOT)
print(zip_path)
```

Download and send back:

`/kaggle/working/v5_5_docking_outputs.zip`
