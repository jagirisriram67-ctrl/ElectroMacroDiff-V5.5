# V5.5 Kaggle GPU-Only Docking Cells

Use this notebook only with a project package that already contains prepared V5.5 docking inputs:

- `06_docking/v5_5_pocket_guided/receptor/docking_grid_5AEP_QUP.json`
- `06_docking/v5_5_pocket_guided/ligands_pdbqt_vina_gpu_clean/CAND_*.pdbqt`
- `06_docking/receptor/jak2_prepared.pdbqt`

This skips RDKit SDF generation, Meeko PDBQT preparation, and Vina-GPU sanitization on Kaggle. Kaggle is used only for Vina-GPU docking plus minimal file copying/packaging.

## Cell 1: Copy Mounted Dataset To Working Directory

```python
import os
import shutil
from pathlib import Path

WORK = Path("/kaggle/working")
BASE = WORK / "EMD_V5_2_Hybrid"

matches = [p for p in Path("/kaggle/input").rglob("EMD_V5_2_Hybrid") if p.is_dir()]
print("Found project folders:")
for p in matches:
    print(" ", p)

if not matches:
    raise FileNotFoundError("Could not find EMD_V5_2_Hybrid under /kaggle/input. Attach the optimized project dataset.")

INPUT_BASE = matches[0]
print("Using INPUT_BASE:", INPUT_BASE)

shutil.rmtree(BASE, ignore_errors=True)
shutil.copytree(INPUT_BASE, BASE)

os.chdir(BASE)
print("BASE:", BASE)
print("cwd:", os.getcwd())
```

## Cell 2: Install Vina-GPU Runtime Dependencies

```python
!apt-get update -qq
!apt-get install -y -qq git build-essential libboost-all-dev ocl-icd-opencl-dev opencl-headers clinfo
!nvidia-smi
!clinfo | head -n 60
```

## Cell 3: Verify Prepared Inputs Exist

```python
from pathlib import Path
import json

BASE = Path("/kaggle/working/EMD_V5_2_Hybrid").resolve()
BRANCH = "v5_5_pocket_guided"

DOCK_DIR = BASE / "06_docking" / BRANCH
RECEPTOR = BASE / "06_docking" / "receptor" / "jak2_prepared.pdbqt"
GRID_JSON = DOCK_DIR / "receptor" / "docking_grid_5AEP_QUP.json"
CLEAN_PDBQT_DIR = DOCK_DIR / "ligands_pdbqt_vina_gpu_clean"
RAW_POSE_DIR = DOCK_DIR / "poses_gpu_raw"
SCORES_DIR = DOCK_DIR / "scores"
VINA_STDOUT_LOG = SCORES_DIR / "vina_gpu_2_1_stdout.log"

RAW_POSE_DIR.mkdir(parents=True, exist_ok=True)
SCORES_DIR.mkdir(parents=True, exist_ok=True)

checks = [RECEPTOR, GRID_JSON, CLEAN_PDBQT_DIR]
for path in checks:
    print(path, "OK" if path.exists() else "MISSING")

clean_ligands = sorted(CLEAN_PDBQT_DIR.glob("CAND_*.pdbqt"))
print("Prepared clean Vina-GPU ligands:", len(clean_ligands))
if not RECEPTOR.exists() or not GRID_JSON.exists() or not clean_ligands:
    raise RuntimeError("Prepared docking inputs are missing. Use the GPU-only optimized project package.")

grid = json.loads(GRID_JSON.read_text(encoding="utf-8"))
print("Grid center:", grid["center"])
print("Grid size:", grid["size"])
```

## Cell 4: Build Or Reuse Vina-GPU 2.1

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
    if not VINA_REPO.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git", str(VINA_REPO)],
            check=True,
        )
    shutil.copy2(BINARY_CACHE, BINARY)
    subprocess.run(["chmod", "+x", str(BINARY)], check=True)
    print("Restored cached binary into Vina-GPU source directory:", BINARY)
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
    build = subprocess.run(["bash", "-lc", "ulimit -s 8192 && make clean && make source -j4"], cwd=VINA_DIR)
    if build.returncode != 0:
        write_makefile(False)
        subprocess.run(["bash", "-lc", "ulimit -s 8192 && make clean && make source -j4"], cwd=VINA_DIR, check=True)
    subprocess.run(["chmod", "+x", str(BINARY)], check=True)
    shutil.copy2(BINARY, BINARY_CACHE)
    print("Built binary and saved cache:", BINARY_CACHE)

subprocess.run(["ldd", str(BINARY)], check=True)
```

## Cell 5: Vina-GPU Smoke Test And Full Docking

```python
import json
import re
import shutil
import subprocess
from pathlib import Path

THREADS = [16384, 8192, 4000, 2048, 1024]
SCORE_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)", re.I)
grid = json.loads(GRID_JSON.read_text(encoding="utf-8"))

def dir_arg(path: Path) -> str:
    text = str(path)
    return text if text.endswith("/") else text + "/"

def write_config(config_path: Path, ligand_dir: Path, output_dir: Path, thread: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = f"""receptor = {RECEPTOR}
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
    match = SCORE_RE.search(path.read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1)) if match else None

smoke_ligands = Path("/kaggle/working/vina_gpu_smoke_ligands")
smoke_out = Path("/kaggle/working/vina_gpu_smoke_out")
shutil.rmtree(smoke_ligands, ignore_errors=True)
shutil.rmtree(smoke_out, ignore_errors=True)
smoke_ligands.mkdir(parents=True)
smoke_out.mkdir(parents=True)

for ligand in sorted(CLEAN_PDBQT_DIR.glob("CAND_*.pdbqt"))[:2]:
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

print("Chosen Vina-GPU thread:", preferred_thread)

shutil.rmtree(RAW_POSE_DIR, ignore_errors=True)
RAW_POSE_DIR.mkdir(parents=True, exist_ok=True)
cfg = write_config(Path(f"/kaggle/working/vina_gpu_{BRANCH}_{preferred_thread}.txt"), CLEAN_PDBQT_DIR, RAW_POSE_DIR, preferred_thread)
log_path = SCORES_DIR / f"vina_gpu_2_1_thread_{preferred_thread}.log"
rc = run_vina(cfg, log_path)
outputs = sorted(RAW_POSE_DIR.rglob("*.pdbqt"))
parsed = sum(1 for path in outputs if parse_score(path) is not None)
print("FULL rc =", rc, "outputs =", len(outputs), "parsed =", parsed)
if rc != 0:
    raise RuntimeError(f"Vina-GPU failed with exit code {rc}")
shutil.copy2(log_path, VINA_STDOUT_LOG)
```

## Cell 6: Zip Raw GPU Docking Outputs

```python
import json
import shutil
from pathlib import Path

OUT_ROOT = Path("/kaggle/working/v5_5_gpu_only_docking_outputs")
shutil.rmtree(OUT_ROOT, ignore_errors=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

summary = {
    "branch": BRANCH,
    "clean_ligand_count": len(list(CLEAN_PDBQT_DIR.glob("CAND_*.pdbqt"))),
    "raw_pose_count": len(list(RAW_POSE_DIR.rglob("*.pdbqt"))),
    "grid_json": str(GRID_JSON),
    "receptor": str(RECEPTOR),
    "note": "Kaggle run skipped RDKit/Meeko ligand preparation and ran Vina-GPU docking only.",
}
(OUT_ROOT / "gpu_only_docking_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

paths_to_copy = [
    RAW_POSE_DIR,
    SCORES_DIR,
    GRID_JSON,
    BASE / "05_generated_candidates" / BRANCH / "generated_v5_5_pocket_guided.csv",
    BASE / "06_docking" / BRANCH / "ligands_pdbqt_vina_gpu_clean",
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

zip_path = shutil.make_archive("/kaggle/working/v5_5_gpu_only_docking_outputs", "zip", OUT_ROOT)
print(zip_path)
```

Download and send back:

`/kaggle/working/v5_5_gpu_only_docking_outputs.zip`
