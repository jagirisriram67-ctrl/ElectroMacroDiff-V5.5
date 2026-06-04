# V5.5 Fast Vina-GPU Docking Cell

Use this instead of the slow full-depth Cell 5 when the all-ligand Vina-GPU run
takes too long.

This is a screening cascade:

1. Fast GPU screen all unfinished ligands with `search_depth = 8`.
2. Keep outputs resumable by skipping already docked ligands.
3. Later, redock only top candidates at higher depth if needed.

Vina-GPU 2.1 documents `search_depth` as the number of searching iterations and
recommends `thread` preferably below `10000`, so this cell avoids `16384`.

```python
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

THREADS = [8192, 6000, 4000, 2048]
SEARCH_DEPTH = 8
SMOKE_TIMEOUT_SECONDS = 240
SCORE_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)", re.I)

grid = json.loads(GRID_JSON.read_text(encoding="utf-8"))

def dir_arg(path: Path) -> str:
    text = str(path)
    return text if text.endswith("/") else text + "/"

def write_config(config_path: Path, ligand_dir: Path, output_dir: Path, thread: int, search_depth: int) -> Path:
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
search_depth = {search_depth}
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path

def run_vina(config_path: Path, log_path: Path, timeout: int | None = None) -> int:
    cmd = f"ulimit -s 8192 && {BINARY} --config {config_path}"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=VINA_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end="")
                log.write(line)
            rc = process.poll()
            if rc is not None:
                return rc
            if timeout is not None and (time.time() - start) > timeout:
                process.kill()
                log.write(f"\nTIMEOUT after {timeout}s\n")
                return -999

def parse_score(path: Path):
    match = SCORE_RE.search(path.read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1)) if match else None

def output_exists_for(candidate_id: str, output_dir: Path) -> bool:
    for path in output_dir.rglob(f"{candidate_id}*.pdbqt"):
        if parse_score(path) is not None:
            return True
    return False

all_ligands = sorted(CLEAN_PDBQT_DIR.glob("CAND_*.pdbqt"))
RAW_POSE_DIR.mkdir(parents=True, exist_ok=True)
already_done = {lig.stem for lig in all_ligands if output_exists_for(lig.stem, RAW_POSE_DIR)}
todo_ligands = [lig for lig in all_ligands if lig.stem not in already_done]

print("Clean ligands:", len(all_ligands))
print("Already docked:", len(already_done))
print("Remaining:", len(todo_ligands))

if not todo_ligands:
    print("Nothing left to dock.")
else:
    smoke_ligands = Path("/kaggle/working/vina_gpu_smoke_ligands_fast")
    smoke_out = Path("/kaggle/working/vina_gpu_smoke_out_fast")
    shutil.rmtree(smoke_ligands, ignore_errors=True)
    shutil.rmtree(smoke_out, ignore_errors=True)
    smoke_ligands.mkdir(parents=True)
    smoke_out.mkdir(parents=True)
    for ligand in todo_ligands[:2]:
        shutil.copy2(ligand, smoke_ligands / ligand.name)

    preferred_thread = None
    for thread in THREADS:
        shutil.rmtree(smoke_out, ignore_errors=True)
        smoke_out.mkdir(parents=True)
        cfg = write_config(
            Path(f"/kaggle/working/vina_gpu_fast_smoke_t{thread}_d{SEARCH_DEPTH}.txt"),
            smoke_ligands,
            smoke_out,
            thread,
            SEARCH_DEPTH,
        )
        log_path = SCORES_DIR / f"vina_gpu_fast_smoke_t{thread}_d{SEARCH_DEPTH}.log"
        rc = run_vina(cfg, log_path, timeout=SMOKE_TIMEOUT_SECONDS)
        outputs = sorted(smoke_out.rglob("*.pdbqt"))
        parsed = sum(1 for path in outputs if parse_score(path) is not None)
        print("FAST SMOKE", thread, "rc=", rc, "outputs=", len(outputs), "parsed=", parsed)
        if rc == 0 and parsed == len(list(smoke_ligands.glob("CAND_*.pdbqt"))):
            preferred_thread = thread
            break

    if preferred_thread is None:
        raise RuntimeError("Fast smoke failed. Do not run full docking.")

    todo_dir = Path("/kaggle/working/vina_gpu_todo_ligands_fast")
    shutil.rmtree(todo_dir, ignore_errors=True)
    todo_dir.mkdir(parents=True)
    for ligand in todo_ligands:
        shutil.copy2(ligand, todo_dir / ligand.name)

    cfg = write_config(
        Path(f"/kaggle/working/vina_gpu_{BRANCH}_fast_t{preferred_thread}_d{SEARCH_DEPTH}.txt"),
        todo_dir,
        RAW_POSE_DIR,
        preferred_thread,
        SEARCH_DEPTH,
    )
    log_path = SCORES_DIR / f"vina_gpu_2_1_fast_t{preferred_thread}_d{SEARCH_DEPTH}.log"
    print("Running fast full screen:", len(todo_ligands), "ligands")
    rc = run_vina(cfg, log_path)

    outputs = sorted(RAW_POSE_DIR.rglob("*.pdbqt"))
    parsed = sum(1 for path in outputs if parse_score(path) is not None)
    print("FAST FULL rc =", rc, "raw_outputs =", len(outputs), "parsed =", parsed)
    if rc != 0:
        raise RuntimeError(f"Fast Vina-GPU failed with exit code {rc}")

    shutil.copy2(log_path, VINA_STDOUT_LOG)
```
