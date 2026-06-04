from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path, PureWindowsPath


VALID_AD_TYPES = {
    "C",
    "A",
    "N",
    "O",
    "P",
    "S",
    "H",
    "F",
    "I",
    "NA",
    "OA",
    "SA",
    "HD",
    "Mg",
    "Mn",
    "Zn",
    "Ca",
    "Fe",
    "Cl",
    "Br",
}

SCORE_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)", re.I)


class ColabVinaGpuRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base = Path(args.base).resolve()
        self.expected_count = int(args.expected_count)
        self.threads = [int(x.strip()) for x in str(args.threads).split(",") if x.strip()]
        self.vina_repo_dir = Path(args.vina_repo_dir).resolve()
        self.vina_dir = self.vina_repo_dir / "AutoDock-Vina-GPU-2.1"
        self.binary = self.vina_dir / "AutoDock-Vina-GPU-2-1"
        self.scores_dir = self.base / "06_docking" / "v5_3_model_guided" / "scores"
        self.log_path = self.scores_dir / "vina_gpu_colab_driver.log"
        self.log_handle = None

    def __enter__(self) -> "ColabVinaGpuRunner":
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        self.log("V5.3 Vina-GPU Colab driver started")
        self.log(f"Base: {self.base}")
        self.log(f"Threads: {self.threads}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None

    def log(self, message: str = "") -> None:
        line = str(message)
        print(line, flush=True)
        if self.log_handle:
            self.log_handle.write(line + "\n")
            self.log_handle.flush()

    def run(
        self,
        command: list[str],
        cwd: Path | None = None,
        check: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        cwd = Path(cwd or self.base)
        printable = " ".join(shlex.quote(str(part)) for part in command)
        self.log("")
        self.log(f"$ {printable}")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output_parts: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            output_parts.append(line)
            print(line, end="", flush=True)
            if self.log_handle:
                self.log_handle.write(line)
                self.log_handle.flush()
        return_code = process.wait()
        output = "".join(output_parts)
        result = subprocess.CompletedProcess(command, return_code, output, None)
        if check and return_code != 0:
            raise RuntimeError(f"Command failed with exit code {return_code}: {printable}")
        return result

    def run_bash(self, script: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
        return self.run(["bash", "-lc", script], cwd=cwd, check=check)

    def check_inputs(self) -> None:
        self.log("")
        self.log("=== 1. Checking project inputs ===")
        required = [
            self.base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles.csv",
            self.base / "06_docking" / "receptor" / "jak2_prepared.pdbqt",
            self.base / "06_docking" / "receptor" / "5AEP_receptor_clean.pdb",
            self.base / "06_docking" / "v5_3_model_guided" / "receptor" / "docking_grid_5AEP_QUP.json",
            self.base / "scripts" / "06_prepare_vina_gpu_ligands.py",
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(path)

        with required[0].open("r", encoding="utf-8", newline="") as handle:
            candidates = list(csv.DictReader(handle))
        raw_ligands = sorted((self.base / "06_docking" / "v5_3_model_guided" / "ligands_pdbqt").glob("CAND_*.pdbqt"))
        self.log(f"Candidates: {len(candidates)}")
        self.log(f"Raw ligand PDBQT files: {len(raw_ligands)}")
        if len(candidates) != self.expected_count:
            raise RuntimeError(f"Expected {self.expected_count} candidates, found {len(candidates)}")
        if len(raw_ligands) != self.expected_count:
            raise RuntimeError(f"Expected {self.expected_count} raw PDBQT files, found {len(raw_ligands)}")

    def install_runtime(self) -> None:
        if self.args.skip_install:
            self.log("Skipping apt/pip install because --skip-install was used.")
            return
        self.log("")
        self.log("=== 2. Installing Colab runtime dependencies ===")
        self.run(["apt-get", "update", "-qq"], cwd=Path("/content"))
        self.run(
            [
                "apt-get",
                "install",
                "-y",
                "-qq",
                "git",
                "build-essential",
                "libboost-all-dev",
                "ocl-icd-opencl-dev",
                "opencl-headers",
                "clinfo",
            ],
            cwd=Path("/content"),
        )
        self.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"], cwd=Path("/content"))
        self.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "pandas",
                "numpy",
                "pyyaml",
                "matplotlib",
            ],
            cwd=Path("/content"),
        )
        rdkit_install = self.run(
            [sys.executable, "-m", "pip", "install", "-q", "rdkit"],
            cwd=Path("/content"),
            check=False,
        )
        if rdkit_install.returncode != 0:
            self.log("pip install rdkit failed. Trying rdkit-pypi fallback.")
            self.run([sys.executable, "-m", "pip", "install", "-q", "rdkit-pypi"], cwd=Path("/content"))
        try:
            import rdkit  # noqa: F401
        except Exception:
            self.log("rdkit import failed after installing rdkit. Trying rdkit-pypi fallback.")
            self.run([sys.executable, "-m", "pip", "install", "-q", "rdkit-pypi"], cwd=Path("/content"))
            import rdkit  # noqa: F401

    def check_gpu_opencl(self) -> None:
        self.log("")
        self.log("=== 3. Checking T4 GPU and OpenCL ===")
        nvidia = self.run(["nvidia-smi"], cwd=Path("/content"), check=False)
        if nvidia.returncode != 0:
            raise RuntimeError("nvidia-smi failed. In Colab choose Runtime > Change runtime type > T4 GPU.")

        clinfo = self.run(["clinfo"], cwd=Path("/content"), check=False)
        cltext = clinfo.stdout or ""
        if clinfo.returncode != 0:
            raise RuntimeError("clinfo failed. OpenCL is not available in this Colab runtime.")
        if re.search(r"Number of platforms\s+0", cltext):
            raise RuntimeError("OpenCL reports zero platforms. Restart Colab with T4 GPU runtime.")
        if "NVIDIA" not in cltext.upper():
            raise RuntimeError("OpenCL is present, but NVIDIA was not detected. Restart Colab with T4 GPU runtime.")
        self.log("GPU/OpenCL check passed.")

    def clean_previous_outputs(self) -> None:
        self.log("")
        self.log("=== 4. Cleaning old Vina-GPU outputs ===")
        raw_pose_dir = self.base / "06_docking" / "v5_3_model_guided" / "poses_gpu_raw"
        pose_dir = self.base / "06_docking" / "v5_3_model_guided" / "poses"
        raw_pose_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)
        for folder, pattern in [
            (raw_pose_dir, "*.pdbqt"),
            (pose_dir, "CAND_*_vina_out.pdbqt"),
            (self.scores_dir, "vina_gpu_2_1*.log"),
        ]:
            for path in folder.glob(pattern):
                if path.is_file():
                    path.unlink()
        for name in [
            "docking_scores_full_vina_gpu_2_1.csv",
            "pose_sanity_scores.csv",
            "vina_gpu_2_1_stdout.log",
        ]:
            path = self.scores_dir / name
            if path.exists():
                path.unlink()
        self.log("Old GPU docking outputs removed.")

    def sanitize_ligands(self) -> None:
        self.log("")
        self.log("=== 5. Sanitizing ligand PDBQT files ===")
        self.run(
            [
                sys.executable,
                "scripts/06_prepare_vina_gpu_ligands.py",
                "--base",
                ".",
                "--clean",
                "--expected-count",
                str(self.expected_count),
            ],
            cwd=self.base,
        )
        self.verify_clean_ligands()

    def verify_clean_ligands(self) -> None:
        clean_dir = self.base / "06_docking" / "v5_3_model_guided" / "ligands_pdbqt_vina_gpu_clean"
        clean_ligands = sorted(clean_dir.glob("CAND_*.pdbqt"))
        if len(clean_ligands) != self.expected_count:
            raise RuntimeError(f"Expected {self.expected_count} clean ligands, found {len(clean_ligands)}")
        bad: list[tuple[str, int, str]] = []
        for path in clean_ligands:
            for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if line.startswith(("MODEL", "ENDMDL")):
                    bad.append((path.name, line_no, "MODEL_TAG"))
                if line.startswith(("ATOM", "HETATM")):
                    atom_type = "".join(line.ljust(79)[77:79].split())
                    if atom_type not in VALID_AD_TYPES:
                        bad.append((path.name, line_no, atom_type))
        if bad:
            raise RuntimeError(f"Clean ligand validation failed. Sample: {bad[:10]}")
        self.log(f"Clean ligand validation passed: {len(clean_ligands)} files.")

    def prepare_vina_gpu(self) -> None:
        self.log("")
        self.log("=== 6. Building Vina-GPU 2.1 from source ===")
        if self.args.clean_vina_repo and self.vina_repo_dir.exists():
            shutil.rmtree(self.vina_repo_dir)
        if not self.vina_repo_dir.exists():
            self.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1.git",
                    str(self.vina_repo_dir),
                ],
                cwd=Path("/content"),
            )
        if not self.vina_dir.exists():
            raise FileNotFoundError(self.vina_dir)

        for kernel in self.vina_dir.glob("Kernel*_Opt.bin"):
            kernel.unlink()
        self.write_makefile(include_stdcppfs=True)
        build = self.run_bash("ulimit -s 8192 && make clean && make source -j2", cwd=self.vina_dir, check=False)
        if build.returncode != 0:
            self.log("First build failed. Retrying without -lstdc++fs.")
            self.write_makefile(include_stdcppfs=False)
            self.run_bash("ulimit -s 8192 && make clean && make source -j2", cwd=self.vina_dir)
        self.run(["chmod", "+x", str(self.binary)], cwd=self.vina_dir)
        if not self.binary.exists():
            raise FileNotFoundError(self.binary)
        ldd = self.run(["ldd", str(self.binary)], cwd=self.vina_dir, check=False)
        if "not found" in (ldd.stdout or ""):
            raise RuntimeError("Vina-GPU binary has missing shared libraries. See ldd output above.")
        self.log(f"Vina-GPU executable ready: {self.binary}")
        self.log(f"opencl_binary_path directory: {self.vina_dir}")

    def write_makefile(self, include_stdcppfs: bool) -> None:
        lib2 = "-lstdc++fs" if include_stdcppfs else ""
        makefile = f"""
WORK_DIR=$(shell pwd)
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
        (self.vina_dir / "Makefile").write_text(makefile.strip() + "\n", encoding="utf-8")

    def write_config(self, config_path: Path, ligand_dir: Path, output_dir: Path, thread: int) -> Path:
        receptor = self.base / "06_docking" / "receptor" / "jak2_prepared.pdbqt"
        grid_path = self.base / "06_docking" / "v5_3_model_guided" / "receptor" / "docking_grid_5AEP_QUP.json"
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        config_text = f"""
receptor = {receptor}
ligand_directory = {self.dir_arg(ligand_dir)}
output_directory = {self.dir_arg(output_dir)}
opencl_binary_path = {self.dir_arg(self.vina_dir)}
center_x = {grid["center"]["x"]}
center_y = {grid["center"]["y"]}
center_z = {grid["center"]["z"]}
size_x = {grid["size"]["x"]}
size_y = {grid["size"]["y"]}
size_z = {grid["size"]["z"]}
thread = {thread}
"""
        config_path.write_text(config_text.strip() + "\n", encoding="utf-8")
        return config_path

    @staticmethod
    def dir_arg(path: Path) -> str:
        text = str(path)
        return text if text.endswith("/") else text + "/"

    def run_vina(self, config_path: Path, log_path: Path) -> int:
        command = f"ulimit -s 8192 && {shlex.quote(str(self.binary))} --config {shlex.quote(str(config_path))}"
        self.log("")
        self.log(f"$ bash -lc {command!r}")
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(self.vina_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if self.log_handle:
                    self.log_handle.write(line)
                    self.log_handle.flush()
        return process.wait()

    def smoke_test(self) -> int:
        self.log("")
        self.log("=== 7. Running two-ligand smoke test ===")
        clean_dir = self.base / "06_docking" / "v5_3_model_guided" / "ligands_pdbqt_vina_gpu_clean"
        smoke_ligands = Path("/content/vina_gpu_smoke_ligands")
        smoke_out = Path("/content/vina_gpu_smoke_out")
        shutil.rmtree(smoke_ligands, ignore_errors=True)
        shutil.rmtree(smoke_out, ignore_errors=True)
        smoke_ligands.mkdir(parents=True)
        smoke_out.mkdir(parents=True)
        for ligand in sorted(clean_dir.glob("CAND_*.pdbqt"))[:2]:
            shutil.copy2(ligand, smoke_ligands / ligand.name)

        last_output = ""
        for thread in self.threads:
            shutil.rmtree(smoke_out, ignore_errors=True)
            smoke_out.mkdir(parents=True)
            config = self.write_config(Path(f"/content/vina_gpu_smoke_{thread}.txt"), smoke_ligands, smoke_out, thread)
            log_path = self.scores_dir / f"vina_gpu_2_1_smoke_thread_{thread}.log"
            rc = self.run_vina(config, log_path)
            raw_outputs = sorted(smoke_out.rglob("*.pdbqt"))
            parsed = sum(1 for path in raw_outputs if self.parse_score(path) is not None)
            last_output = log_path.read_text(encoding="utf-8", errors="ignore")[-5000:]
            self.log(f"Smoke thread {thread}: rc={rc}, outputs={len(raw_outputs)}, parsed_scores={parsed}")
            if rc == 0 and len(raw_outputs) == 2 and parsed == 2:
                Path("/content/vina_gpu_successful_thread.txt").write_text(str(thread), encoding="utf-8")
                self.log(f"Smoke test passed with thread {thread}.")
                return thread
        raise RuntimeError(f"Smoke test failed for all threads. Last output:\n{last_output}")

    def full_docking(self, preferred_thread: int) -> int:
        self.log("")
        self.log("=== 8. Running full 114-ligand Vina-GPU docking ===")
        clean_dir = self.base / "06_docking" / "v5_3_model_guided" / "ligands_pdbqt_vina_gpu_clean"
        raw_out = self.base / "06_docking" / "v5_3_model_guided" / "poses_gpu_raw"
        raw_out.mkdir(parents=True, exist_ok=True)
        thread_order = [preferred_thread] + [thread for thread in self.threads if thread < preferred_thread]
        last_output = ""
        for thread in thread_order:
            for path in raw_out.rglob("*.pdbqt"):
                if path.is_file():
                    path.unlink()
            config = self.write_config(Path(f"/content/vina_gpu_v5_3_thread_{thread}.txt"), clean_dir, raw_out, thread)
            log_path = self.scores_dir / f"vina_gpu_2_1_thread_{thread}.log"
            rc = self.run_vina(config, log_path)
            raw_outputs = sorted(raw_out.rglob("*.pdbqt"))
            parsed = sum(1 for path in raw_outputs if self.parse_score(path) is not None)
            last_output = log_path.read_text(encoding="utf-8", errors="ignore")[-5000:]
            self.log(f"Full thread {thread}: rc={rc}, outputs={len(raw_outputs)}, parsed_scores={parsed}")
            if rc == 0 and len(raw_outputs) == self.expected_count and parsed == self.expected_count:
                shutil.copy2(log_path, self.scores_dir / "vina_gpu_2_1_stdout.log")
                Path("/content/vina_gpu_final_thread.txt").write_text(str(thread), encoding="utf-8")
                self.log(f"Full docking completed with thread {thread}.")
                return thread
        raise RuntimeError(f"Full Vina-GPU failed for all threads. Last output:\n{last_output}")

    @staticmethod
    def parse_score(path: Path) -> float | None:
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = SCORE_RE.search(text)
        return float(match.group(1)) if match else None

    def parse_outputs(self) -> None:
        self.log("")
        self.log("=== 9. Parsing Vina-GPU outputs ===")
        import pandas as pd

        raw_pose_dir = self.base / "06_docking" / "v5_3_model_guided" / "poses_gpu_raw"
        pose_dir = self.base / "06_docking" / "v5_3_model_guided" / "poses"
        clean_dir = self.base / "06_docking" / "v5_3_model_guided" / "ligands_pdbqt_vina_gpu_clean"
        candidates_path = self.base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles.csv"
        grid_path = self.base / "06_docking" / "v5_3_model_guided" / "receptor" / "docking_grid_5AEP_QUP.json"
        manifest_path = self.scores_dir / "vina_command_manifest.csv"
        output_csv = self.scores_dir / "docking_scores_full_vina_gpu_2_1.csv"
        pose_dir.mkdir(parents=True, exist_ok=True)
        grid = json.loads(grid_path.read_text(encoding="utf-8"))
        candidates = pd.read_csv(candidates_path)
        smiles_by_id = dict(zip(candidates["candidate_id"].astype(str), candidates["canonical_smiles"].astype(str)))

        all_raw = sorted(raw_pose_dir.rglob("*.pdbqt"))
        rows = []
        for ligand in sorted(clean_dir.glob("CAND_*.pdbqt")):
            candidate_id = ligand.stem
            raw_pose = self.find_raw_pose(candidate_id, all_raw)
            final_pose = pose_dir / f"{candidate_id}_vina_out.pdbqt"
            best_score = None
            note = "missing_gpu_pose"
            if raw_pose is not None:
                shutil.copy2(raw_pose, final_pose)
                best_score = self.parse_score(raw_pose)
                note = "parsed_vina_gpu_2_1" if best_score is not None else "gpu_pose_no_score_found"
            rows.append(
                {
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
            )

        docking = pd.DataFrame(rows)
        docking.to_csv(output_csv, index=False)
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            scored_ids = set(docking.loc[docking["best_score"].notna(), "candidate_id"].astype(str))
            if "ligand_pdbqt" in manifest.columns:
                manifest["status"] = [
                    "completed" if self.path_stem_any(row["ligand_pdbqt"]) in scored_ids else "pending"
                    for _, row in manifest.iterrows()
                ]
                manifest.to_csv(manifest_path, index=False)

        parsed = int(docking["best_score"].notna().sum())
        self.log(f"Raw GPU pose files: {len(all_raw)}")
        self.log(f"Parsed GPU scores: {parsed} / {len(docking)}")
        self.log(str(docking.sort_values("best_score", na_position="last").head(10)[["candidate_id", "best_score", "notes"]]))
        if parsed != self.expected_count:
            raise RuntimeError(f"Expected {self.expected_count} parsed GPU scores, got {parsed}")

    @staticmethod
    def find_raw_pose(candidate_id: str, all_raw: list[Path]) -> Path | None:
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

    @staticmethod
    def path_stem_any(value: object) -> str:
        text = str(value)
        if "\\" in text:
            return PureWindowsPath(text).stem
        return Path(text).stem

    def postprocess(self) -> None:
        self.log("")
        self.log("=== 10. Running ADMET, ranking, pose sanity, and benchmark ===")
        commands = [
            [
                sys.executable,
                "scripts/05_score_admet_synthesis.py",
                "--base",
                ".",
                "--candidates-csv",
                "05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv",
                "--output-csv",
                "07_admet_synthesis/v5_3_model_guided_admet_scores.csv",
                "--flags-csv",
                "07_admet_synthesis/v5_3_model_guided_filter_flags.csv",
                "--notes-csv",
                "07_admet_synthesis/v5_3_model_guided_safety_proxy_notes.csv",
                "--stage",
                "V5_3_model_guided_vina_gpu_2_1_admet",
            ],
            [
                sys.executable,
                "scripts/07_rank_candidates.py",
                "--base",
                ".",
                "--candidates-csv",
                "05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv",
                "--docking-csv",
                "06_docking/v5_3_model_guided/scores/docking_scores_full_vina_gpu_2_1.csv",
                "--admet-csv",
                "07_admet_synthesis/v5_3_model_guided_admet_scores.csv",
                "--output-csv",
                "08_final_ranking/v5_3_model_guided_ranked_candidates.csv",
                "--stage",
                "V5_3_model_guided_vina_gpu_2_1_ranking",
            ],
            [
                sys.executable,
                "scripts/06_pose_sanity.py",
                "--base",
                ".",
                "--top-n",
                "10",
                "--ranking-csv",
                "08_final_ranking/v5_3_model_guided_ranked_candidates.csv",
                "--receptor-pdb",
                "06_docking/receptor/5AEP_receptor_clean.pdb",
                "--grid-json",
                "06_docking/v5_3_model_guided/receptor/docking_grid_5AEP_QUP.json",
                "--pose-dir",
                "06_docking/v5_3_model_guided/poses",
                "--output-csv",
                "06_docking/v5_3_model_guided/scores/pose_sanity_scores.csv",
                "--image-dir",
                "06_docking/v5_3_model_guided/images/pose_sanity",
                "--stage",
                "V5_3_model_guided_vina_gpu_2_1_pose_sanity",
            ],
            [
                sys.executable,
                "scripts/07_rank_candidates.py",
                "--base",
                ".",
                "--candidates-csv",
                "05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv",
                "--docking-csv",
                "06_docking/v5_3_model_guided/scores/docking_scores_full_vina_gpu_2_1.csv",
                "--pose-sanity-csv",
                "06_docking/v5_3_model_guided/scores/pose_sanity_scores.csv",
                "--admet-csv",
                "07_admet_synthesis/v5_3_model_guided_admet_scores.csv",
                "--output-csv",
                "08_final_ranking/v5_3_model_guided_ranked_candidates.csv",
                "--stage",
                "V5_3_model_guided_vina_gpu_2_1_ranking_with_pose",
            ],
            [sys.executable, "scripts/18_build_v5_3_benchmark_report.py", "--base", "."],
        ]
        for command in commands:
            self.run(command, cwd=self.base)

    def final_check(self) -> None:
        self.log("")
        self.log("=== 11. Final checks ===")
        import pandas as pd

        docking = pd.read_csv(self.scores_dir / "docking_scores_full_vina_gpu_2_1.csv")
        docking["best_score"] = pd.to_numeric(docking["best_score"], errors="coerce")
        parsed = int(docking["best_score"].notna().sum())
        self.log(f"Parsed scores: {parsed} / {len(docking)}")
        self.log(f"Best score: {docking['best_score'].min()}")
        self.log(f"Median score: {docking['best_score'].median()}")
        if parsed != self.expected_count:
            raise RuntimeError(f"Final parsed score count mismatch: {parsed}")

        ranked = pd.read_csv(self.base / "08_final_ranking" / "v5_3_model_guided_ranked_candidates.csv")
        keep = [c for c in ["rank", "candidate_id", "best_score", "pose_decision", "final_weighted_score", "decision"] if c in ranked.columns]
        self.log(str(ranked[keep].head(10)))

    def package_results(self, output_zip: Path) -> None:
        self.log("")
        self.log("=== 12. Packaging results ===")
        output_zip = Path(output_zip)
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        files = [
            "06_docking/v5_3_model_guided/scores/vina_command_manifest.csv",
            "06_docking/v5_3_model_guided/scores/vina_gpu_ligand_sanitization_summary.csv",
            "06_docking/v5_3_model_guided/scores/vina_gpu_colab_driver.log",
            "06_docking/v5_3_model_guided/scores/vina_gpu_2_1_stdout.log",
            "06_docking/v5_3_model_guided/scores/docking_scores_full_vina_gpu_2_1.csv",
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
        folders = [
            "06_docking/v5_3_model_guided/ligands_pdbqt_vina_gpu_clean",
            "06_docking/v5_3_model_guided/poses",
            "06_docking/v5_3_model_guided/poses_gpu_raw",
            "06_docking/v5_3_model_guided/images/pose_sanity",
        ]
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for rel in files:
                path = self.base / rel
                if path.exists():
                    z.write(path, rel)
                else:
                    self.log(f"Missing optional file: {rel}")
            for folder in folders:
                path = self.base / folder
                if path.exists():
                    for root, _dirs, names in os.walk(path):
                        for name in names:
                            full = Path(root) / name
                            z.write(full, full.relative_to(self.base))
        self.log(f"Saved result zip: {output_zip}")
        self.log(f"Size MB: {round(output_zip.stat().st_size / 1024 / 1024, 2)}")

    def package_failure_debug(self) -> None:
        debug_zip = Path(self.args.failure_debug_zip)
        debug_zip.parent.mkdir(parents=True, exist_ok=True)
        include = [
            self.log_path,
            self.scores_dir / "vina_gpu_ligand_sanitization_summary.csv",
            self.scores_dir / "docking_scores_full_vina_gpu_2_1.csv",
            self.scores_dir / "vina_gpu_2_1_stdout.log",
        ]
        include.extend(self.scores_dir.glob("vina_gpu_2_1*.log"))
        include.extend(Path("/content").glob("vina_gpu_*.txt"))
        with zipfile.ZipFile(debug_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in include:
                if path.exists() and path.is_file():
                    arcname = path.name if path.is_absolute() and str(path).startswith("/content") else str(path.relative_to(self.base))
                    z.write(path, arcname)
        self.log(f"Saved failure debug zip: {debug_zip}")

    def run_all(self) -> None:
        self.check_inputs()
        self.install_runtime()
        self.check_gpu_opencl()
        self.clean_previous_outputs()
        self.sanitize_ligands()
        self.prepare_vina_gpu()
        good_thread = self.smoke_test()
        self.full_docking(good_thread)
        self.parse_outputs()
        self.postprocess()
        self.final_check()
        self.package_results(Path(self.args.output_zip))
        self.log("")
        self.log("ALL DONE: Vina-GPU 2.1 docking completed and results zip was saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command Colab Vina-GPU 2.1 runner for EMD V5.3.")
    parser.add_argument("--base", required=True, help="Extracted EMD project directory.")
    parser.add_argument("--expected-count", type=int, default=114)
    parser.add_argument("--threads", default="4000,2048,1024")
    parser.add_argument("--vina-repo-dir", default="/content/Vina-GPU-2.1")
    parser.add_argument("--output-zip", default="/content/drive/MyDrive/EMD_V5_3_vina_gpu_2_1_results.zip")
    parser.add_argument("--failure-debug-zip", default="/content/drive/MyDrive/EMD_V5_3_vina_gpu_failure_debug.zip")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--clean-vina-repo", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with ColabVinaGpuRunner(args) as runner:
        try:
            runner.run_all()
        except Exception:
            runner.log("")
            runner.log("PIPELINE FAILED")
            runner.log(traceback.format_exc())
            try:
                runner.package_failure_debug()
            except Exception:
                runner.log("Could not package failure debug zip.")
                runner.log(traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
