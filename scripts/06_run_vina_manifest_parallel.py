from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_run


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas to run the Vina manifest") from exc
    return pd


def normalize_statuses(frame):
    if "status" not in frame.columns:
        frame["status"] = "pending"
    frame["status"] = frame["status"].fillna("pending").replace({"running": "pending"})
    return frame


def find_vina_executable(base: Path, explicit: str | None = None) -> str | None:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return str(path)
        found = shutil.which(explicit)
        if found:
            return found
        return explicit
    project_vina = base / "tools" / "vina" / "vina.exe"
    if project_vina.exists():
        return str(project_vina)
    return shutil.which("vina")


def row_to_vina_args(row: dict, vina_exe: str = "vina") -> list[str] | None:
    required = [
        "receptor_pdbqt",
        "ligand_pdbqt",
        "pose_pdbqt",
        "log_path",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
    ]
    if any(key not in row or str(row.get(key, "")) == "" for key in required):
        return None
    return [
        vina_exe,
        "--receptor",
        str(row["receptor_pdbqt"]),
        "--ligand",
        str(row["ligand_pdbqt"]),
        "--out",
        str(row["pose_pdbqt"]),
        "--center_x",
        str(row["center_x"]),
        "--center_y",
        str(row["center_y"]),
        "--center_z",
        str(row["center_z"]),
        "--size_x",
        str(row["size_x"]),
        "--size_y",
        str(row["size_y"]),
        "--size_z",
        str(row["size_z"]),
        "--exhaustiveness",
        str(row.get("exhaustiveness", 16)),
        "--num_modes",
        str(row.get("num_modes", 9)),
        "--energy_range",
        str(row.get("energy_range", 3)),
    ]


def run_one(index: int, row: dict, vina_exe: str, cpu_per_job: int | None) -> tuple[int, str, str]:
    args_list = row_to_vina_args(row, vina_exe=vina_exe)
    if args_list and cpu_per_job and "--cpu" not in args_list:
        args_list.extend(["--cpu", str(cpu_per_job)])
    command = str(row.get("command", ""))
    if args_list:
        result = subprocess.run(args_list, capture_output=True, text=True)
        command_text = " ".join(args_list)
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        command_text = command
    log_path = Path(str(row.get("log_path", "")))
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND:\n"
            + command_text
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr,
            encoding="utf-8",
        )
    status = "completed" if result.returncode == 0 else "failed"
    return index, status, command_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vina manifest rows in a controlled parallel worker pool.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending rows to process in this invocation.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--cpu-per-job", type=int, default=1)
    parser.add_argument("--vina-exe", default=None)
    parser.add_argument("--stage", default="M5_vina_parallel_run")
    args = parser.parse_args()

    pd = _pandas()
    base = Path(args.base).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else base / "06_docking" / "scores" / "vina_command_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Vina manifest: {manifest_path}")
    vina_exe = find_vina_executable(base, args.vina_exe)
    if vina_exe is None:
        raise FileNotFoundError("Vina executable was not found. Install Vina or run scripts/06_download_vina.py.")

    manifest = normalize_statuses(pd.read_csv(manifest_path))
    pending = manifest[manifest["status"] != "completed"]
    if args.limit:
        pending = pending.head(args.limit)
    indices = list(pending.index)
    if not indices:
        print(f"No pending rows. Manifest: {manifest_path}")
        return

    workers = max(1, min(int(args.workers), len(indices)))
    print(f"Starting Vina parallel run: rows={len(indices)} workers={workers} cpu_per_job={args.cpu_per_job}")
    print(f"Already completed: {int((manifest['status'] == 'completed').sum())}; pending before chunk: {int((manifest['status'] != 'completed').sum())}")

    for index in indices:
        manifest.loc[index, "status"] = "running"
    manifest.to_csv(manifest_path, index=False)

    completed = 0
    failed = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, int(index), manifest.loc[index].to_dict(), vina_exe, args.cpu_per_job): index
            for index in indices
        }
        for future in as_completed(futures):
            index, status, _command = future.result()
            manifest.loc[index, "status"] = status
            if status == "completed":
                completed += 1
            else:
                failed += 1
            manifest.to_csv(manifest_path, index=False)
            total_completed = int((manifest["status"] == "completed").sum())
            total_remaining = int((manifest["status"] != "completed").sum())
            elapsed = time.time() - started
            print(
                f"chunk_done={completed + failed}/{len(indices)} "
                f"chunk_completed={completed} failed={failed} "
                f"total_completed={total_completed}/114 remaining={total_remaining} "
                f"elapsed_min={elapsed / 60:.1f}"
            )

    register_run(
        base,
        stage=args.stage,
        status="completed" if failed == 0 else "completed_with_failures",
        input_path=str(manifest_path),
        output_path=str(manifest_path),
        molecules_in=len(indices),
        molecules_out=completed,
        notes=f"parallel_workers={workers}; cpu_per_job={args.cpu_per_job}; failed={failed}",
    )
    print(f"Completed: {completed}; failed: {failed}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
