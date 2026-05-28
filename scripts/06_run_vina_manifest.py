from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_run


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Vina commands from vina_command_manifest.csv resumably.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vina-exe", default=None, help="Path/name of Vina executable. Defaults to tools/vina/vina.exe or PATH.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--stage", default="M5_vina_run")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else base / "06_docking" / "scores" / "vina_command_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Vina manifest: {manifest_path}")
    vina_exe = find_vina_executable(base, args.vina_exe)
    if vina_exe is None and not args.dry_run:
        raise FileNotFoundError("Vina executable was not found. Install Vina or run scripts/06_download_vina.py.")
    vina_exe = vina_exe or "vina"

    manifest = pd.read_csv(manifest_path)
    pending = manifest[manifest["status"].fillna("pending") != "completed"]
    if args.limit:
        pending = pending.head(args.limit)

    completed = 0
    failed = 0
    for index, row in pending.iterrows():
        row_dict = row.to_dict()
        args_list = row_to_vina_args(row_dict, vina_exe=vina_exe)
        command = str(row_dict.get("command", ""))
        if args.dry_run:
            print(" ".join(args_list) if args_list else command)
            continue
        if args_list:
            result = subprocess.run(args_list, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        log_path = Path(str(row_dict.get("log_path", "")))
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "COMMAND:\n"
                + (" ".join(args_list) if args_list else command)
                + "\n\nSTDOUT:\n"
                + result.stdout
                + "\n\nSTDERR:\n"
                + result.stderr,
                encoding="utf-8",
            )
        if result.returncode == 0:
            manifest.loc[index, "status"] = "completed"
            completed += 1
        else:
            manifest.loc[index, "status"] = "failed"
            failed += 1
        manifest.to_csv(manifest_path, index=False)

    register_run(
        base,
        stage=args.stage,
        status="dry_run" if args.dry_run else "completed",
        input_path=str(manifest_path),
        output_path=str(manifest_path),
        molecules_in=len(pending),
        molecules_out=completed,
        notes=f"failed={failed}",
    )
    print(f"Completed: {completed}; failed: {failed}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
