"""Ligand 3D SDF preparation for downstream docking conversion."""

from __future__ import annotations

import csv
import os
import site
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from .config import MissingDependencyError


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError("Install rdkit to prepare ligand SDF files") from exc
    return Chem, AllChem


def find_executable(names: list[str]) -> str | None:
    """Find a command on PATH or in common Python user script folders."""

    import shutil

    candidates: list[Path] = []
    for name in names:
        found = shutil.which(name)
        if found:
            return found
        candidates.append(Path(name))

    search_dirs = [
        Path(sys.executable).parent,
        Path(sys.executable).parent / "Scripts",
        Path(site.USER_BASE) / "Scripts",
        Path(site.USER_BASE) / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts",
        Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts",
    ]
    for directory in search_dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return str(path)
    return None


def run_command(args: list[str], log_path: str | Path) -> int:
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(args, capture_output=True, text=True)
    log.write_text(
        "COMMAND:\n"
        + " ".join(args)
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr,
        encoding="utf-8",
    )
    return int(result.returncode)


def smiles_to_3d_mol(smiles: str, random_seed: int = 42):
    Chem, AllChem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(
            mol,
            randomSeed=int(random_seed),
            useRandomCoords=True,
            maxAttempts=500,
        )
    if status != 0:
        return None
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=300)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=300)
        except Exception:
            pass
    return mol


def _write_single_ligand_sdf(task: tuple[dict, str, int]) -> dict:
    row, output_dir, random_seed = task
    Chem, _AllChem = _rdkit()
    out_dir = Path(output_dir)
    candidate_id = str(row.get("candidate_id", "candidate"))
    smiles = str(row.get("smiles", row.get("canonical_smiles", "")))
    single_path = out_dir / f"{candidate_id}.sdf"
    try:
        mol = smiles_to_3d_mol(smiles, random_seed=random_seed)
        if mol is None:
            return {
                "candidate_id": candidate_id,
                "smiles": smiles,
                "sdf_path": "",
                "status": "failed_embed",
            }
        mol.SetProp("_Name", candidate_id)
        for key, value in row.items():
            if value is not None:
                mol.SetProp(str(key), str(value))
        single_writer = Chem.SDWriter(str(single_path))
        single_writer.write(mol)
        single_writer.close()
        return {
            "candidate_id": candidate_id,
            "smiles": smiles,
            "sdf_path": str(single_path),
            "status": "prepared",
        }
    except Exception as exc:
        return {
            "candidate_id": candidate_id,
            "smiles": smiles,
            "sdf_path": "",
            "status": f"failed_exception:{type(exc).__name__}",
        }


def write_ligand_sdf_batch(
    candidate_rows: Iterable[dict],
    output_dir: str | Path,
    combined_sdf: str | Path,
    random_seed: int = 42,
    workers: int = 1,
) -> dict:
    _rdkit()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = Path(combined_sdf)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "docking_input_manifest.csv"
    rows = list(candidate_rows)
    worker_count = max(int(workers), 1)
    tasks = [(row, str(out_dir), int(random_seed)) for row in rows]
    if worker_count == 1 or len(tasks) <= 1:
        manifest_rows = [_write_single_ligand_sdf(task) for task in tasks]
    else:
        indexed_results: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_write_single_ligand_sdf, task): index for index, task in enumerate(tasks)}
            for future in as_completed(futures):
                indexed_results[futures[future]] = future.result()
        manifest_rows = [indexed_results[index] for index in range(len(tasks))]

    with combined_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            sdf_path = Path(str(row.get("sdf_path", "")))
            if sdf_path.exists():
                handle.write(sdf_path.read_text(encoding="utf-8", errors="ignore"))

    prepared = sum(1 for row in manifest_rows if row["status"] == "prepared")
    failed = len(manifest_rows) - prepared
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=["candidate_id", "smiles", "sdf_path", "status"])
        csv_writer.writeheader()
        csv_writer.writerows(manifest_rows)
    return {
        "prepared": prepared,
        "failed": failed,
        "combined_sdf": str(combined_path),
        "manifest": str(manifest_path),
        "workers": worker_count,
    }


def _prepare_one_ligand_pdbqt(task: tuple[str, str, str, list[str]]) -> dict:
    sdf_text, output_dir, log_dir, command_prefix = task
    sdf = Path(sdf_text)
    out_dir = Path(output_dir)
    logs = Path(log_dir)
    pdbqt = out_dir / f"{sdf.stem}.pdbqt"
    log = logs / f"{sdf.stem}_mk_prepare_ligand.log"
    if pdbqt.exists() and pdbqt.stat().st_size > 0:
        status = "already_prepared"
        returncode = 0
    else:
        args = [*command_prefix, "-i", str(sdf), "-o", str(pdbqt)]
        returncode = run_command(args, log)
        status = "prepared" if returncode == 0 and pdbqt.exists() and pdbqt.stat().st_size > 0 else "failed"
    return {
        "candidate_id": sdf.stem,
        "sdf_path": str(sdf),
        "pdbqt_path": str(pdbqt),
        "log_path": str(log),
        "status": status,
        "returncode": returncode,
    }


def prepare_ligand_pdbqt_batch(
    sdf_dir: str | Path,
    output_dir: str | Path,
    log_dir: str | Path,
    limit: int | None = None,
    workers: int = 1,
) -> dict:
    executable = find_executable(
        ["mk_prepare_ligand.py", "mk_prepare_ligand", "mk_prepare_ligand.exe"]
    )
    command_prefix = [executable] if executable else [sys.executable, "-m", "meeko.cli.mk_prepare_ligand"]

    sdf_files = sorted(Path(sdf_dir).glob("CAND_*.sdf"))
    if limit:
        sdf_files = sdf_files[:limit]
    out_dir = Path(output_dir)
    logs = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    worker_count = max(int(workers), 1)
    tasks = [(str(sdf), str(out_dir), str(logs), command_prefix) for sdf in sdf_files]
    if worker_count == 1 or len(tasks) <= 1:
        manifest_rows = [_prepare_one_ligand_pdbqt(task) for task in tasks]
    else:
        indexed_results: dict[int, dict] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_prepare_one_ligand_pdbqt, task): index for index, task in enumerate(tasks)}
            for future in as_completed(futures):
                indexed_results[futures[future]] = future.result()
        manifest_rows = [indexed_results[index] for index in range(len(tasks))]

    prepared = sum(1 for row in manifest_rows if row["status"] == "prepared")
    skipped = sum(1 for row in manifest_rows if row["status"] == "already_prepared")
    failed = sum(1 for row in manifest_rows if row["status"] == "failed")

    manifest_path = out_dir / "ligand_pdbqt_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_id", "sdf_path", "pdbqt_path", "log_path", "status", "returncode"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "executable": " ".join(command_prefix),
        "prepared": prepared,
        "failed": failed,
        "skipped": skipped,
        "total": len(sdf_files),
        "manifest": str(manifest_path),
        "workers": worker_count,
    }


def clean_receptor_pdb(input_pdb: str | Path, output_pdb: str | Path) -> Path:
    """Write a docking-oriented receptor PDB.

    Keeps standard ATOM records, removes HETATM ligands/waters/ions, keeps only
    the first/default alternate location, and rewrites accepted altloc labels as
    blank so Meeko does not stop on unresolved altloc choices.
    """

    output = Path(output_pdb)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept_lines: list[str] = []
    accepted_altlocs = {" ", "", "A", "1"}
    with Path(input_pdb).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6]
            if record == "ATOM  ":
                altloc = line[16]
                if altloc not in accepted_altlocs:
                    continue
                if altloc != " ":
                    line = line[:16] + " " + line[17:]
                kept_lines.append(line)
            elif record in {"TER   ", "END   "}:
                kept_lines.append(line)
    if not kept_lines or not any(line.startswith("ATOM") for line in kept_lines):
        raise ValueError(f"No receptor ATOM records retained from {input_pdb}")
    if not kept_lines[-1].startswith("END"):
        kept_lines.append("END\n")
    output.write_text("".join(kept_lines), encoding="utf-8")
    return output


def prepare_receptor_pdbqt(
    receptor_pdb: str | Path,
    output_pdbqt: str | Path,
    log_path: str | Path,
    center: dict,
    size: dict,
    clean_input: bool = True,
) -> dict:
    executable = find_executable(
        ["mk_prepare_receptor.py", "mk_prepare_receptor", "mk_prepare_receptor.exe"]
    )
    command_prefix = [executable] if executable else [sys.executable, "-m", "meeko.cli.mk_prepare_receptor"]

    output = Path(output_pdbqt)
    output.parent.mkdir(parents=True, exist_ok=True)
    receptor_input = Path(receptor_pdb)
    cleaned_pdb = ""
    if clean_input:
        receptor_input = clean_receptor_pdb(
            receptor_pdb,
            output.parent / f"{Path(receptor_pdb).stem}_receptor_clean.pdb",
        )
        cleaned_pdb = str(receptor_input)
    basename = output.with_suffix("")
    vina_box = output.parent / "vina_box_5AEP_QUP.txt"
    args = [
        *command_prefix,
        "--read_pdb",
        str(receptor_input),
        "-o",
        str(basename),
        "-p",
        str(output),
        "-v",
        str(vina_box),
        "--box_center",
        str(center["x"]),
        str(center["y"]),
        str(center["z"]),
        "--box_size",
        str(size["x"]),
        str(size["y"]),
        str(size["z"]),
        "-a",
    ]
    returncode = run_command(args, log_path)
    status = "prepared" if returncode == 0 and output.exists() and output.stat().st_size > 0 else "failed"
    return {
        "executable": " ".join(command_prefix),
        "status": status,
        "returncode": returncode,
        "pdbqt": str(output),
        "cleaned_pdb": cleaned_pdb,
        "vina_box": str(vina_box),
        "log_path": str(log_path),
    }
