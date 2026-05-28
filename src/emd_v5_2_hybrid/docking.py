"""Docking preparation helpers and command manifest generation.

The actual docking engines are environment-sensitive in Colab. These helpers
standardize inputs, outputs, and resumable command manifests so the team can
use Vina/GNINA without mixing results by hand.
"""

from __future__ import annotations

import csv
import json
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schemas import DOCKING_SCORE_COLUMNS


def vina_command(
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str | Path,
    out_pdbqt: str | Path,
    log_path: str | Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int = 16,
    num_modes: int = 9,
    energy_range: int = 3,
) -> str:
    args = [
        "vina",
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--out",
        str(out_pdbqt),
        "--center_x",
        str(center[0]),
        "--center_y",
        str(center[1]),
        "--center_z",
        str(center[2]),
        "--size_x",
        str(size[0]),
        "--size_y",
        str(size[1]),
        "--size_z",
        str(size[2]),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--energy_range",
        str(energy_range),
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def write_vina_manifest(
    ligand_pdbqt_files: Iterable[str | Path],
    receptor_pdbqt: str | Path,
    output_csv: str | Path,
    pose_dir: str | Path,
    log_dir: str | Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int = 16,
) -> Path:
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(pose_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for ligand in ligand_pdbqt_files:
        ligand_path = Path(ligand)
        stem = ligand_path.stem
        pose_path = Path(pose_dir) / f"{stem}_vina_out.pdbqt"
        log_path = Path(log_dir) / f"{stem}_vina.log"
        status = "pending"
        if pose_path.exists() and log_path.exists():
            score = parse_vina_best_score(log_path.read_text(encoding="utf-8", errors="ignore"))
            if score is not None:
                status = "completed"
        rows.append(
            {
                "receptor_pdbqt": str(receptor_pdbqt),
                "ligand_pdbqt": str(ligand_path),
                "pose_pdbqt": str(pose_path),
                "log_path": str(log_path),
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "size_x": size[0],
                "size_y": size[1],
                "size_z": size[2],
                "exhaustiveness": exhaustiveness,
                "num_modes": 9,
                "energy_range": 3,
                "command": vina_command(
                    receptor_pdbqt,
                    ligand_path,
                    pose_path,
                    log_path,
                    center=center,
                    size=size,
                    exhaustiveness=exhaustiveness,
                ),
                "status": status,
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
                "exhaustiveness",
                "num_modes",
                "energy_range",
                "command",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def parse_vina_best_score(log_text: str) -> float | None:
    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("mode"):
            continue
        parts = stripped.split()
        if parts and parts[0].isdigit() and len(parts) >= 2:
            try:
                return float(parts[1])
            except ValueError:
                continue
    return None


def initialize_docking_scores(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(DOCKING_SCORE_COLUMNS)
    return output


EXCLUDED_HET_RESNAMES = {
    "HOH",
    "WAT",
    "DOD",
    "NA",
    "K",
    "CL",
    "CA",
    "MG",
    "MN",
    "ZN",
    "FE",
    "SO4",
    "PO4",
    "GOL",
    "EDO",
    "PTR",
    "SEP",
    "TPO",
}


def parse_pdb_hetatm_groups(pdb_path: str | Path) -> dict[str, list[tuple[float, float, float]]]:
    groups: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with Path(pdb_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in EXCLUDED_HET_RESNAMES:
                continue
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = f"{resname}:{chain}:{resseq}:{icode}"
            groups[key].append((x, y, z))
    return dict(groups)


def infer_grid_from_ligand(
    pdb_path: str | Path,
    padding: float = 8.0,
    minimum_size: float = 18.0,
    maximum_size: float = 26.0,
) -> dict:
    groups = parse_pdb_hetatm_groups(pdb_path)
    if not groups:
        raise ValueError(f"No non-excluded HETATM ligand groups found in {pdb_path}")
    ligand_id, coords = max(groups.items(), key=lambda item: len(item[1]))
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    zs = [coord[2] for coord in coords]
    center = {
        "x": round(sum(xs) / len(xs), 3),
        "y": round(sum(ys) / len(ys), 3),
        "z": round(sum(zs) / len(zs), 3),
    }
    raw_size = {
        "x": max(xs) - min(xs) + padding,
        "y": max(ys) - min(ys) + padding,
        "z": max(zs) - min(zs) + padding,
    }
    size = {
        axis: round(max(minimum_size, min(maximum_size, value)), 3)
        for axis, value in raw_size.items()
    }
    return {
        "source_pdb": str(pdb_path),
        "reference_ligand": ligand_id,
        "num_ligand_atoms": len(coords),
        "center": center,
        "size": size,
        "method": "centroid_and_extent_of_largest_nonexcluded_hetatm_ligand",
        "notes": "Verify visually before production docking.",
    }


def write_grid_json(grid: dict, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(grid, indent=2), encoding="utf-8")
    return output
