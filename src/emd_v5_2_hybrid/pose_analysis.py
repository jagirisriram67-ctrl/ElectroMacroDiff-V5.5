"""Pose sanity analysis and lightweight pose rendering."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


def parse_pdb_like_atoms(path: str | Path) -> list[dict]:
    atoms = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            residue = line[17:20].strip()
            chain = line[21].strip()
            residue_id = line[22:26].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            if not element:
                # PDBQT often stores atom type in the final token.
                tokens = line.split()
                element = tokens[-1] if tokens else atom_name[:1]
            atoms.append(
                {
                    "atom_name": atom_name,
                    "residue": residue,
                    "chain": chain,
                    "residue_id": residue_id,
                    "element": normalize_element(element),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
    return atoms


def normalize_element(element: str) -> str:
    element = "".join(ch for ch in str(element).strip() if ch.isalpha())
    if not element:
        return "X"
    if len(element) >= 2 and element[:2].upper() in {"CL", "BR", "NA", "MG", "CA", "ZN", "FE", "MN"}:
        return element[:2].upper().title()
    return element[0].upper()


def centroid(atoms: list[dict]) -> tuple[float, float, float]:
    if not atoms:
        return (float("nan"), float("nan"), float("nan"))
    return (
        sum(atom["x"] for atom in atoms) / len(atoms),
        sum(atom["y"] for atom in atoms) / len(atoms),
        sum(atom["z"] for atom in atoms) / len(atoms),
    )


def distance(a: dict | tuple[float, float, float], b: dict | tuple[float, float, float]) -> float:
    if isinstance(a, dict):
        ax, ay, az = a["x"], a["y"], a["z"]
    else:
        ax, ay, az = a
    if isinstance(b, dict):
        bx, by, bz = b["x"], b["y"], b["z"]
    else:
        bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def residue_label(atom: dict) -> str:
    return f"{atom['residue']}:{atom['chain']}:{atom['residue_id']}"


def analyze_pose(
    candidate_id: str,
    pose_pdbqt: str | Path,
    receptor_pdb: str | Path,
    grid_json: str | Path,
    docking_score: float | None = None,
) -> dict:
    ligand_atoms = parse_pdb_like_atoms(pose_pdbqt)
    receptor_atoms = parse_pdb_like_atoms(receptor_pdb)
    grid = json.loads(Path(grid_json).read_text(encoding="utf-8"))
    grid_center = (grid["center"]["x"], grid["center"]["y"], grid["center"]["z"])
    lig_centroid = centroid(ligand_atoms)
    contact_residues = set()
    contacts_4a = 0
    contacts_5a = 0
    clashes_2a = 0
    min_distance = float("inf")
    for lig_atom in ligand_atoms:
        for rec_atom in receptor_atoms:
            d = distance(lig_atom, rec_atom)
            if d < min_distance:
                min_distance = d
            if d <= 5.0:
                contacts_5a += 1
                contact_residues.add(residue_label(rec_atom))
            if d <= 4.0:
                contacts_4a += 1
            if d < 1.8:
                clashes_2a += 1

    centroid_offset = distance(lig_centroid, grid_center)
    # Conservative sanity score: contacts are good, large offsets and hard clashes are bad.
    score = 1.0
    if contacts_4a < 10:
        score -= 0.25
    if centroid_offset > 8.0:
        score -= 0.25
    if clashes_2a > 0:
        score -= min(0.45, 0.08 * clashes_2a)
    pose_score = round(max(0.0, min(1.0, score)), 6)
    return {
        "candidate_id": candidate_id,
        "pose_file": str(pose_pdbqt),
        "docking_score": docking_score,
        "num_ligand_atoms": len(ligand_atoms),
        "ligand_centroid_x": round(lig_centroid[0], 3),
        "ligand_centroid_y": round(lig_centroid[1], 3),
        "ligand_centroid_z": round(lig_centroid[2], 3),
        "grid_center_offset": round(centroid_offset, 3),
        "min_protein_distance": round(min_distance, 3) if math.isfinite(min_distance) else "",
        "contacts_within_4A": contacts_4a,
        "contacts_within_5A": contacts_5a,
        "unique_contact_residues_5A": len(contact_residues),
        "contact_residues_5A": ";".join(sorted(contact_residues)[:30]),
        "hard_clashes_lt_1_8A": clashes_2a,
        "pose_score": pose_score,
        "pose_decision": "pass" if pose_score >= 0.7 else "review",
    }


def render_pose_png(
    candidate_id: str,
    pose_pdbqt: str | Path,
    receptor_pdb: str | Path,
    output_png: str | Path,
    contact_cutoff: float = 5.0,
) -> Path:
    import matplotlib.pyplot as plt

    ligand_atoms = parse_pdb_like_atoms(pose_pdbqt)
    receptor_atoms = parse_pdb_like_atoms(receptor_pdb)
    nearby = []
    for rec_atom in receptor_atoms:
        if any(distance(rec_atom, lig_atom) <= contact_cutoff for lig_atom in ligand_atoms):
            nearby.append(rec_atom)

    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    if nearby:
        ax.scatter(
            [atom["x"] for atom in nearby],
            [atom["y"] for atom in nearby],
            [atom["z"] for atom in nearby],
            c="#8a8f98",
            s=16,
            alpha=0.35,
            label="JAK2 pocket atoms <=5A",
        )
    if ligand_atoms:
        colors = [atom_color(atom["element"]) for atom in ligand_atoms]
        ax.scatter(
            [atom["x"] for atom in ligand_atoms],
            [atom["y"] for atom in ligand_atoms],
            [atom["z"] for atom in ligand_atoms],
            c=colors,
            s=70,
            edgecolors="#111111",
            linewidths=0.4,
            label="Docked ligand",
        )
    ax.set_title(f"{candidate_id} docked pose")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    ax.view_init(elev=22, azim=42)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def atom_color(element: str) -> str:
    return {
        "C": "#4a5568",
        "N": "#2563eb",
        "O": "#dc2626",
        "S": "#ca8a04",
        "F": "#16a34a",
        "Cl": "#22c55e",
        "Br": "#92400e",
        "H": "#cbd5e1",
    }.get(element, "#6b7280")
