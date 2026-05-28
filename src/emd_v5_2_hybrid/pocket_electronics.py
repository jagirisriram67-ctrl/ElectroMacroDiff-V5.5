"""Pocket electronic-fit scoring for docked JAK2 candidates.

This is a practical, Colab-safe proxy for pocket complementarity. It does not
claim quantum electron density. It uses receptor/ligand PDBQT partial charges,
atom/residue types, distances, and simple interaction rules to score how well a
docked ligand pose fits the binding pocket's electronic and contact pattern.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .pose_analysis import distance, normalize_element, parse_pdb_like_atoms, residue_label


HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TYR", "TRP", "PRO", "CYS"}
AROMATIC_RESIDUES = {"PHE", "TYR", "TRP", "HIS"}
POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}
NEGATIVE_RESIDUES = {"ASP", "GLU"}
ELECTRONEGATIVE_ELEMENTS = {"N", "O", "S", "F", "Cl", "Br"}
HYDROPHOBIC_ELEMENTS = {"C", "F", "Cl", "Br", "I", "S"}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def parse_pdbqt_atoms(path: str | Path) -> list[dict[str, Any]]:
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
            tokens = line.split()
            atom_type = tokens[-1] if tokens else ""
            charge = _safe_float(tokens[-2], 0.0) if len(tokens) >= 2 else 0.0
            atom_name = line[12:16].strip()
            residue = line[17:20].strip()
            chain = line[21].strip()
            residue_id = line[22:26].strip()
            element = normalize_element(atom_type or (line[76:78].strip() if len(line) >= 78 else atom_name[:1]))
            atoms.append(
                {
                    "atom_name": atom_name,
                    "residue": residue,
                    "chain": chain,
                    "residue_id": residue_id,
                    "element": element,
                    "atom_type": atom_type,
                    "charge": charge,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
    return atoms


def load_atoms_with_charge(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".pdbqt":
        atoms = parse_pdbqt_atoms(path)
        if atoms:
            return atoms
    atoms = parse_pdb_like_atoms(path)
    for atom in atoms:
        atom["charge"] = residue_atom_charge_proxy(atom)
        atom["atom_type"] = atom.get("element", "")
    return atoms


def residue_atom_charge_proxy(atom: dict[str, Any]) -> float:
    residue = str(atom.get("residue", "")).upper()
    name = str(atom.get("atom_name", "")).upper()
    element = str(atom.get("element", "")).title()
    if residue in NEGATIVE_RESIDUES and (name.startswith("OD") or name.startswith("OE")):
        return -0.6
    if residue in POSITIVE_RESIDUES and element == "N":
        return 0.45
    if element == "O":
        return -0.25
    if element == "N":
        return -0.10
    if element == "H":
        return 0.15
    return 0.0


def is_hbond_acceptor(atom: dict[str, Any]) -> bool:
    element = str(atom.get("element", "")).title()
    charge = _safe_float(atom.get("charge"), 0.0)
    atom_type = str(atom.get("atom_type", "")).upper()
    return element in {"O", "N", "S", "F"} and charge <= 0.15 and not atom_type.startswith("HD")


def is_hbond_donor(atom: dict[str, Any]) -> bool:
    element = str(atom.get("element", "")).title()
    atom_type = str(atom.get("atom_type", "")).upper()
    charge = _safe_float(atom.get("charge"), 0.0)
    residue = str(atom.get("residue", "")).upper()
    if atom_type.startswith("HD") or element == "H":
        return True
    if element in {"N", "O", "S"} and charge >= -0.25:
        return True
    return residue in POSITIVE_RESIDUES and element == "N"


def is_hydrophobic_atom(atom: dict[str, Any]) -> bool:
    element = str(atom.get("element", "")).title()
    residue = str(atom.get("residue", "")).upper()
    return element in HYDROPHOBIC_ELEMENTS and (residue in HYDROPHOBIC_RESIDUES or residue == "UNL")


def is_aromatic_like(atom: dict[str, Any]) -> bool:
    atom_type = str(atom.get("atom_type", "")).upper()
    residue = str(atom.get("residue", "")).upper()
    element = str(atom.get("element", "")).title()
    return atom_type == "A" or (residue in AROMATIC_RESIDUES and element == "C")


def nearby_receptor_atoms(ligand_atoms: list[dict], receptor_atoms: list[dict], cutoff: float = 6.0) -> list[dict]:
    nearby = []
    seen = set()
    for rec_atom in receptor_atoms:
        if any(distance(rec_atom, lig_atom) <= cutoff for lig_atom in ligand_atoms):
            key = (rec_atom.get("atom_name"), rec_atom.get("residue"), rec_atom.get("chain"), rec_atom.get("residue_id"), rec_atom.get("x"), rec_atom.get("y"), rec_atom.get("z"))
            if key not in seen:
                seen.add(key)
                nearby.append(rec_atom)
    return nearby


def pocket_profile_from_pose(ligand_atoms: list[dict], receptor_atoms: list[dict], cutoff: float = 6.0) -> dict[str, Any]:
    pocket_atoms = nearby_receptor_atoms(ligand_atoms, receptor_atoms, cutoff=cutoff)
    residue_counts: dict[str, int] = {}
    charge_sum = 0.0
    hydrophobic_atoms = 0
    donor_atoms = 0
    acceptor_atoms = 0
    aromatic_atoms = 0
    for atom in pocket_atoms:
        residue_counts[residue_label(atom)] = residue_counts.get(residue_label(atom), 0) + 1
        charge_sum += _safe_float(atom.get("charge"), 0.0)
        hydrophobic_atoms += int(is_hydrophobic_atom(atom))
        donor_atoms += int(is_hbond_donor(atom))
        acceptor_atoms += int(is_hbond_acceptor(atom))
        aromatic_atoms += int(is_aromatic_like(atom))
    top_residues = sorted(residue_counts.items(), key=lambda item: (-item[1], item[0]))[:25]
    return {
        "pocket_atom_count": len(pocket_atoms),
        "pocket_net_charge_proxy": round(charge_sum, 4),
        "hydrophobic_atom_count": hydrophobic_atoms,
        "hbond_donor_atom_count": donor_atoms,
        "hbond_acceptor_atom_count": acceptor_atoms,
        "aromatic_atom_count": aromatic_atoms,
        "top_contact_residues": [{"residue": residue, "atom_contacts": count} for residue, count in top_residues],
    }


def _saturating_score(value: float, half_saturation: float) -> float:
    value = max(0.0, float(value))
    return value / (value + max(half_saturation, 1e-9))


def score_pocket_electronic_fit(
    candidate_id: str,
    pose_pdbqt: str | Path,
    receptor_path: str | Path,
    docking_score: float | None = None,
    contact_cutoff: float = 5.0,
) -> dict[str, Any]:
    ligand_atoms = load_atoms_with_charge(pose_pdbqt)
    receptor_atoms = load_atoms_with_charge(receptor_path)
    if not ligand_atoms:
        raise ValueError(f"No ligand atoms parsed from {pose_pdbqt}")
    if not receptor_atoms:
        raise ValueError(f"No receptor atoms parsed from {receptor_path}")

    pocket_atoms = nearby_receptor_atoms(ligand_atoms, receptor_atoms, cutoff=max(contact_cutoff, 6.0))
    electro_favorable = 0.0
    electro_unfavorable = 0.0
    hbond_pairs = 0
    hydrophobic_contacts = 0
    aromatic_contacts = 0
    polar_contacts = 0
    close_contacts = 0
    clashes = 0
    min_distance = float("inf")
    contact_residues: set[str] = set()

    for lig_atom in ligand_atoms:
        q_lig = _safe_float(lig_atom.get("charge"), 0.0)
        for rec_atom in pocket_atoms:
            d = distance(lig_atom, rec_atom)
            if d < min_distance:
                min_distance = d
            if d > contact_cutoff:
                continue
            close_contacts += 1
            contact_residues.add(residue_label(rec_atom))
            q_rec = _safe_float(rec_atom.get("charge"), 0.0)
            product = q_lig * q_rec
            if d >= 1.8:
                if product < 0:
                    electro_favorable += (-product) / (d * d)
                elif product > 0:
                    electro_unfavorable += product / (d * d)
            if d < 1.8:
                clashes += 1
            if 2.3 <= d <= 3.6:
                donor_acceptor = (
                    is_hbond_donor(lig_atom)
                    and is_hbond_acceptor(rec_atom)
                    or is_hbond_donor(rec_atom)
                    and is_hbond_acceptor(lig_atom)
                )
                if donor_acceptor:
                    hbond_pairs += 1
            if d <= 4.6 and is_hydrophobic_atom(lig_atom) and is_hydrophobic_atom(rec_atom):
                hydrophobic_contacts += 1
            if d <= 5.2 and is_aromatic_like(lig_atom) and is_aromatic_like(rec_atom):
                aromatic_contacts += 1
            if d <= 4.0 and (lig_atom.get("element") in ELECTRONEGATIVE_ELEMENTS or rec_atom.get("element") in ELECTRONEGATIVE_ELEMENTS):
                polar_contacts += 1

    electro_balance = electro_favorable / max(electro_favorable + electro_unfavorable, 1e-9)
    electro_strength_score = _saturating_score(electro_favorable, 0.12)
    electrostatic_score = 0.65 * electro_balance + 0.35 * electro_strength_score
    hbond_score = _saturating_score(hbond_pairs, 3.0)
    hydrophobic_score = _saturating_score(hydrophobic_contacts, 35.0)
    aromatic_score = _saturating_score(aromatic_contacts, 8.0)
    contact_score = _saturating_score(close_contacts, 80.0)
    clash_penalty = min(0.45, 0.045 * clashes)
    pocket_fit_score = (
        0.30 * electrostatic_score
        + 0.20 * hbond_score
        + 0.20 * hydrophobic_score
        + 0.10 * aromatic_score
        + 0.20 * contact_score
        - clash_penalty
    )
    pocket_fit_score = max(0.0, min(1.0, pocket_fit_score))
    profile = pocket_profile_from_pose(ligand_atoms, receptor_atoms, cutoff=6.0)
    return {
        "candidate_id": candidate_id,
        "pose_file": str(pose_pdbqt),
        "receptor_file": str(receptor_path),
        "docking_score": docking_score,
        "num_ligand_atoms": len(ligand_atoms),
        "pocket_atom_count": profile["pocket_atom_count"],
        "min_protein_distance": round(min_distance, 3) if math.isfinite(min_distance) else "",
        "close_contacts": int(close_contacts),
        "unique_contact_residues": int(len(contact_residues)),
        "hbond_opportunity_pairs": int(hbond_pairs),
        "hydrophobic_contacts": int(hydrophobic_contacts),
        "aromatic_contacts": int(aromatic_contacts),
        "polar_contacts": int(polar_contacts),
        "hard_clashes_lt_1_8A": int(clashes),
        "electrostatic_favorable": round(electro_favorable, 6),
        "electrostatic_unfavorable": round(electro_unfavorable, 6),
        "electrostatic_score": round(electrostatic_score, 6),
        "hbond_score": round(hbond_score, 6),
        "hydrophobic_score": round(hydrophobic_score, 6),
        "aromatic_score": round(aromatic_score, 6),
        "contact_score": round(contact_score, 6),
        "pocket_electronic_fit_score": round(pocket_fit_score, 6),
        "pocket_fit_decision": "pass" if pocket_fit_score >= 0.58 and clashes == 0 else "review",
        "contact_residues": ";".join(sorted(contact_residues)[:40]),
    }


def write_pocket_profile(
    output_json: str | Path,
    receptor_path: str | Path,
    example_pose: str | Path,
) -> Path:
    ligand_atoms = load_atoms_with_charge(example_pose)
    receptor_atoms = load_atoms_with_charge(receptor_path)
    profile = pocket_profile_from_pose(ligand_atoms, receptor_atoms, cutoff=6.0)
    profile["receptor_file"] = str(receptor_path)
    profile["example_pose"] = str(example_pose)
    profile["note"] = (
        "Proxy pocket electronic profile from PDBQT partial charges and residue/atom classes; "
        "not a quantum electron-density map."
    )
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return output
