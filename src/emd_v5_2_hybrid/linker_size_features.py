"""Feature extraction for V5.3 linker-size prediction.

The linker-size model is strongest when it sees the two proposed anchor atoms.
For training rows, those anchors are represented by the dummy atoms in the
fragmented core. For generation, they are the candidate atom pair being bridged.
"""

from __future__ import annotations

import math
from typing import Any


LINKER_FEATURE_COLUMNS = [
    "desired_ring_size",
    "core_heavy_atoms",
    "core_num_atoms",
    "core_num_rings",
    "core_max_ring_size",
    "num_dummy_atoms",
    "aromatic_fraction",
    "ring_atom_fraction",
    "carbon_fraction",
    "nitrogen_fraction",
    "oxygen_fraction",
    "sulfur_fraction",
    "halogen_fraction",
    "anchor_path_bonds",
    "anchor_path_3d_distance",
    "anchor_path_rotatable_fraction",
]


def _rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install rdkit for linker-size feature extraction") from exc
    return Chem


def _base_feature_map(desired_ring_size: int, core_heavy_atoms: int = 0) -> dict[str, float]:
    return {
        "desired_ring_size": desired_ring_size / 24.0,
        "core_heavy_atoms": core_heavy_atoms / 120.0,
        "core_num_atoms": 0.0,
        "core_num_rings": 0.0,
        "core_max_ring_size": 0.0,
        "num_dummy_atoms": 0.0,
        "aromatic_fraction": 0.0,
        "ring_atom_fraction": 0.0,
        "carbon_fraction": 0.0,
        "nitrogen_fraction": 0.0,
        "oxygen_fraction": 0.0,
        "sulfur_fraction": 0.0,
        "halogen_fraction": 0.0,
        "anchor_path_bonds": 0.0,
        "anchor_path_3d_distance": 0.0,
        "anchor_path_rotatable_fraction": 0.0,
    }


def _is_rotatable_path_bond(bond: Any) -> bool:
    Chem = _rdkit()
    if bond is None:
        return False
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    begin = bond.GetBeginAtom()
    end = bond.GetEndAtom()
    if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
        return False
    if begin.GetTotalDegree() <= 1 or end.GetTotalDegree() <= 1:
        return False
    return True


def anchor_path_feature_map(mol: Any, atom_a: int | None, atom_b: int | None) -> dict[str, float]:
    Chem = _rdkit()
    if atom_a is None or atom_b is None or atom_a == atom_b:
        return {
            "anchor_path_bonds": 0.0,
            "anchor_path_3d_distance": 0.0,
            "anchor_path_rotatable_fraction": 0.0,
        }
    try:
        path = tuple(int(idx) for idx in Chem.GetShortestPath(mol, int(atom_a), int(atom_b)))
    except Exception:
        path = ()
    path_bonds = max(len(path) - 1, 0)
    rotatable = 0
    if path_bonds:
        for begin, end in zip(path, path[1:]):
            rotatable += int(_is_rotatable_path_bond(mol.GetBondBetweenAtoms(int(begin), int(end))))
    distance = 0.0
    if mol.GetNumConformers() and path:
        try:
            conformer = mol.GetConformer()
            pos_a = conformer.GetAtomPosition(int(atom_a))
            pos_b = conformer.GetAtomPosition(int(atom_b))
            distance = math.sqrt((pos_a.x - pos_b.x) ** 2 + (pos_a.y - pos_b.y) ** 2 + (pos_a.z - pos_b.z) ** 2)
        except Exception:
            distance = 0.0
    return {
        "anchor_path_bonds": min(path_bonds, 24) / 24.0,
        "anchor_path_3d_distance": min(distance, 20.0) / 20.0,
        "anchor_path_rotatable_fraction": rotatable / max(path_bonds, 1),
    }


def _dummy_neighbor_anchor_pair(mol: Any) -> tuple[int | None, int | None]:
    Chem = _rdkit()
    anchors: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() > 0:
                anchors.append(int(neighbor.GetIdx()))
    anchors = sorted(set(anchors))
    if len(anchors) < 2:
        return None, None
    if len(anchors) == 2:
        return anchors[0], anchors[1]
    best_pair: tuple[int | None, int | None] = (anchors[0], anchors[1])
    best_path = -1
    for index, atom_a in enumerate(anchors):
        for atom_b in anchors[index + 1 :]:
            try:
                path_length = len(Chem.GetShortestPath(mol, int(atom_a), int(atom_b)))
            except Exception:
                path_length = 0
            if path_length > best_path:
                best_path = path_length
                best_pair = (atom_a, atom_b)
    return best_pair


def molecule_feature_map(
    mol: Any,
    desired_ring_size: int,
    core_heavy_atoms: int | None = None,
    atom_a: int | None = None,
    atom_b: int | None = None,
) -> dict[str, float]:
    row = _base_feature_map(desired_ring_size, core_heavy_atoms or int(mol.GetNumHeavyAtoms()))
    atoms = list(mol.GetAtoms())
    non_dummy = [atom for atom in atoms if atom.GetAtomicNum() > 0]
    atom_count = max(len(non_dummy), 1)
    try:
        rings = list(mol.GetRingInfo().AtomRings()) if mol.GetRingInfo() else []
    except Exception:
        rings = []
    max_ring = max((len(ring) for ring in rings), default=0)
    halogens = {9, 17, 35, 53}
    row.update(
        {
            "core_heavy_atoms": (core_heavy_atoms or int(mol.GetNumHeavyAtoms())) / 120.0,
            "core_num_atoms": len(non_dummy) / 120.0,
            "core_num_rings": len(rings) / 12.0,
            "core_max_ring_size": max_ring / 24.0,
            "num_dummy_atoms": sum(1 for atom in atoms if atom.GetAtomicNum() == 0) / 8.0,
            "aromatic_fraction": sum(1 for atom in non_dummy if atom.GetIsAromatic()) / atom_count,
            "ring_atom_fraction": sum(1 for atom in non_dummy if atom.IsInRing()) / atom_count,
            "carbon_fraction": sum(1 for atom in non_dummy if atom.GetAtomicNum() == 6) / atom_count,
            "nitrogen_fraction": sum(1 for atom in non_dummy if atom.GetAtomicNum() == 7) / atom_count,
            "oxygen_fraction": sum(1 for atom in non_dummy if atom.GetAtomicNum() == 8) / atom_count,
            "sulfur_fraction": sum(1 for atom in non_dummy if atom.GetAtomicNum() == 16) / atom_count,
            "halogen_fraction": sum(1 for atom in non_dummy if atom.GetAtomicNum() in halogens) / atom_count,
        }
    )
    if atom_a is None or atom_b is None:
        atom_a, atom_b = _dummy_neighbor_anchor_pair(mol)
    row.update(anchor_path_feature_map(mol, atom_a, atom_b))
    return row


def core_features(core_smiles: str, desired_ring_size: int, core_heavy_atoms: int) -> dict[str, float]:
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(str(core_smiles), sanitize=False)
    if mol is None:
        return _base_feature_map(desired_ring_size, core_heavy_atoms)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return molecule_feature_map(
        mol,
        desired_ring_size=desired_ring_size,
        core_heavy_atoms=core_heavy_atoms,
    )


def feature_vector(feature_map: dict[str, float], columns: list[str] | tuple[str, ...]) -> list[float]:
    return [float(feature_map.get(column, 0.0)) for column in columns]
