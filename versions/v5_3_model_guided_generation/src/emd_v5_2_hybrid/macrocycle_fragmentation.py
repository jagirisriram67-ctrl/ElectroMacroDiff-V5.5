"""Macrocycle fragmentation utilities for V5.3 anchor/linker training.

The goal is a practical MED-like training table: real macrocycles are cut at
two single bonds in the largest ring to create an acyclic core with dummy
attachment atoms plus a linker fragment. This creates supervision for
attachment-site prediction and linker-size priors.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Any

from .chemistry import canonicalize_smiles, mol_from_smiles, mol_id_from_smiles, max_ring_size


def _rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install rdkit for macrocycle fragmentation") from exc
    return Chem


FRAGMENT_COLUMNS = [
    "fragment_id",
    "mol_id",
    "macrocycle_smiles",
    "core_smiles",
    "linker_smiles",
    "macrocycle_ring_size",
    "linker_heavy_atoms",
    "core_heavy_atoms",
    "linker_max_ring_size",
    "bond_a",
    "bond_b",
    "anchor_atom_indices",
    "notes",
]

ANCHOR_ATOM_COLUMNS = [
    "fragment_id",
    "mol_id",
    "atom_index",
    "atomic_num",
    "degree",
    "formal_charge",
    "is_aromatic",
    "is_ring",
    "total_h",
    "hybridization_sp",
    "hybridization_sp2",
    "hybridization_sp3",
    "mass",
    "gasteiger_charge",
    "num_rotatable_neighbors",
    "shortest_path_to_ring",
    "is_between_rings",
    "neighbor_heteroatom_count",
    "is_terminal_chain_atom",
    "local_connectivity_index",
    "ring_size_of_nearest_ring",
    "label_anchor",
]


def largest_ring_atoms(mol: Any) -> tuple[int, set[int]]:
    rings = list(mol.GetRingInfo().AtomRings())
    if not rings:
        return 0, set()
    largest = max(rings, key=len)
    return len(largest), set(int(idx) for idx in largest)


def eligible_largest_ring_bonds(mol: Any, ring_atoms: set[int]) -> list[int]:
    Chem = _rdkit()
    bond_ids: list[int] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.GetIsAromatic():
            continue
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        if begin in ring_atoms and end in ring_atoms and bond.IsInRing():
            bond_ids.append(int(bond.GetIdx()))
    return bond_ids


def _structural_heavy_atoms(fragment_mol: Any) -> int:
    return sum(1 for atom in fragment_mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _dummy_neighbor_original_indices(fragment_mol: Any) -> list[int]:
    indices: list[int] = []
    for atom in fragment_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() > 1:
                original_idx = neighbor.GetIntProp("original_atom_idx") if neighbor.HasProp("original_atom_idx") else -1
                if original_idx >= 0:
                    indices.append(int(original_idx))
    return sorted(set(indices))


def _annotate_original_atom_indices(mol: Any) -> None:
    for atom in mol.GetAtoms():
        atom.SetIntProp("original_atom_idx", int(atom.GetIdx()))


def _safe_gasteiger_charge(atom: Any) -> float:
    if not atom.HasProp("_GasteigerCharge"):
        return 0.0
    try:
        charge = float(atom.GetProp("_GasteigerCharge"))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(charge):
        return 0.0
    return max(-1.0, min(1.0, charge / 2.0))


def _distance_to_largest_ring(mol: Any, ring_atoms: set[int]) -> dict[int, int]:
    distances = {int(atom.GetIdx()): 99 for atom in mol.GetAtoms()}
    queue: deque[int] = deque()
    for atom_idx in ring_atoms:
        distances[int(atom_idx)] = 0
        queue.append(int(atom_idx))
    while queue:
        atom_idx = queue.popleft()
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = int(neighbor.GetIdx())
            if distances[neighbor_idx] > distances[atom_idx] + 1:
                distances[neighbor_idx] = distances[atom_idx] + 1
                queue.append(neighbor_idx)
    return distances


def _nearby_ring_ids(mol: Any, start_atom_idx: int, ring_memberships: dict[int, set[int]], max_depth: int = 3) -> set[int]:
    seen = {start_atom_idx}
    queue: deque[tuple[int, int]] = deque([(start_atom_idx, 0)])
    found: set[int] = set(ring_memberships.get(start_atom_idx, set()))
    while queue:
        atom_idx, depth = queue.popleft()
        if depth >= max_depth:
            continue
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = int(neighbor.GetIdx())
            if neighbor_idx in seen:
                continue
            seen.add(neighbor_idx)
            found.update(ring_memberships.get(neighbor_idx, set()))
            queue.append((neighbor_idx, depth + 1))
    return found


def _nearest_ring_size(atom_idx: int, rings: list[tuple[int, ...]], distances: dict[int, int]) -> int:
    best_distance = 999
    best_size = 0
    for ring in rings:
        ring_distance = min(distances.get(int(idx), 999) for idx in ring)
        if atom_idx in ring:
            ring_distance = 0
        if ring_distance < best_distance:
            best_distance = ring_distance
            best_size = len(ring)
    return best_size


def _rotatable_neighbor_count(atom: Any) -> int:
    Chem = _rdkit()
    count = 0
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() <= 1 or end.GetAtomicNum() <= 1:
            continue
        if begin.GetTotalDegree() <= 1 or end.GetTotalDegree() <= 1:
            continue
        count += 1
    return count


def anchor_atom_feature_row(
    mol: Any,
    atom: Any,
    ring_atoms: set[int],
    rings: list[tuple[int, ...]],
    ring_memberships: dict[int, set[int]],
    distance_to_ring: dict[int, int],
) -> dict[str, float]:
    hybridization = str(atom.GetHybridization())
    atom_idx = int(atom.GetIdx())
    heteroatomic_nums = {7, 8, 9, 15, 16, 17, 35, 53}
    neighbor_hetero = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() in heteroatomic_nums)
    local_connectivity = atom.GetTotalDegree() + sum(neighbor.GetTotalDegree() for neighbor in atom.GetNeighbors())
    nearby_rings = _nearby_ring_ids(mol, atom_idx, ring_memberships)
    shortest_to_ring = min(distance_to_ring.get(atom_idx, 99), 12)
    nearest_size = _nearest_ring_size(atom_idx, rings, distance_to_ring)
    return {
        "atomic_num": atom.GetAtomicNum() / 100.0,
        "degree": atom.GetTotalDegree() / 6.0,
        "formal_charge": atom.GetFormalCharge() / 3.0,
        "is_aromatic": float(atom.GetIsAromatic()),
        "is_ring": float(atom.IsInRing()),
        "total_h": atom.GetTotalNumHs() / 4.0,
        "hybridization_sp": float(hybridization == "SP"),
        "hybridization_sp2": float(hybridization == "SP2"),
        "hybridization_sp3": float(hybridization == "SP3"),
        "mass": atom.GetMass() / 250.0,
        "gasteiger_charge": _safe_gasteiger_charge(atom),
        "num_rotatable_neighbors": min(_rotatable_neighbor_count(atom), 4) / 4.0,
        "shortest_path_to_ring": shortest_to_ring / 12.0,
        "is_between_rings": float(len(nearby_rings) >= 2),
        "neighbor_heteroatom_count": min(neighbor_hetero, 4) / 4.0,
        "is_terminal_chain_atom": float(not atom.IsInRing() and atom.GetAtomicNum() > 1 and atom.GetTotalDegree() <= 1),
        "local_connectivity_index": min(local_connectivity, 36) / 36.0,
        "ring_size_of_nearest_ring": nearest_size / 24.0,
    }


def _fragment_pair_record(mol: Any, mol_id: str, smiles: str, bond_a: int, bond_b: int) -> dict | None:
    Chem = _rdkit()
    working = Chem.Mol(mol)
    _annotate_original_atom_indices(working)
    try:
        fragmented = Chem.FragmentOnBonds(working, [bond_a, bond_b], addDummies=True)
        fragments = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True)
    except Exception:
        return None
    if len(fragments) < 2:
        return None

    candidates = []
    for frag in fragments:
        heavy = _structural_heavy_atoms(frag)
        dummy_neighbors = _dummy_neighbor_original_indices(frag)
        if len(dummy_neighbors) >= 2:
            candidates.append((heavy, max_ring_size(frag), dummy_neighbors, frag))
    if len(candidates) < 2:
        return None

    linker_options = [
        item for item in candidates if 3 <= item[0] <= 9 and item[1] < 7
    ]
    if linker_options:
        linker_heavy, linker_ring, linker_anchors, linker = sorted(linker_options, key=lambda item: item[0])[0]
        core_options = [item for item in candidates if item[3] is not linker]
        if not core_options:
            return None
        core_heavy, _core_ring, core_anchors, core = sorted(core_options, key=lambda item: item[0], reverse=True)[0]
    else:
        candidates = sorted(candidates, key=lambda item: item[0])
        linker_heavy, linker_ring, linker_anchors, linker = candidates[0]
        core_heavy, _core_ring, core_anchors, core = candidates[-1]
        if not (3 <= linker_heavy <= 12):
            return None

    try:
        core_smiles = Chem.MolToSmiles(core, canonical=True, isomericSmiles=True)
        linker_smiles = Chem.MolToSmiles(linker, canonical=True, isomericSmiles=True)
    except Exception:
        return None
    if "*" not in core_smiles or "*" not in linker_smiles:
        return None

    ring_size, _ring_atoms = largest_ring_atoms(mol)
    anchor_indices = sorted(set(core_anchors))
    if len(anchor_indices) < 2:
        anchor_indices = sorted(set(linker_anchors))
    if len(anchor_indices) < 2:
        return None

    key = f"{smiles}|{core_smiles}|{linker_smiles}|{bond_a}|{bond_b}"
    return {
        "fragment_id": mol_id_from_smiles(key, "FRAG"),
        "mol_id": mol_id,
        "macrocycle_smiles": smiles,
        "core_smiles": core_smiles,
        "linker_smiles": linker_smiles,
        "macrocycle_ring_size": ring_size,
        "linker_heavy_atoms": linker_heavy,
        "core_heavy_atoms": core_heavy,
        "linker_max_ring_size": linker_ring,
        "bond_a": str(bond_a),
        "bond_b": str(bond_b),
        "anchor_atom_indices": ";".join(str(idx) for idx in anchor_indices),
        "notes": "largest_ring_two_bond_fragmentation",
    }


def fragment_macrocycle_record(
    row: dict,
    max_pairs_per_molecule: int = 20,
    min_ring_size: int = 12,
    max_ring_size_allowed: int = 24,
) -> list[dict]:
    smiles = canonicalize_smiles(str(row.get("canonical_smiles", "")))
    if not smiles:
        return []
    mol = mol_from_smiles(smiles)
    if mol is None:
        return []
    ring_size, ring_atoms = largest_ring_atoms(mol)
    if not (min_ring_size <= ring_size <= max_ring_size_allowed):
        return []
    bond_ids = eligible_largest_ring_bonds(mol, ring_atoms)
    records: list[dict] = []
    seen: set[str] = set()
    for bond_a, bond_b in combinations(bond_ids, 2):
        if len(records) >= max_pairs_per_molecule:
            break
        record = _fragment_pair_record(mol, str(row.get("mol_id", "")), smiles, bond_a, bond_b)
        if record is None:
            continue
        dedupe_key = record["core_smiles"] + "|" + record["linker_smiles"]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(record)
    return records


def extract_linker_smiles_from_smiles(
    smiles: str,
    mol_id: str = "",
    max_pairs_per_molecule: int = 50,
    min_ring_size: int = 12,
    max_ring_size_allowed: int = 24,
) -> list[str]:
    fragments = fragment_macrocycle_record(
        {
            "mol_id": mol_id,
            "canonical_smiles": smiles,
        },
        max_pairs_per_molecule=max_pairs_per_molecule,
        min_ring_size=min_ring_size,
        max_ring_size_allowed=max_ring_size_allowed,
    )
    return sorted(
        {
            str(fragment["linker_smiles"])
            for fragment in fragments
            if fragment.get("linker_smiles")
        }
    )


def build_fragment_linker_dataset(
    curated_csv: str | Path,
    output_csv: str | Path,
    max_pairs_per_molecule: int = 20,
    min_ring_size: int = 12,
    max_ring_size_allowed: int = 24,
) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas for macrocycle fragmentation") from exc

    frame = pd.read_csv(curated_csv)
    if "max_ring_size" in frame.columns:
        ring_sizes = pd.to_numeric(frame["max_ring_size"], errors="coerce")
        frame = frame[(ring_sizes >= min_ring_size) & (ring_sizes <= max_ring_size_allowed)].copy()
    elif "has_macrocycle_12_20" in frame.columns:
        mask = frame["has_macrocycle_12_20"].astype(str).str.lower().isin(["true", "1", "yes"])
        frame = frame[mask].copy()
    rows: list[dict] = []
    for record in frame.to_dict(orient="records"):
        rows.extend(
            fragment_macrocycle_record(
                record,
                max_pairs_per_molecule=max_pairs_per_molecule,
                min_ring_size=min_ring_size,
                max_ring_size_allowed=max_ring_size_allowed,
            )
        )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FRAGMENT_COLUMNS).to_csv(output, index=False)
    return output


def atom_training_rows_from_fragments(fragment_csv: str | Path) -> list[dict]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas for anchor training rows") from exc
    Chem = _rdkit()
    frame = pd.read_csv(fragment_csv)
    rows: list[dict] = []
    for record in frame.to_dict(orient="records"):
        mol = Chem.MolFromSmiles(str(record.get("macrocycle_smiles", "")))
        if mol is None:
            continue
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception:
            pass
        ring_size, ring_atoms = largest_ring_atoms(mol)
        rings = [tuple(int(idx) for idx in ring) for ring in mol.GetRingInfo().AtomRings()]
        ring_memberships: dict[int, set[int]] = {}
        for ring_id, ring in enumerate(rings):
            for atom_idx in ring:
                ring_memberships.setdefault(int(atom_idx), set()).add(ring_id)
        distance_to_ring = _distance_to_largest_ring(mol, ring_atoms)
        labels = {
            int(value)
            for value in str(record.get("anchor_atom_indices", "")).split(";")
            if value.strip().isdigit()
        }
        for atom in mol.GetAtoms():
            feature_row = anchor_atom_feature_row(mol, atom, ring_atoms, rings, ring_memberships, distance_to_ring)
            feature_row.update(
                {
                    "fragment_id": record.get("fragment_id", ""),
                    "mol_id": record.get("mol_id", ""),
                    "atom_index": int(atom.GetIdx()),
                    "label_anchor": float(int(atom.GetIdx()) in labels),
                }
            )
            rows.append(feature_row)
    return rows


def build_anchor_atom_dataset(fragment_csv: str | Path, output_csv: str | Path) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas for anchor atom dataset") from exc
    rows = atom_training_rows_from_fragments(fragment_csv)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ANCHOR_ATOM_COLUMNS).to_csv(output, index=False)
    return output
