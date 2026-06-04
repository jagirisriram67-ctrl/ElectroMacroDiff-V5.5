"""RDKit and SELFIES baseline generation for the hybrid candidate module."""

from __future__ import annotations

import random
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Iterable

from .chemistry import canonicalize_smiles, mol_from_smiles, mol_id_from_smiles, murcko_scaffold_smiles, summarize_molecule
from .schemas import GENERATED_CANDIDATE_COLUMNS


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install rdkit for baseline generation") from exc
    return Chem, AllChem


def _selfies():
    try:
        import selfies
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install selfies for SELFIES mutation baseline") from exc
    return selfies


SELFIES_ALPHABET = [
    "[C]",
    "[N]",
    "[O]",
    "[S]",
    "[F]",
    "[Cl]",
    "[Branch1]",
    "[Ring1]",
    "[=C]",
    "[=N]",
]

LEGACY_LINKER_CHEMOTYPES = [
    "alkyl",
    "oxa",
    "aza",
    "thioether",
    "oxa_aza",
]

DIVERSE_LINKER_CHEMOTYPES = LEGACY_LINKER_CHEMOTYPES + [
    "dioxa",
    "diaza",
    "dithio",
    "aza_oxa",
    "oxa_thio",
    "aza_thio",
]

LINKER_CHEMOTYPES = list(LEGACY_LINKER_CHEMOTYPES)


def linker_chemotypes(chemotype_set: str = "legacy") -> list[str]:
    if chemotype_set == "legacy":
        return list(LEGACY_LINKER_CHEMOTYPES)
    if chemotype_set == "diverse":
        return list(DIVERSE_LINKER_CHEMOTYPES)
    raise ValueError(f"Unsupported chemotype set: {chemotype_set}")


def candidate_record(
    smiles: str,
    source_generator: str,
    parent_mol_id: str = "",
    training_inchikeys: set[str] | None = None,
    note: str = "",
) -> dict | None:
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    summary = summarize_molecule(canonical)
    if summary is None:
        return None
    mol = mol_from_smiles(canonical)
    scaffold = murcko_scaffold_smiles(mol) if mol is not None else ""
    training_inchikeys = training_inchikeys or set()
    return {
        "candidate_id": mol_id_from_smiles(canonical, "CAND"),
        "source_generator": source_generator,
        "parent_mol_id": parent_mol_id,
        "canonical_smiles": canonical,
        "inchikey": summary.inchikey,
        "valid_rdkit": True,
        "unique_flag": True,
        "novel_flag": summary.inchikey not in training_inchikeys,
        "mw": summary.mw,
        "logp": summary.logp,
        "tpsa": summary.tpsa,
        "hbd": summary.hbd,
        "hba": summary.hba,
        "rotatable_bonds": summary.rotatable_bonds,
        "qed": summary.qed,
        "sa_score": summary.sa_score,
        "murcko_scaffold": scaffold,
        "max_ring_size": summary.max_ring_size,
        "has_macrocycle_12_20": summary.has_macrocycle_12_20,
        "has_constrained_ring_8_11": summary.has_constrained_ring_8_11,
        "passes_basic_filters": summary.passes_basic_filters,
        "generation_notes": note,
    }


def scaffold_cluster_id(scaffold: str, fallback_smiles: str) -> str:
    key = scaffold or fallback_smiles
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"scaffold_{digest}"


def mutate_selfies_smiles(smiles: str, rng: random.Random, max_mutations: int = 3) -> str | None:
    selfies = _selfies()
    try:
        encoded = selfies.encoder(smiles)
        tokens = list(selfies.split_selfies(encoded))
    except Exception:
        return None
    if not tokens:
        return None
    for _ in range(rng.randint(1, max_mutations)):
        action = rng.choice(["replace", "insert", "delete"])
        index = rng.randrange(len(tokens))
        if action == "replace":
            tokens[index] = rng.choice(SELFIES_ALPHABET)
        elif action == "insert" and len(tokens) < 120:
            tokens.insert(index, rng.choice(SELFIES_ALPHABET))
        elif action == "delete" and len(tokens) > 2:
            tokens.pop(index)
    try:
        return selfies.decoder("".join(tokens))
    except Exception:
        return None


def generate_selfies_candidates(
    seed_rows: Iterable[dict],
    n_per_seed: int = 10,
    random_seed: int = 42,
    training_inchikeys: set[str] | None = None,
) -> list[dict]:
    rng = random.Random(random_seed)
    records: list[dict] = []
    seen: set[str] = set()
    for row in seed_rows:
        seed_smiles = str(row.get("canonical_smiles", ""))
        parent_id = str(row.get("mol_id", ""))
        for _ in range(n_per_seed):
            mutated = mutate_selfies_smiles(seed_smiles, rng)
            if not mutated:
                continue
            record = candidate_record(
                mutated,
                source_generator="selfies",
                parent_mol_id=parent_id,
                training_inchikeys=training_inchikeys,
                note="selfies_token_mutation",
            )
            if record and record["inchikey"] not in seen:
                seen.add(record["inchikey"])
                records.append(record)
    return records


def generate_rdkit_aromatic_substitutions(
    seed_rows: Iterable[dict],
    max_products_per_seed: int = 5,
    training_inchikeys: set[str] | None = None,
) -> list[dict]:
    Chem, AllChem = _rdkit()
    reactions = [
        ("aryl_F", AllChem.ReactionFromSmarts("[cH:1]>>[c:1]F")),
        ("aryl_Cl", AllChem.ReactionFromSmarts("[cH:1]>>[c:1]Cl")),
        ("aryl_Me", AllChem.ReactionFromSmarts("[cH:1]>>[c:1]C")),
        ("aryl_OMe", AllChem.ReactionFromSmarts("[cH:1]>>[c:1]OC")),
        ("aryl_NMe2", AllChem.ReactionFromSmarts("[cH:1]>>[c:1]N(C)C")),
    ]
    records: list[dict] = []
    seen: set[str] = set()
    for row in seed_rows:
        mol = Chem.MolFromSmiles(str(row.get("canonical_smiles", "")))
        if mol is None:
            continue
        parent_id = str(row.get("mol_id", ""))
        produced = 0
        for reaction_name, reaction in reactions:
            if produced >= max_products_per_seed:
                break
            try:
                product_sets = reaction.RunReactants((mol,))
            except Exception:
                continue
            for product_tuple in product_sets:
                if produced >= max_products_per_seed:
                    break
                product = product_tuple[0]
                try:
                    Chem.SanitizeMol(product)
                except Exception:
                    continue
                smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
                record = candidate_record(
                    smiles,
                    source_generator="rdkit",
                    parent_mol_id=parent_id,
                    training_inchikeys=training_inchikeys,
                    note=f"rdkit_reaction_{reaction_name}",
                )
                if record and record["inchikey"] not in seen:
                    seen.add(record["inchikey"])
                    records.append(record)
                    produced += 1
    return records


def _macrocycle_anchor_indices(mol) -> list[int]:
    Chem, _AllChem = _rdkit()
    allowed_hybridizations = {
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    }
    anchors: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetHybridization() not in allowed_hybridizations:
            continue
        if atom.GetTotalNumHs() <= 0:
            continue
        if atom.GetDegree() >= 4:
            continue
        anchors.append(atom.GetIdx())
    return anchors


def _linker_atom_numbers(length: int, chemotype: str) -> list[int]:
    atoms = [6 for _ in range(length)]
    if length < 3:
        return atoms
    middle = length // 2
    if chemotype == "oxa":
        atoms[middle] = 8
    elif chemotype == "aza":
        atoms[middle] = 7
    elif chemotype == "thioether":
        atoms[middle] = 16
    elif chemotype == "oxa_aza" and length >= 5:
        atoms[max(1, length // 3)] = 8
        atoms[min(length - 2, (2 * length) // 3)] = 7
    elif chemotype == "dioxa" and length >= 5:
        atoms[max(1, length // 3)] = 8
        atoms[min(length - 2, (2 * length) // 3)] = 8
    elif chemotype == "diaza" and length >= 5:
        atoms[max(1, length // 3)] = 7
        atoms[min(length - 2, (2 * length) // 3)] = 7
    elif chemotype == "dithio" and length >= 5:
        atoms[max(1, length // 3)] = 16
        atoms[min(length - 2, (2 * length) // 3)] = 16
    elif chemotype == "aza_oxa" and length >= 5:
        atoms[max(1, length // 3)] = 7
        atoms[min(length - 2, (2 * length) // 3)] = 8
    elif chemotype == "oxa_thio" and length >= 5:
        atoms[max(1, length // 3)] = 8
        atoms[min(length - 2, (2 * length) // 3)] = 16
    elif chemotype == "aza_thio" and length >= 5:
        atoms[max(1, length // 3)] = 7
        atoms[min(length - 2, (2 * length) // 3)] = 16
    return atoms


def _try_bridge_atoms(mol, atom_a: int, atom_b: int, linker_atoms: list[int]):
    Chem, _AllChem = _rdkit()
    rw_mol = Chem.RWMol(mol)
    added_indices: list[int] = []
    for atomic_num in linker_atoms:
        atom = Chem.Atom(int(atomic_num))
        atom.SetFormalCharge(0)
        added_indices.append(rw_mol.AddAtom(atom))
    if not added_indices:
        return None
    rw_mol.AddBond(atom_a, added_indices[0], Chem.BondType.SINGLE)
    for begin, end in zip(added_indices, added_indices[1:]):
        rw_mol.AddBond(begin, end, Chem.BondType.SINGLE)
    rw_mol.AddBond(added_indices[-1], atom_b, Chem.BondType.SINGLE)
    product = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(product)
    except Exception:
        return None
    return product


def generate_macrocycle_linker_candidates(
    seed_rows: Iterable[dict],
    max_products_per_seed: int = 8,
    random_seed: int = 42,
    training_inchikeys: set[str] | None = None,
    min_ring_size: int = 12,
    max_ring_size_target: int = 20,
    min_linker_atoms: int = 3,
    max_linker_atoms: int = 9,
    chemotype_set: str = "legacy",
) -> list[dict]:
    """Generate valid 12-20 atom macrocycles by bridging two seed scaffold atoms.

    This is a low-resource analogue of MED's anchor-plus-linker idea. It does not
    train a transformer/EDM pair, but it explicitly samples anchor pairs and
    heteroatom-containing linkers that close a macrocycle around JAK2-like seeds.
    """

    Chem, _AllChem = _rdkit()
    rng = random.Random(random_seed)
    records: list[dict] = []
    seen: set[str] = set()
    for row in seed_rows:
        seed_smiles = str(row.get("canonical_smiles", ""))
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is None:
            continue
        parent_id = str(row.get("mol_id", ""))
        anchors = _macrocycle_anchor_indices(mol)
        pair_options: list[tuple[int, int, int, int]] = []
        for atom_a, atom_b in combinations(anchors, 2):
            try:
                path = Chem.GetShortestPath(mol, atom_a, atom_b)
            except Exception:
                continue
            path_bonds = max(len(path) - 1, 0)
            if path_bonds < 4 or path_bonds > max_ring_size_target - min_linker_atoms:
                continue
            needed_lengths = [
                target - path_bonds
                for target in range(min_ring_size, max_ring_size_target + 1)
                if min_linker_atoms <= target - path_bonds <= max_linker_atoms
            ]
            for linker_length in needed_lengths:
                target_ring = path_bonds + linker_length
                pair_options.append((atom_a, atom_b, path_bonds, target_ring))

        rng.shuffle(pair_options)
        produced = 0
        for atom_a, atom_b, path_bonds, target_ring in pair_options:
            if produced >= max_products_per_seed:
                break
            linker_length = target_ring - path_bonds
            chemotypes = linker_chemotypes(chemotype_set)
            rng.shuffle(chemotypes)
            for chemotype in chemotypes:
                if produced >= max_products_per_seed:
                    break
                product = _try_bridge_atoms(mol, atom_a, atom_b, _linker_atom_numbers(linker_length, chemotype))
                if product is None:
                    continue
                smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
                record = candidate_record(
                    smiles,
                    source_generator="macrocycle_linker",
                    parent_mol_id=parent_id,
                    training_inchikeys=training_inchikeys,
                    note=f"macrocycle_bridge_{chemotype}_path{path_bonds}_ring{target_ring}",
                )
                if not record or record["inchikey"] in seen:
                    continue
                if not record["has_macrocycle_12_20"]:
                    continue
                if record["max_ring_size"] < min_ring_size or record["max_ring_size"] > max_ring_size_target:
                    continue
                seen.add(record["inchikey"])
                records.append(record)
                produced += 1
    return records


def write_candidates_csv(records: list[dict], output_csv: str | Path, filter_basic: bool = True) -> Path:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas to write candidate CSV files") from exc
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=GENERATED_CANDIDATE_COLUMNS)
    else:
        if filter_basic and "passes_basic_filters" in frame.columns:
            frame = frame[frame["passes_basic_filters"].astype(bool)].copy()
        frame = frame.drop_duplicates("inchikey")
        frame = frame[GENERATED_CANDIDATE_COLUMNS]
    frame.to_csv(output, index=False)
    return output
