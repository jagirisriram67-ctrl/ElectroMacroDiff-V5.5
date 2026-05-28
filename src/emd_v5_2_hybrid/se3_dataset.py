"""RDKit conformer to graph tensor conversion for the SE(3) debug pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chemistry import canonicalize_smiles, max_ring_size
from .config import MissingDependencyError


def _rdkit_3d():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError("Install rdkit to build SE(3) graph datasets") from exc
    return Chem, AllChem


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError("Install torch to save graph tensors") from exc
    return torch


def atom_features(atom: Any, dim: int = 64) -> list[float]:
    hybridization = str(atom.GetHybridization())
    values = [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 3.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetTotalNumHs() / 4.0,
        float(hybridization == "SP"),
        float(hybridization == "SP2"),
        float(hybridization == "SP3"),
        float(atom.GetChiralTag() != 0),
        atom.GetMass() / 250.0,
    ]
    if len(values) > dim:
        return values[:dim]
    return values + [0.0] * (dim - len(values))


def bond_features(bond: Any) -> list[float]:
    bond_type = str(bond.GetBondType())
    return [
        float(bond_type == "SINGLE"),
        float(bond_type == "DOUBLE"),
        float(bond_type == "TRIPLE"),
        float(bond_type == "AROMATIC"),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def embed_molecule(smiles: str, random_seed: int = 42):
    Chem, AllChem = _rdkit_3d()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    status = AllChem.EmbedMolecule(mol, randomSeed=random_seed, useRandomCoords=True)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=random_seed, useRandomCoords=True, maxAttempts=100)
    if status != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
    return Chem.RemoveHs(mol)


def graph_from_smiles(
    smiles: str,
    mol_id: str,
    atom_feature_dim: int = 64,
    random_seed: int = 42,
) -> dict[str, Any] | None:
    torch = _torch()
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    mol = embed_molecule(canonical, random_seed=random_seed)
    if mol is None:
        return None

    conformer = mol.GetConformer()
    coords = []
    for atom_idx in range(mol.GetNumAtoms()):
        position = conformer.GetAtomPosition(atom_idx)
        coords.append([position.x, position.y, position.z])

    atom_tensor = torch.tensor(
        [atom_features(atom, atom_feature_dim) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )
    coord_tensor = torch.tensor(coords, dtype=torch.float32)

    edges: list[list[int]] = []
    edge_features: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edges.append([begin, end])
        edge_features.append(features)
        edges.append([end, begin])
        edge_features.append(features)
    if edges:
        bond_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        bond_feature_tensor = torch.tensor(edge_features, dtype=torch.float32)
    else:
        bond_index = torch.empty((2, 0), dtype=torch.long)
        bond_feature_tensor = torch.empty((0, 6), dtype=torch.float32)

    largest_ring = max_ring_size(mol)
    ring_mask = torch.tensor([float(atom.IsInRing()) for atom in mol.GetAtoms()], dtype=torch.float32)
    macro_mask = torch.tensor(
        [float(atom.IsInRing() and 12 <= largest_ring <= 20) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )
    return {
        "mol_id": mol_id,
        "smiles": canonical,
        "atom_features": atom_tensor,
        "bond_index": bond_index,
        "bond_features": bond_feature_tensor,
        "coords": coord_tensor,
        "ring_mask": ring_mask,
        "macrocycle_mask": macro_mask,
    }


def build_graphs_from_csv(
    curated_csv: str | Path,
    output_graphs: str | Path,
    output_index: str | Path,
    atom_feature_dim: int = 64,
    limit: int | None = None,
    random_seed: int = 42,
) -> tuple[Path, Path]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError("Install pandas to build graph datasets") from exc
    torch = _torch()

    frame = pd.read_csv(curated_csv)
    if limit:
        frame = frame.head(limit)
    graphs = []
    index_rows = []
    for row in frame.to_dict(orient="records"):
        graph = graph_from_smiles(
            str(row["canonical_smiles"]),
            str(row["mol_id"]),
            atom_feature_dim=atom_feature_dim,
            random_seed=random_seed,
        )
        if graph is None:
            continue
        graphs.append(graph)
        index_rows.append(
            {
                "mol_id": graph["mol_id"],
                "smiles": graph["smiles"],
                "num_atoms": int(graph["coords"].shape[0]),
                "split": row.get("split", ""),
            }
        )

    graph_path = Path(output_graphs)
    index_path = Path(output_index)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, graph_path)
    pd.DataFrame(index_rows).to_csv(index_path, index=False)
    return graph_path, index_path
