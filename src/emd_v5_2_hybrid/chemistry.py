"""RDKit-backed chemistry utilities with clear optional dependency boundaries."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .config import MissingDependencyError


def _rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError("Install rdkit to use chemistry utilities") from exc
    return Chem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


@dataclass(frozen=True)
class MoleculeSummary:
    canonical_smiles: str
    inchikey: str
    valid_rdkit: bool
    mw: float
    logp: float
    tpsa: float
    hbd: int
    hba: int
    rotatable_bonds: int
    qed: float
    num_atoms: int
    num_heavy_atoms: int
    num_rings: int
    max_ring_size: int
    formal_charge: int
    has_macrocycle_12_20: bool
    has_constrained_ring_8_11: bool
    passes_basic_filters: bool
    sa_score: float


def mol_id_from_smiles(smiles: str, prefix: str = "MOL") -> str:
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def mol_from_smiles(smiles: str) -> Any:
    Chem, *_ = _rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return mol


def canonicalize_smiles(smiles: str, include_stereo: bool = True) -> str | None:
    Chem, *_ = _rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=include_stereo)


def inchikey_from_mol(mol: Any) -> str:
    Chem, *_ = _rdkit()
    return Chem.MolToInchiKey(mol)


def murcko_scaffold_smiles(mol: Any) -> str:
    Chem, *_ = _rdkit()
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return ""
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def ring_sizes(mol: Any) -> list[int]:
    rings = mol.GetRingInfo().AtomRings()
    return [len(ring) for ring in rings]


def max_ring_size(mol: Any) -> int:
    sizes = ring_sizes(mol)
    return max(sizes) if sizes else 0


def approximate_sa_score(mol: Any) -> float:
    """Small transparent SA proxy on a 1-10 scale.

    This is not the Ertl-Schuffenhauer SA score. Use it when the full RDKit
    contrib implementation is unavailable, and label it as an approximation.
    """

    Chem, Crippen, Descriptors, Lipinski, _QED, _rdMolDescriptors = _rdkit()
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    rings = mol.GetRingInfo().NumRings()
    largest_ring = max_ring_size(mol)
    stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    score = 1.0
    score += min(max((mw - 350.0) / 180.0, 0.0), 2.0)
    score += min(rot / 8.0, 1.5)
    score += min(rings / 4.0, 1.2)
    score += min(stereo_centers / 4.0, 1.0)
    if largest_ring >= 12:
        score += 1.2
    elif largest_ring >= 8:
        score += 0.5
    if logp > 5:
        score += min((logp - 5.0) / 2.0, 1.0)
    return round(float(max(1.0, min(10.0, score))), 3)


def summarize_molecule(
    smiles: str,
    mw_min: float = 250,
    mw_max: float = 900,
    logp_max: float = 7,
    tpsa_max: float = 200,
) -> MoleculeSummary | None:
    Chem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors = _rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    sizes = ring_sizes(mol)
    largest_ring = max(sizes) if sizes else 0
    mw = float(Descriptors.MolWt(mol))
    logp = float(Crippen.MolLogP(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rot = int(Lipinski.NumRotatableBonds(mol))
    qed = float(QED.qed(mol))
    formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    passes = bool(mw_min <= mw <= mw_max and logp <= logp_max and tpsa <= tpsa_max)
    return MoleculeSummary(
        canonical_smiles=canonical,
        inchikey=Chem.MolToInchiKey(mol),
        valid_rdkit=True,
        mw=round(mw, 4),
        logp=round(logp, 4),
        tpsa=round(tpsa, 4),
        hbd=hbd,
        hba=hba,
        rotatable_bonds=rot,
        qed=round(qed, 4),
        num_atoms=int(mol.GetNumAtoms()),
        num_heavy_atoms=int(mol.GetNumHeavyAtoms()),
        num_rings=int(mol.GetRingInfo().NumRings()),
        max_ring_size=int(largest_ring),
        formal_charge=formal_charge,
        has_macrocycle_12_20=bool(12 <= largest_ring <= 20),
        has_constrained_ring_8_11=bool(8 <= largest_ring <= 11),
        passes_basic_filters=passes,
        sa_score=approximate_sa_score(mol),
    )


def p_activity_from_nm(value_nm: float | int | str | None) -> float | None:
    if value_nm in (None, ""):
        return None
    value = float(value_nm)
    if not math.isfinite(value) or value <= 0:
        return None
    return -math.log10(value * 1e-9)


def summary_as_dict(summary: MoleculeSummary) -> dict[str, Any]:
    return {
        "canonical_smiles": summary.canonical_smiles,
        "inchikey": summary.inchikey,
        "valid_rdkit": summary.valid_rdkit,
        "mw": summary.mw,
        "logp": summary.logp,
        "tpsa": summary.tpsa,
        "hbd": summary.hbd,
        "hba": summary.hba,
        "rotatable_bonds": summary.rotatable_bonds,
        "qed": summary.qed,
        "num_atoms": summary.num_atoms,
        "num_heavy_atoms": summary.num_heavy_atoms,
        "num_rings": summary.num_rings,
        "max_ring_size": summary.max_ring_size,
        "formal_charge": summary.formal_charge,
        "has_macrocycle_12_20": summary.has_macrocycle_12_20,
        "has_constrained_ring_8_11": summary.has_constrained_ring_8_11,
        "passes_basic_filters": summary.passes_basic_filters,
        "sa_score": summary.sa_score,
    }
