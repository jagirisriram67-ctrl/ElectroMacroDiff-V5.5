"""Fixed-size pocket electronic feature vectors for V5.5 conditioned models.

Converts the variable-length pocket profile JSON into a fixed-dimension
numeric vector that can be concatenated with ligand features and fed
directly into PocketAnchorGNN, PocketLinkerPolicy, and the
PocketValidityRewardModel.

The pocket feature vector encodes:
  - Raw interaction counts (normalized)
  - Fractional composition
  - Charge properties
  - Amino-acid-type fingerprint of contact residues
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


# 20 standard amino acid 3-letter codes
AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

POCKET_FEATURE_DIM = 32


def pocket_feature_vector(profile: dict[str, Any]) -> list[float]:
    """Convert a pocket electronic profile dict to a 32-dim feature vector.

    Parameters
    ----------
    profile : dict
        The dict loaded from ``jak2_pocket_electronic_profile.json``.

    Returns
    -------
    list[float]
        A fixed-length numeric vector of dimension ``POCKET_FEATURE_DIM``.
    """
    atom_count = max(float(profile.get("pocket_atom_count", 1)), 1.0)
    hydro = float(profile.get("hydrophobic_atom_count", 0))
    donor = float(profile.get("hbond_donor_atom_count", 0))
    acceptor = float(profile.get("hbond_acceptor_atom_count", 0))
    aromatic = float(profile.get("aromatic_atom_count", 0))
    charge = float(profile.get("pocket_net_charge_proxy", 0.0))

    # --- Normalized raw counts (6) ---
    features = [
        atom_count / 500.0,
        hydro / 200.0,
        donor / 200.0,
        acceptor / 200.0,
        aromatic / 100.0,
        _safe_clip(charge, -1.0, 1.0),
    ]

    # --- Fractional composition (5) ---
    features.extend([
        hydro / atom_count,
        donor / atom_count,
        acceptor / atom_count,
        aromatic / atom_count,
        abs(charge) / atom_count,
    ])

    # --- Derived ratios (1) ---
    features.append(donor / max(acceptor, 1.0))

    # --- Amino acid contact fingerprint (20) ---
    aa_fingerprint = [0.0] * 20
    contact_residues = profile.get("top_contact_residues", [])
    total_contacts = max(sum(r.get("atom_contacts", 0) for r in contact_residues), 1)
    for residue_entry in contact_residues:
        residue_name = str(residue_entry.get("residue", ""))
        aa_code = residue_name.split(":")[0].upper() if ":" in residue_name else residue_name[:3].upper()
        if aa_code in AA_INDEX:
            contacts = float(residue_entry.get("atom_contacts", 1))
            aa_fingerprint[AA_INDEX[aa_code]] += contacts / total_contacts
    features.extend(aa_fingerprint)

    # Ensure exactly POCKET_FEATURE_DIM
    assert len(features) == POCKET_FEATURE_DIM, f"Expected {POCKET_FEATURE_DIM}, got {len(features)}"
    return features


def load_pocket_profile(path: str | Path) -> dict[str, Any]:
    """Load the pocket electronic profile JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pocket_feature_vector(path: str | Path) -> list[float]:
    """Load profile JSON and return the feature vector."""
    return pocket_feature_vector(load_pocket_profile(path))


def pocket_feature_tensor(profile: dict[str, Any]):
    """Return a PyTorch tensor of the pocket feature vector."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch") from exc
    return torch.tensor(pocket_feature_vector(profile), dtype=torch.float32)


def _safe_clip(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(lo, min(hi, value))
