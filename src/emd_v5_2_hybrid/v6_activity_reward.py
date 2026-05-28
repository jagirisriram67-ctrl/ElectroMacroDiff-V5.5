"""V6 JAK/JAK2 activity reward model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pocket_features import POCKET_FEATURE_DIM


DESCRIPTOR_COLUMNS = [
    "mw",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "qed",
    "heavy_atoms",
    "ring_count",
    "max_ring_size",
    "formal_charge",
]
FINGERPRINT_BITS = 256
V6_ACTIVITY_INPUT_DIM = len(DESCRIPTOR_COLUMNS) + FINGERPRINT_BITS + POCKET_FEATURE_DIM


@dataclass(frozen=True)
class ActivityExample:
    """One activity reward training example."""

    mol_id: str
    canonical_smiles: str
    inchikey: str
    p_activity: float
    active_label: int
    feature_vector: list[float]


def featurize_activity_smiles(smiles: str, pocket_vector: list[float]) -> list[float] | None:
    """Convert a SMILES string into descriptor + fingerprint + pocket features."""

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
    except ImportError as exc:
        raise RuntimeError("Install rdkit to build V6 activity reward features") from exc

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    ring_info = mol.GetRingInfo()
    max_ring = max((len(ring) for ring in ring_info.AtomRings()), default=0)
    descriptors = [
        _scale(Descriptors.MolWt(mol), 800.0),
        _scale(Crippen.MolLogP(mol), 10.0),
        _scale(rdMolDescriptors.CalcTPSA(mol), 250.0),
        _scale(Lipinski.NumHDonors(mol), 10.0),
        _scale(Lipinski.NumHAcceptors(mol), 15.0),
        _scale(Lipinski.NumRotatableBonds(mol), 20.0),
        float(QED.qed(mol)),
        _scale(mol.GetNumHeavyAtoms(), 80.0),
        _scale(ring_info.NumRings(), 10.0),
        _scale(max_ring, 25.0),
        _scale(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()), 5.0),
    ]

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FINGERPRINT_BITS)
    fp_bits = [float(int(bit)) for bit in fp.ToBitString()]
    return descriptors + fp_bits + [float(x) for x in pocket_vector]


def build_activity_examples(
    rows: list[dict[str, Any]],
    pocket_vector: list[float],
    active_threshold: float = 7.0,
) -> list[ActivityExample]:
    """Build activity examples from curated rows."""

    examples: list[ActivityExample] = []
    for idx, row in enumerate(rows):
        smiles = str(row.get("canonical_smiles") or row.get("smiles") or "")
        p_activity = _to_float(row.get("p_activity") or row.get("pchembl_value"))
        if not smiles or p_activity is None:
            continue
        features = featurize_activity_smiles(smiles, pocket_vector)
        if features is None:
            continue
        examples.append(
            ActivityExample(
                mol_id=str(row.get("mol_id") or row.get("molecule_chembl_id") or f"ACT_{idx:06d}"),
                canonical_smiles=smiles,
                inchikey=str(row.get("inchikey") or row.get("standard_inchi_key") or ""),
                p_activity=p_activity,
                active_label=int(p_activity >= active_threshold),
                feature_vector=features,
            )
        )
    return examples


def save_activity_reward_checkpoint(
    path: str | Path,
    model: Any,
    payload: dict[str, Any],
) -> Path:
    """Save a V6 activity reward checkpoint."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install torch to save the activity reward model") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_type": "v6_activity_reward",
        "input_dim": V6_ACTIVITY_INPUT_DIM,
        "descriptor_columns": DESCRIPTOR_COLUMNS,
        "fingerprint_bits": FINGERPRINT_BITS,
        "pocket_dim": POCKET_FEATURE_DIM,
        "model_state_dict": model.state_dict(),
    }
    data.update(payload)
    torch.save(data, output)
    return output


def make_activity_reward_mlp(input_dim: int = V6_ACTIVITY_INPUT_DIM, hidden_dim: int = 256):
    """Create the dual-head V6 activity reward MLP."""

    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install torch to train V6 activity reward") from exc

    class V6ActivityRewardMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.1),
            )
            self.activity_head = nn.Linear(hidden_dim // 2, 1)
            self.pactivity_head = nn.Linear(hidden_dim // 2, 1)

        def forward(self, x):
            h = self.encoder(x)
            return self.activity_head(h).view(-1), self.pactivity_head(h).view(-1)

    return V6ActivityRewardMLP()


def _scale(value: float, divisor: float) -> float:
    return float(value) / float(divisor)


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
