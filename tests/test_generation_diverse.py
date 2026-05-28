from __future__ import annotations

import importlib.util
import sys
import unittest
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.baseline_generation import (
    _linker_atom_numbers,
    _macrocycle_anchor_indices,
    _try_bridge_atoms,
    candidate_record,
    linker_chemotypes,
)
from emd_v5_2_hybrid.macrocycle_fragmentation import extract_linker_smiles_from_smiles


def load_generation_module():
    module_path = ROOT / "scripts" / "17_generate_v5_3_model_guided_macrocycles.py"
    spec = importlib.util.spec_from_file_location("generate_v5_3_model_guided_macrocycles", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DiverseGenerationTests(unittest.TestCase):
    def test_diverse_chemotype_patterns_are_supported_by_chain_builder(self) -> None:
        diverse = linker_chemotypes("diverse")
        self.assertIn("dioxa", diverse)
        self.assertIn("diaza", diverse)
        self.assertIn("aza_thio", diverse)

        for chemotype in ["dioxa", "diaza", "dithio", "aza_oxa", "oxa_thio", "aza_thio"]:
            atoms = _linker_atom_numbers(7, chemotype)
            self.assertEqual(len(atoms), 7)
            self.assertTrue(any(atom != 6 for atom in atoms), msg=chemotype)

    def test_generation_rejects_known_training_linkers_when_novelty_is_enforced(self) -> None:
        try:
            from rdkit import Chem
        except ImportError as exc:
            self.skipTest(str(exc))

        module = load_generation_module()
        seed_rows = [
            {
                "mol_id": "seed1",
                "canonical_smiles": "CC(C)(C)c1nc2c3ccc(F)cc3c3c(=O)[nH]ccc3c2[nH]1",
            }
        ]
        seed_smiles = seed_rows[0]["canonical_smiles"]
        mol = Chem.MolFromSmiles(seed_smiles)
        self.assertIsNotNone(mol)

        selected = None
        for atom_a, atom_b in combinations(_macrocycle_anchor_indices(mol), 2):
            try:
                path = Chem.GetShortestPath(mol, atom_a, atom_b)
            except Exception:
                continue
            path_bonds = max(len(path) - 1, 0)
            for target_ring in range(12, 21):
                linker_length = target_ring - path_bonds
                if not 3 <= linker_length <= 9:
                    continue
                product = _try_bridge_atoms(mol, atom_a, atom_b, _linker_atom_numbers(linker_length, "alkyl"))
                if product is None:
                    continue
                selected = (atom_a, atom_b, path_bonds, linker_length, target_ring)
                break
            if selected is not None:
                break

        if selected is None:
            self.skipTest("Could not find a valid anchor pair for the novelty rejection test")

        atom_a, atom_b, path_bonds, linker_length, target_ring = selected
        training_linkers: set[str] = set()
        for chemotype in linker_chemotypes("legacy"):
            product = _try_bridge_atoms(mol, atom_a, atom_b, _linker_atom_numbers(linker_length, chemotype))
            if product is None:
                continue
            smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
            record = candidate_record(smiles, source_generator="test")
            if not record or not record["passes_basic_filters"] or not record["has_macrocycle_12_20"]:
                continue
            training_linkers.update(
                extract_linker_smiles_from_smiles(
                    str(record["canonical_smiles"]),
                    mol_id=str(record["candidate_id"]),
                )
            )

        if not training_linkers:
            self.skipTest("Could not derive training linker identities for the novelty rejection test")

        original_score_anchor_atoms = module.score_anchor_atoms
        original_ranked_pair_options = module.ranked_pair_options
        try:
            module.score_anchor_atoms = lambda *args, **kwargs: [(atom_a, 0.95), (atom_b, 0.93)]
            module.ranked_pair_options = lambda *args, **kwargs: [
                {
                    "atom_a": atom_a,
                    "atom_b": atom_b,
                    "path_bonds": path_bonds,
                    "linker_length": linker_length,
                    "target_ring": target_ring,
                    "anchor_score_a": 0.95,
                    "anchor_score_b": 0.93,
                    "linker_probability": 0.82,
                    "pair_score": 0.72,
                }
            ]
            records, attempts = module.generate_model_guided_macrocycles(
                seed_rows=seed_rows,
                anchor_model=None,
                linker_model=None,
                linker_labels=[],
                device="cpu",
                training_inchikeys=set(),
                top_anchor_atoms=2,
                min_anchor_probability=0.05,
                max_products_per_seed=1,
                random_seed=42,
                anchor_feature_columns=[],
                linker_feature_columns=[],
                chemotype_set="legacy",
                enforce_linker_novelty=True,
                novelty_mode="exact",
                training_linkers=training_linkers,
            )
        finally:
            module.score_anchor_atoms = original_score_anchor_atoms
            module.ranked_pair_options = original_ranked_pair_options

        self.assertEqual(records, [])
        self.assertTrue(any(row.get("status") == "rejected_known_linker" for row in attempts))


if __name__ == "__main__":
    unittest.main()
