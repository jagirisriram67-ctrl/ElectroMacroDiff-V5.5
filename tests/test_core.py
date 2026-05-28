from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.admet_synthesis import lipinski_violations, score_admet_from_descriptors, veber_pass
from emd_v5_2_hybrid.baseline_generation import generate_macrocycle_linker_candidates
from emd_v5_2_hybrid.benchmarking import MED_REFERENCE, compare_to_reference, summarize_candidate_frame
from emd_v5_2_hybrid.docking import infer_grid_from_ligand, parse_vina_best_score
from emd_v5_2_hybrid.linker_size_features import LINKER_FEATURE_COLUMNS, core_features, molecule_feature_map
from emd_v5_2_hybrid.pocket_electronics import score_pocket_electronic_fit
from emd_v5_2_hybrid.pose_analysis import analyze_pose
from emd_v5_2_hybrid.ranking import CandidateScore, decision_label, docking_to_score, weighted_score
from emd_v5_2_hybrid.registry import ensure_project_tree, load_progress, register_run, save_progress
from emd_v5_2_hybrid.validation import validate_docking_scores, validate_project_dirs, validation_summary


class RegistryTests(unittest.TestCase):
    def test_project_tree_and_run_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ensure_project_tree(base)
            run_id = register_run(base, "test_stage", "completed", molecules_in=3, molecules_out=2)
            registry = base / "00_project_registry" / "run_registry.csv"
            self.assertTrue(registry.exists())
            with registry.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]["run_id"], run_id)
            self.assertEqual(rows[-1]["stage"], "test_stage")

    def test_progress_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            self.assertEqual(load_progress(path), {"last_completed_index": -1})
            save_progress(path, {"last_completed_index": 7, "stage": "x"})
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["last_completed_index"], 7)
            self.assertIn("updated_at", loaded)


class RankingTests(unittest.TestCase):
    def test_docking_to_score_more_negative_is_better(self) -> None:
        scores = docking_to_score([-10.0, -8.0, -6.0])
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_docking_to_score_handles_missing_values(self) -> None:
        scores = docking_to_score([-10.0, None, float("nan"), -6.0])
        self.assertEqual(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)
        self.assertGreater(scores[0], scores[3])

    def test_weighted_score_and_decision(self) -> None:
        candidate = CandidateScore(
            candidate_id="c1",
            docking_score=1.0,
            pose_score=0.8,
            admet_score=0.7,
            synthesis_score=0.7,
            novelty_score=0.9,
            safety_proxy_score=0.8,
        )
        score = weighted_score(candidate)
        self.assertGreater(score, 0.7)
        self.assertEqual(decision_label(0.8), "primary_candidate")
        self.assertEqual(decision_label(0.2), "reject_or_low_priority")


class AdmetTests(unittest.TestCase):
    def test_basic_rules(self) -> None:
        self.assertEqual(lipinski_violations(450, 3, 2, 6), 0)
        self.assertEqual(lipinski_violations(650, 6, 8, 12), 4)
        self.assertTrue(veber_pass(90, 7))
        self.assertFalse(veber_pass(180, 12))

    def test_profile_bounds(self) -> None:
        profile = score_admet_from_descriptors(
            qed=0.7,
            sa_score=4.0,
            mw=450,
            logp=3.2,
            tpsa=95,
            hbd=2,
            hba=7,
            rotatable_bonds=7,
        )
        self.assertGreaterEqual(profile.admet_score, 0.0)
        self.assertLessEqual(profile.admet_score, 1.0)
        self.assertGreaterEqual(profile.synthesis_score, 0.0)
        self.assertLessEqual(profile.synthesis_score, 1.0)


class BaselineGenerationTests(unittest.TestCase):
    def test_macrocycle_linker_generator_closes_macrocycles(self) -> None:
        seed_rows = [
            {
                "mol_id": "seed1",
                "canonical_smiles": "CC(C)(C)c1nc2c3ccc(F)cc3c3c(=O)[nH]ccc3c2[nH]1",
            }
        ]
        try:
            records = generate_macrocycle_linker_candidates(seed_rows, max_products_per_seed=3)
        except RuntimeError as exc:
            self.skipTest(str(exc))
        self.assertGreater(len(records), 0)
        self.assertTrue(all(record["has_macrocycle_12_20"] for record in records))
        self.assertTrue(all(12 <= int(record["max_ring_size"]) <= 20 for record in records))

    def test_linker_size_features_include_anchor_pair_context(self) -> None:
        try:
            from rdkit import Chem
        except ImportError as exc:
            self.skipTest(str(exc))
        mol = Chem.MolFromSmiles("CCCOCCC")
        features = molecule_feature_map(mol, desired_ring_size=14, atom_a=0, atom_b=6)
        self.assertIn("anchor_path_bonds", LINKER_FEATURE_COLUMNS)
        self.assertGreater(features["anchor_path_bonds"], 0.0)
        core = core_features("[*:1]CCCO[*:2]", desired_ring_size=14, core_heavy_atoms=7)
        self.assertGreater(core["anchor_path_bonds"], 0.0)

    def test_anchor_gnn_forward_matches_atom_count(self) -> None:
        try:
            from rdkit import Chem
            from emd_v5_2_hybrid.anchor_gnn import AnchorMessageGNN, graph_from_mol_for_anchor_gnn
        except ImportError as exc:
            self.skipTest(str(exc))
        mol = Chem.MolFromSmiles("c1ccccc1CCO")
        graph = graph_from_mol_for_anchor_gnn(mol)
        model = AnchorMessageGNN(node_dim=64, edge_dim=6, hidden_dim=16, num_layers=1)
        logits = model(graph["node_features"], graph["edge_index"], graph["edge_features"])
        self.assertEqual(tuple(logits.shape), (mol.GetNumAtoms(),))


class BenchmarkingTests(unittest.TestCase):
    def test_candidate_summary_marks_filtered_survivor_metrics(self) -> None:
        try:
            import pandas as pd
        except ImportError as exc:
            self.skipTest(str(exc))
        frame = pd.DataFrame(
            [
                {
                    "inchikey": "A",
                    "valid_rdkit": True,
                    "has_macrocycle_12_20": True,
                    "novel_flag": True,
                    "passes_basic_filters": True,
                    "qed": 0.4,
                },
                {
                    "inchikey": "B",
                    "valid_rdkit": True,
                    "has_macrocycle_12_20": True,
                    "novel_flag": True,
                    "passes_basic_filters": True,
                    "qed": 0.6,
                },
            ]
        )
        summary = summarize_candidate_frame(frame, "test_branch", "filtered_survivor")
        self.assertEqual(summary["validity_percent"], 100.0)
        self.assertEqual(summary["uniqueness_percent"], 100.0)
        self.assertEqual(summary["macrocyclization_percent"], 100.0)
        self.assertEqual(summary["linker_novelty_percent"], None)
        comparison = compare_to_reference(summary, MED_REFERENCE["MED"])
        self.assertFalse(comparison["paper_equivalent_claim_allowed"])


class ValidationAndDockingTests(unittest.TestCase):
    def test_project_dir_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ensure_project_tree(base)
            summary = validation_summary(validate_project_dirs(base))
            self.assertEqual(summary["failed"], 0)

    def test_infer_grid_from_ligand(self) -> None:
        pdb_text = (
            "HETATM    1  C1  QUP A2000      10.000  20.000  30.000  1.00 20.00           C  \n"
            "HETATM    2  C2  QUP A2000      12.000  22.000  32.000  1.00 20.00           C  \n"
            "HETATM    3  O   HOH A3000       0.000   0.000   0.000  1.00 20.00           O  \n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mini.pdb"
            path.write_text(pdb_text, encoding="utf-8")
            grid = infer_grid_from_ligand(path, padding=8.0)
            self.assertEqual(grid["reference_ligand"], "QUP:A:2000:")
            self.assertEqual(grid["center"], {"x": 11.0, "y": 21.0, "z": 31.0})

    def test_docking_score_validation_requires_numeric_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docking_scores.csv"
            path.write_text("candidate_id,best_score\nA,\n", encoding="utf-8")
            self.assertFalse(validate_docking_scores(path).passed)
            path.write_text("candidate_id,best_score\nA,-8.1\n", encoding="utf-8")
            self.assertTrue(validate_docking_scores(path).passed)

    def test_parse_vina_best_score(self) -> None:
        log_text = """
-----+------------+----------+----------
mode | affinity   | dist from best mode
-----+------------+----------+----------
   1       -9.42          0          0
   2       -8.10      1.234      2.345
"""
        self.assertEqual(parse_vina_best_score(log_text), -9.42)

    def test_vina_gpu_ligand_sanitizer_rewrites_fixed_atom_type_columns(self) -> None:
        module_path = ROOT / "scripts" / "06_prepare_vina_gpu_ligands.py"
        spec = importlib.util.spec_from_file_location("prepare_vina_gpu_ligands", module_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        spec.loader.exec_module(module)

        line = "ATOM     24  C   UNL     1       0.093   1.868   0.400  1.00  0.00     0.100 CG0\n"
        rewritten, raw_type, new_type = module.rewrite_atom_type(line)
        self.assertEqual(raw_type, "CG")
        self.assertEqual(new_type, "C")
        self.assertEqual("".join(rewritten.ljust(79)[77:79].split()), "C")
        self.assertNotIn("CG0", rewritten)

    def test_pose_analysis_flags_reasonable_pose_as_pass(self) -> None:
        def atom_line(record: str, serial: int, atom: str, residue: str, x: float, y: float, z: float, element: str) -> str:
            return f"{record:<6}{serial:5d} {atom:<4} {residue:>3} A{1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2}\n"

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            receptor = base / "receptor.pdb"
            pose = base / "ligand.pdbqt"
            grid = base / "grid.json"
            receptor.write_text(
                atom_line("ATOM", 1, "CA", "ALA", 0.0, 0.0, 0.0, "C")
                + atom_line("ATOM", 2, "CB", "ALA", 5.0, 0.0, 0.0, "C"),
                encoding="utf-8",
            )
            pose.write_text(
                atom_line("HETATM", 1, "C1", "LIG", 2.5, 0.0, 0.0, "C")
                + atom_line("HETATM", 2, "N1", "LIG", 2.8, 0.0, 0.0, "N"),
                encoding="utf-8",
            )
            grid.write_text(json.dumps({"center": {"x": 2.65, "y": 0.0, "z": 0.0}}), encoding="utf-8")
            result = analyze_pose("CAND_TEST", pose, receptor, grid, docking_score=-8.0)
            self.assertEqual(result["pose_decision"], "pass")
            self.assertEqual(result["hard_clashes_lt_1_8A"], 0)
            self.assertGreater(result["contacts_within_4A"], 0)

    def test_pocket_electronic_fit_scores_opposite_charges(self) -> None:
        def pdbqt_atom(record: str, serial: int, atom: str, residue: str, x: float, y: float, z: float, charge: float, atom_type: str) -> str:
            return (
                f"{record:<6}{serial:5d} {atom:<4} {residue:>3} A{1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    {charge:6.3f} {atom_type:<2}\n"
            )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            receptor = base / "rec.pdbqt"
            pose = base / "lig.pdbqt"
            receptor.write_text(
                pdbqt_atom("ATOM", 1, "NZ", "LYS", 0.0, 0.0, 0.0, 0.45, "N")
                + pdbqt_atom("ATOM", 2, "CB", "LEU", 4.2, 0.0, 0.0, 0.02, "C"),
                encoding="utf-8",
            )
            pose.write_text(
                pdbqt_atom("ATOM", 1, "O1", "UNL", 2.8, 0.0, 0.0, -0.45, "OA")
                + pdbqt_atom("ATOM", 2, "C1", "UNL", 4.4, 0.0, 0.0, 0.02, "C"),
                encoding="utf-8",
            )
            result = score_pocket_electronic_fit("CAND_TEST", pose, receptor, docking_score=-8.0)
            self.assertGreater(result["electrostatic_favorable"], 0.0)
            self.assertGreater(result["pocket_electronic_fit_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
