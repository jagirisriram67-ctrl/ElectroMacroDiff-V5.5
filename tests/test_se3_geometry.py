from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_se3_script():
    module_path = ROOT / "scripts" / "23_score_se3_geometry.py"
    spec = importlib.util.spec_from_file_location("score_se3_geometry", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SE3GeometryScriptTests(unittest.TestCase):
    def test_se3_geometry_scoring_writes_rows_and_merged_ranking(self) -> None:
        try:
            import torch
            from emd_v5_2_hybrid.se3_flow import SE3FlowConfig, SE3FlowMatching
        except ImportError as exc:
            self.skipTest(str(exc))

        module = load_se3_script()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ranking_csv = base / "ranking.csv"
            checkpoint_path = base / "se3_best_checkpoint.pt"
            output_csv = base / "scores.csv"
            output_ranking_csv = base / "ranking_with_se3.csv"

            with ranking_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["candidate_id", "canonical_smiles", "rank"])
                writer.writeheader()
                writer.writerow({"candidate_id": "CAND_TEST", "canonical_smiles": "CCO", "rank": 1})

            model = SE3FlowMatching(SE3FlowConfig(atom_feature_dim=64, hidden_dim=32, num_layers=2))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"atom_feature_dim": 64, "hidden_dim": 32, "num_layers": 2},
                },
                checkpoint_path,
            )

            argv_backup = sys.argv[:]
            try:
                sys.argv = [
                    str(ROOT / "scripts" / "23_score_se3_geometry.py"),
                    "--base",
                    str(base),
                    "--ranking-csv",
                    str(ranking_csv),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--output-csv",
                    str(output_csv),
                    "--output-ranking-csv",
                    str(output_ranking_csv),
                    "--device",
                    "cpu",
                    "--num-repeats",
                    "2",
                ]
                module.main()
            finally:
                sys.argv = argv_backup

            self.assertTrue(output_csv.exists())
            self.assertTrue(output_ranking_csv.exists())

            with output_csv.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_id"], "CAND_TEST")
            self.assertEqual(rows[0]["se3_geometry_status"], "scored")


if __name__ == "__main__":
    unittest.main()
