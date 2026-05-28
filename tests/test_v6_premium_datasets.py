from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.v6_premium_datasets import scan_all_datasets, write_v6_manifest


class V6PremiumDatasetTests(unittest.TestCase):
    def test_scan_activity_table_marks_first_lane_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            activity = base / "mounted_activity"
            activity.mkdir(parents=True)
            (activity / "jak2_activity.csv").write_text(
                "canonical_smiles,target,value_nm\nCCO,JAK2,100\n",
                encoding="utf-8",
            )

            results = scan_all_datasets(
                base,
                dataset_roots={"chembl_bindingdb_jak": [activity]},
            )
            by_key = {row.dataset_key: row for row in results}

            self.assertTrue(by_key["chembl_bindingdb_jak"].ready_for_training)
            self.assertEqual(by_key["chembl_bindingdb_jak"].discovered_records, 1)
            self.assertFalse(by_key["pdbbind"].ready_for_training)

    def test_scan_pdbbind_requires_index_protein_and_ligand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pdbbind = base / "pdbbind"
            complex_dir = pdbbind / "1abc"
            complex_dir.mkdir(parents=True)
            (pdbbind / "INDEX_refined_data.2020").write_text(
                "# header\n1abc 2.00 2019 7.50 Kd=31nM // ref (LIG)\n",
                encoding="utf-8",
            )
            (complex_dir / "1abc_protein.pdb").write_text("ATOM\n", encoding="utf-8")
            (complex_dir / "1abc_ligand.sdf").write_text("LIG\n$$$$\n", encoding="utf-8")

            results = scan_all_datasets(base, dataset_roots={"pdbbind": [pdbbind]})
            by_key = {row.dataset_key: row for row in results}

            self.assertTrue(by_key["pdbbind"].ready_for_training)
            self.assertEqual(by_key["pdbbind"].discovered_records, 1)

    def test_manifest_writer_creates_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            activity = base / "activity"
            activity.mkdir(parents=True)
            (activity / "BindingDB_JAK.csv").write_text(
                "smiles,target,ki_nm\nCCN,JAK2,50\n",
                encoding="utf-8",
            )

            results = scan_all_datasets(
                base,
                dataset_roots={"chembl_bindingdb_jak": [activity]},
            )
            summary = write_v6_manifest(base, results)

            manifest = base / "03_features" / "v6_premium_dataset_manifest.csv"
            summary_path = base / "03_features" / "v6_premium_dataset_summary.json"
            self.assertTrue(manifest.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary["recommended_next_lane"], "account_1_activity_reward")

            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)

            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertIn("chembl_bindingdb_jak", loaded["ready_datasets"])

    def test_activity_reward_features_have_expected_dimension(self) -> None:
        try:
            from emd_v5_2_hybrid.pocket_features import POCKET_FEATURE_DIM
            from emd_v5_2_hybrid.v6_activity_reward import (
                FINGERPRINT_BITS,
                DESCRIPTOR_COLUMNS,
                V6_ACTIVITY_INPUT_DIM,
                featurize_activity_smiles,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))

        pocket = [0.0] * POCKET_FEATURE_DIM
        features = featurize_activity_smiles("CCOc1ccccc1", pocket)

        self.assertIsNotNone(features)
        self.assertEqual(len(features), V6_ACTIVITY_INPUT_DIM)
        self.assertEqual(V6_ACTIVITY_INPUT_DIM, len(DESCRIPTOR_COLUMNS) + FINGERPRINT_BITS + POCKET_FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()
