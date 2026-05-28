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

from emd_v5_2_hybrid.baseline_generation import generate_macrocycle_linker_candidates
from emd_v5_2_hybrid.macrocycle_fragmentation import extract_linker_smiles_from_smiles


def load_gap_module():
    module_path = ROOT / "scripts" / "20_measure_v5_3_benchmark_gaps.py"
    spec = importlib.util.spec_from_file_location("measure_v5_3_benchmark_gaps", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BenchmarkSidecarTests(unittest.TestCase):
    def test_gap_measurement_main_honors_custom_sidecar_paths(self) -> None:
        try:
            records = generate_macrocycle_linker_candidates(
                [
                    {
                        "mol_id": "seed1",
                        "canonical_smiles": "CC(C)(C)c1nc2c3ccc(F)cc3c3c(=O)[nH]ccc3c2[nH]1",
                    }
                ],
                max_products_per_seed=1,
            )
        except RuntimeError as exc:
            self.skipTest(str(exc))
        if not records:
            self.skipTest("No macrocycle candidate could be generated for sidecar benchmark test")

        candidate = records[0]
        linkers = extract_linker_smiles_from_smiles(candidate["canonical_smiles"], mol_id=candidate["candidate_id"])
        if not linkers:
            self.skipTest("Could not extract linker identities from generated candidate")

        module = load_gap_module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            generated_csv = base / "generated.csv"
            attempt_log = base / "attempt_log.csv"
            output_csv = base / "novelty.csv"
            summary_json = base / "summary.json"
            train_a = base / "train_a.csv"
            train_b = base / "train_b.csv"

            with generated_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["candidate_id", "canonical_smiles"])
                writer.writeheader()
                writer.writerow({"candidate_id": candidate["candidate_id"], "canonical_smiles": candidate["canonical_smiles"]})

            with attempt_log.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["status"])
                writer.writeheader()
                writer.writerow({"status": "valid_output"})
                writer.writerow({"status": "rejected_known_linker"})
                writer.writerow({"status": "ring_closure_failed"})

            for path in [train_a, train_b]:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["linker_smiles"])
                    writer.writeheader()
                    for linker in linkers:
                        writer.writerow({"linker_smiles": linker})

            argv_backup = sys.argv[:]
            try:
                sys.argv = [
                    str(ROOT / "scripts" / "20_measure_v5_3_benchmark_gaps.py"),
                    "--base",
                    str(base),
                    "--generated-csv",
                    str(generated_csv),
                    "--attempt-log",
                    str(attempt_log),
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--training-fragment-csv",
                    str(train_a),
                    "--training-fragment-csv",
                    str(train_b),
                ]
                module.main()
            finally:
                sys.argv = argv_backup

            summary = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(summary["generated_candidates"], 1)
            self.assertAlmostEqual(float(summary["raw_attempt_validity_percent"]), 33.3333, places=3)
            self.assertEqual(float(summary["linker_novelty_percent"]), 0.0)
            self.assertTrue(output_csv.exists())


if __name__ == "__main__":
    unittest.main()
