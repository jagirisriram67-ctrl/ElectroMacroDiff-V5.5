from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def copy_and_load_extract_module(temp_interface_dir: Path):
    source = ROOT / "interface" / "extract_data.py"
    target = temp_interface_dir / "extract_data.py"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("extract_data_temp", target)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module, target.parent / "data.js"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class InterfaceExtractTests(unittest.TestCase):
    def build_minimal_project(self, base: Path, include_se3: bool) -> None:
        write_csv(
            base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates.csv",
            ["candidate_id", "canonical_smiles", "pocket_guided_rank", "best_score", "pocket_electronic_fit_score", "pocket_guided_final_score", "lipinski_violations", "veber_pass"],
            [
                {
                    "candidate_id": "CAND_TEST",
                    "canonical_smiles": "CCO",
                    "pocket_guided_rank": 1,
                    "best_score": -8.1,
                    "pocket_electronic_fit_score": 0.7,
                    "pocket_guided_final_score": 0.8,
                    "lipinski_violations": 0,
                    "veber_pass": "True",
                }
            ],
        )
        write_csv(
            base / "09_reports" / "v5_3_benchmark" / "reference_med_style_metrics.csv",
            ["method", "validity_percent", "uniqueness_percent", "macrocyclization_percent", "linker_novelty_percent"],
            [{"method": "MED", "validity_percent": 93.92, "uniqueness_percent": 99.94, "macrocyclization_percent": 99.92, "linker_novelty_percent": 82.81}],
        )
        (base / "09_reports" / "v5_3_benchmark").mkdir(parents=True, exist_ok=True)
        (base / "09_reports" / "v5_3_benchmark" / "emd_v5_3_benchmark_summary.json").write_text(
            json.dumps(
                {
                    "active_branch_for_med_comparison": "kaggle_logged_rerun",
                    "candidate_metrics": [
                        {
                            "branch": "kaggle_logged_rerun",
                            "benchmark_scope": "raw_attempt_logged",
                            "validity_percent": 41.2,
                            "uniqueness_percent": 100.0,
                            "macrocyclization_percent": 100.0,
                            "linker_novelty_percent": 30.5,
                            "attempt_status_counts": {"valid_output": 10, "rejected_known_linker": 4},
                        }
                    ],
                    "docking_metrics": [{"branch": "v5_3_model_guided_vina_gpu_2_1", "best_score": -8.1, "median_score": -8.1, "mean_score": -8.1}],
                    "med_comparison": {"claim_note": "Conservative claim style."},
                }
            ),
            encoding="utf-8",
        )
        (base / "06_docking" / "v5_3_model_guided" / "scores").mkdir(parents=True, exist_ok=True)
        (base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json").write_text(
            json.dumps({"hydrophobic_atom_count": 5}),
            encoding="utf-8",
        )
        write_csv(
            base / "06_docking" / "v5_3_model_guided" / "scores" / "docking_scores_full_vina_gpu_2_1.csv",
            ["candidate_id", "best_score", "grid_center_x", "grid_center_y", "grid_center_z", "grid_size_x", "grid_size_y", "grid_size_z", "receptor_pdb", "docking_engine"],
            [{"candidate_id": "CAND_TEST", "best_score": -8.1, "grid_center_x": 1.0, "grid_center_y": 2.0, "grid_center_z": 3.0, "grid_size_x": 20.0, "grid_size_y": 20.0, "grid_size_z": 20.0, "receptor_pdb": "jak2_prepared.pdbqt", "docking_engine": "vina-gpu"}],
        )
        write_csv(
            base / "02_curated_data" / "jak2_curated_ligands.csv",
            ["mol_id", "canonical_smiles"],
            [{"mol_id": "J1", "canonical_smiles": "CCO"}],
        )
        (base / "06_docking" / "receptor").mkdir(parents=True, exist_ok=True)
        (base / "06_docking" / "receptor" / "jak2_prepared.pdbqt").write_text("", encoding="utf-8")
        if include_se3:
            write_csv(
                base / "06_docking" / "v5_3_model_guided" / "scores" / "se3_geometry_scores.csv",
                ["candidate_id", "se3_geometry_loss", "se3_geometry_confidence", "se3_geometry_status"],
                [{"candidate_id": "CAND_TEST", "se3_geometry_loss": 1.2, "se3_geometry_confidence": 0.45, "se3_geometry_status": "scored"}],
            )

    def test_extract_data_handles_missing_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            interface_dir = base / "interface"
            interface_dir.mkdir(parents=True, exist_ok=True)
            self.build_minimal_project(base, include_se3=False)
            module, data_js = copy_and_load_extract_module(interface_dir)
            module.BASE = str(base)
            module.main()
            text = data_js.read_text(encoding="utf-8")
            self.assertIn("REAL_ACTIVE_BRANCH", text)
            self.assertIn('"available": false', text.lower())

    def test_extract_data_handles_present_se3_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            interface_dir = base / "interface"
            interface_dir.mkdir(parents=True, exist_ok=True)
            self.build_minimal_project(base, include_se3=True)
            module, data_js = copy_and_load_extract_module(interface_dir)
            module.BASE = str(base)
            module.main()
            text = data_js.read_text(encoding="utf-8")
            self.assertIn("REAL_SE3_SUMMARY", text)
            self.assertIn('"available": true', text.lower())


if __name__ == "__main__":
    unittest.main()
