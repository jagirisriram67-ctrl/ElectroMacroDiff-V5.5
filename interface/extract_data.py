"""Extract only real project outputs into interface/data.js.

The dashboard is a project artifact.  It must not invent benchmark values,
candidate rows, docking scores, or claims.  Missing values are exported as null
and shown as "not available" by the interface.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "interface" / "data.js"


def base_path() -> Path:
    return Path(BASE)


def out_path() -> Path:
    return base_path() / "interface" / "data.js"


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    return None if number is None else int(number)


def safe_bool(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def candidate_sort_key(row: dict) -> tuple:
    rank = row.get("rank")
    return (rank is None, rank or 10**9, row.get("candidate_id", ""))


def load_candidates() -> tuple[list[dict], Path | None]:
    base = base_path()
    ranked_file = first_existing(
        [
            base / "08_final_ranking" / "v5_5_pocket_guided_pocket_electronic_ranked_candidates_with_se3.csv",
            base / "08_final_ranking" / "v5_5_pocket_guided_pocket_electronic_ranked_candidates.csv",
            base / "08_final_ranking" / "v5_5_pocket_guided_ranked_candidates.csv",
            base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates_with_se3.csv",
            base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates.csv",
            base / "08_final_ranking" / "v5_3_model_guided_ranked_candidates.csv",
        ]
    )
    if ranked_file is None:
        return [], None

    rows = []
    for row in load_csv_rows(ranked_file):
        best_score = safe_float(row.get("best_score") or row.get("docking_score"))
        pocket_fit = safe_float(row.get("pocket_electronic_fit_score"))
        final_score = safe_float(row.get("pocket_guided_final_score") or row.get("final_weighted_score"))
        candidate = {
            "rank": safe_int(row.get("pocket_guided_rank") or row.get("rank")),
            "candidate_id": row.get("candidate_id", ""),
            "source": row.get("source_generator", ""),
            "parent_mol_id": row.get("parent_mol_id", ""),
            "smiles": row.get("canonical_smiles") or row.get("smiles", ""),
            "inchikey": row.get("inchikey", ""),
            "best_score": best_score,
            "has_docking": best_score is not None,
            "pocket_electronic_fit_score": pocket_fit,
            "has_pocket_score": pocket_fit is not None,
            "final_weighted_score": safe_float(row.get("final_weighted_score")),
            "pocket_guided_final_score": final_score,
            "decision": row.get("pocket_guided_decision") or row.get("decision") or row.get("selection_tier", ""),
            "selection_tier": row.get("selection_tier", ""),
            "pose_score": safe_float(row.get("pose_score")),
            "pose_decision": row.get("pose_decision", ""),
            "mw": safe_float(row.get("mw")),
            "logp": safe_float(row.get("logp")),
            "tpsa": safe_float(row.get("tpsa")),
            "hbd": safe_int(row.get("hbd")),
            "hba": safe_int(row.get("hba")),
            "rotatable_bonds": safe_int(row.get("rotatable_bonds")),
            "qed": safe_float(row.get("qed")),
            "sa_score": safe_float(row.get("sa_score")),
            "admet_score": safe_float(row.get("admet_score")),
            "synthesis_score": safe_float(row.get("synthesis_score")),
            "safety_proxy_score": safe_float(row.get("safety_proxy_score")),
            "novelty_score": safe_float(row.get("novelty_score")),
            "lipinski_violations": safe_int(row.get("lipinski_violations")),
            "veber_pass": safe_bool(row.get("veber_pass")),
            "max_ring_size": safe_int(row.get("max_ring_size")),
            "has_macrocycle": safe_bool(row.get("has_macrocycle_12_20")),
            "electrostatic_score": safe_float(row.get("electrostatic_score")),
            "hbond_score": safe_float(row.get("hbond_score")),
            "hydrophobic_score": safe_float(row.get("hydrophobic_score")),
            "aromatic_score": safe_float(row.get("aromatic_score")),
            "hbond_opportunity_pairs": safe_int(row.get("hbond_opportunity_pairs")),
            "hydrophobic_contacts": safe_int(row.get("hydrophobic_contacts")),
            "aromatic_contacts": safe_int(row.get("aromatic_contacts")),
            "polar_contacts": safe_int(row.get("polar_contacts")),
            "contact_residues": row.get("contact_residues", ""),
            "diversity_cluster": row.get("diversity_cluster", ""),
            "se3_geometry_loss": safe_float(row.get("se3_geometry_loss")),
            "se3_geometry_confidence": safe_float(row.get("se3_geometry_confidence")),
            "se3_geometry_status": row.get("se3_geometry_status", ""),
        }
        rows.append(candidate)

    rows.sort(key=candidate_sort_key)
    return rows, ranked_file


def load_curated_count() -> int:
    path = base_path() / "02_curated_data" / "jak2_curated_ligands.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def load_generation_count() -> int:
    path = base_path() / "05_generated_candidates" / "v5_5_pocket_guided" / "generated_v5_5_pocket_guided.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def numeric_values(candidates: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in candidates if row.get(key) is not None]


def maybe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def maybe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def histogram(values: list[float], width: float, precision: int = 1) -> list[dict]:
    if not values:
        return []
    buckets: dict[float, int] = {}
    for value in values:
        start = math.floor(value / width) * width
        buckets[start] = buckets.get(start, 0) + 1
    rows = []
    for start in sorted(buckets):
        end = start + width
        rows.append(
            {
                "bin": f"{start:.{precision}f} to {end:.{precision}f}",
                "count": buckets[start],
            }
        )
    return rows


def load_branch_metrics() -> tuple[dict, list[dict], dict]:
    benchmark = load_json(base_path() / "09_reports" / "v5_3_benchmark" / "emd_v5_3_benchmark_summary.json")
    metrics = benchmark.get("candidate_metrics", [])
    active_branch = benchmark.get("active_branch_for_med_comparison", "kaggle_v5_5_pocket_guided")
    active = next((row for row in metrics if row.get("branch") == active_branch), {})
    return benchmark, metrics, active


def load_grid() -> dict:
    scores = load_csv_rows(base_path() / "06_docking" / "v5_5_pocket_guided" / "scores" / "docking_scores_full_vina_gpu_2_1.csv")
    if not scores:
        return {}
    row = scores[0]
    return {
        "engine": row.get("docking_engine", ""),
        "receptor": row.get("receptor_pdb", ""),
        "center_x": safe_float(row.get("grid_center_x")),
        "center_y": safe_float(row.get("grid_center_y")),
        "center_z": safe_float(row.get("grid_center_z")),
        "size_x": safe_float(row.get("grid_size_x")),
        "size_y": safe_float(row.get("grid_size_y")),
        "size_z": safe_float(row.get("grid_size_z")),
    }


def load_se3_summary() -> dict:
    base = base_path()
    scores_file = first_existing(
        [
            base / "06_docking" / "v5_5_pocket_guided" / "scores" / "se3_geometry_scores.csv",
            base / "06_docking" / "v5_3_model_guided" / "scores" / "se3_geometry_scores.csv",
        ]
    )
    if scores_file is None:
        return {"available": False, "scored_rows": 0, "source": ""}

    rows = load_csv_rows(scores_file)
    losses = [value for value in (safe_float(row.get("se3_geometry_loss")) for row in rows) if value is not None]
    confidences = [
        value for value in (safe_float(row.get("se3_geometry_confidence")) for row in rows) if value is not None
    ]
    return {
        "available": bool(rows),
        "scored_rows": len(rows),
        "source": str(scores_file),
        "median_se3_geometry_loss": maybe_median(losses),
        "median_se3_geometry_confidence": maybe_median(confidences),
        "best_se3_geometry_confidence": max(confidences) if confidences else None,
    }


def normalize_pose_text(path: Path) -> str:
    lines: list[str] = []
    in_model = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("MODEL"):
                in_model = "1" in stripped.split()
                continue
            if stripped.startswith("ENDMDL") and in_model:
                break
            if stripped.startswith(("ATOM", "HETATM")) and (not in_model or in_model):
                lines.append(stripped)
    return "\n".join(lines) + ("\nEND\n" if lines else "")


def load_pose_data(top_candidates: list[dict]) -> dict[str, str]:
    pose_dir = base_path() / "06_docking" / "v5_5_pocket_guided" / "poses"
    pose_data: dict[str, str] = {}
    for candidate in top_candidates:
        candidate_id = candidate["candidate_id"]
        pose_file = pose_dir / f"{candidate_id}_vina_out.pdbqt"
        if not pose_file.exists():
            continue
        pose_text = normalize_pose_text(pose_file)
        if pose_text:
            pose_data[candidate_id] = pose_text
    return pose_data


def load_pose_paths(candidates: list[dict]) -> dict[str, str]:
    pose_dir = base_path() / "06_docking" / "v5_5_pocket_guided" / "poses"
    pose_paths: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        pose_file = pose_dir / f"{candidate_id}_vina_out.pdbqt"
        if pose_file.exists():
            pose_paths[candidate_id] = f"../06_docking/v5_5_pocket_guided/poses/{candidate_id}_vina_out.pdbqt"
    return pose_paths


def load_protein_pdb() -> str:
    receptor_path = base_path() / "06_docking" / "receptor" / "5AEP_receptor_clean.pdb"
    if not receptor_path.exists():
        return ""
    return receptor_path.read_text(encoding="utf-8", errors="replace")


def validation_summary() -> dict:
    validation = load_json(base_path() / "00_project_registry" / "validation_report.json")
    verification = load_json(base_path() / "00_project_registry" / "final_project_verification.json")
    validation_root = validation.get("summary", validation)
    return {
        "total": validation_root.get("total"),
        "passed": validation_root.get("passed"),
        "failed": validation_root.get("failed"),
        "unit_tests_total": verification.get("unit_tests_total"),
        "unit_tests_passed": verification.get("unit_tests_passed"),
    }


def main() -> None:
    candidates, ranked_file = load_candidates()
    top_candidates = candidates[:20]
    pose_candidates = candidates
    benchmark, branch_metrics, active_metrics = load_branch_metrics()
    docking_summary = load_json(base_path() / "06_docking" / "v5_5_pocket_guided" / "scores" / "v5_5_docking_and_pocket_summary.json")
    generation_summary = load_json(base_path() / "05_generated_candidates" / "v5_5_pocket_guided" / "v5_5_pocket_guided_summary.json")
    parse_summary = load_json(base_path() / "06_docking" / "v5_5_pocket_guided" / "scores" / "v5_5_vina_gpu_parse_summary.json")
    pocket_profile = load_json(base_path() / "06_docking" / "v5_5_pocket_guided" / "scores" / "jak2_pocket_electronic_profile.json")

    docking_values = numeric_values(candidates, "best_score")
    pocket_values = numeric_values(candidates, "pocket_electronic_fit_score")
    qed_values = numeric_values(candidates, "qed")
    mw_values = numeric_values(candidates, "mw")
    logp_values = numeric_values(candidates, "logp")
    final_values = numeric_values(candidates, "pocket_guided_final_score")

    summary = {
        "project": "ElectroMacroDiff",
        "branch": "v5_5_pocket_guided",
        "active_branch_for_benchmark": benchmark.get("active_branch_for_med_comparison"),
        "ranking_source": str(ranked_file) if ranked_file else "",
        "generated_candidates": load_generation_count(),
        "ranked_candidates": len(candidates),
        "curated_jak2_ligands": load_curated_count(),
        "docked_candidates": parse_summary.get("parsed_scores") or sum(1 for c in candidates if c["has_docking"]),
        "raw_pose_files": parse_summary.get("raw_pose_files"),
        "pocket_scored_candidates": docking_summary.get("pocket_scored_top_n") or sum(1 for c in candidates if c["has_pocket_score"]),
        "best_vina_gpu_score": docking_summary.get("best_vina_gpu_score") or maybe_min(docking_values),
        "median_vina_gpu_score": docking_summary.get("median_vina_gpu_score") or maybe_median(docking_values),
        "best_pocket_electronic_fit_score": docking_summary.get("best_pocket_electronic_fit_score") or maybe_min(pocket_values),
        "median_pocket_electronic_fit_score": docking_summary.get("median_pocket_electronic_fit_score") or maybe_median(pocket_values),
        "top_ranked_candidate": candidates[0]["candidate_id"] if candidates else None,
        "top_ranked_final_score": candidates[0]["pocket_guided_final_score"] if candidates else None,
        "raw_attempts": generation_summary.get("total_attempts"),
        "raw_attempt_validity_percent": generation_summary.get("raw_attempt_validity_percent"),
        "linker_novelty_percent": active_metrics.get("linker_novelty_percent"),
        "uniqueness_percent": active_metrics.get("uniqueness_percent"),
        "macrocyclization_percent": active_metrics.get("macrocyclization_percent"),
        "median_mw": maybe_median(mw_values),
        "median_logp": maybe_median(logp_values),
        "median_qed": maybe_median(qed_values),
        "median_final_score": maybe_median(final_values),
        "validation": validation_summary(),
        "truth_note": "All values are exported from project files. Null means the project did not produce that value for that row.",
    }

    admet = {
        "lipinski_0": sum(1 for c in candidates if c.get("lipinski_violations") == 0),
        "lipinski_1": sum(1 for c in candidates if c.get("lipinski_violations") == 1),
        "lipinski_2plus": sum(1 for c in candidates if (c.get("lipinski_violations") or 0) >= 2),
        "veber_pass": sum(1 for c in candidates if c.get("veber_pass") is True),
        "veber_fail": sum(1 for c in candidates if c.get("veber_pass") is False),
    }

    payloads = {
        "REAL_ACTIVE_BRANCH": benchmark.get("active_branch_for_med_comparison") or "v5_5_pocket_guided",
        "REAL_PROJECT_SUMMARY": summary,
        "REAL_ALL_CANDIDATES": candidates,
        "REAL_TOP_CANDIDATES": top_candidates,
        "REAL_BRANCH_METRICS": branch_metrics,
        "REAL_SE3_SUMMARY": load_se3_summary(),
        "REAL_ATTEMPT_STATUS_COUNTS": generation_summary.get("attempt_status_counts", {}),
        "REAL_DOCKING_HIST": histogram(docking_values, width=1.0, precision=1),
        "REAL_POCKET_HIST": histogram(pocket_values, width=0.05, precision=2),
        "REAL_ADMET_SUMMARY": admet,
        "REAL_GRID": load_grid(),
        "REAL_POCKET_PROFILE": pocket_profile,
        "REAL_POSE_DATA": load_pose_data(pose_candidates),
        "REAL_POSE_PATHS": load_pose_paths(candidates),
        "REAL_PROTEIN_PDB": load_protein_pdb(),
    }

    output = out_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("// AUTO-GENERATED from real project outputs. Do not edit manually.\n")
        handle.write(f"// Generated at: {datetime.now().isoformat()}\n\n")
        for name, value in payloads.items():
            handle.write(f"const {name} = ")
            handle.write(json.dumps(value, indent=2))
            handle.write(";\n\n")

    print(
        f"Done: exported {len(candidates)} ranked candidates, "
        f"{summary['docked_candidates']} docked scores, "
        f"ranking={ranked_file}"
    )


if __name__ == "__main__":
    main()
