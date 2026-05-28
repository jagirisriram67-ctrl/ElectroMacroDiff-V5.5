"""Project-level validation checks for artifact schemas and sprint gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .schemas import (
    CURATED_LIGAND_COLUMNS,
    DOCKING_SCORE_COLUMNS,
    FINAL_RANKING_COLUMNS,
    GENERATED_CANDIDATE_COLUMNS,
    LIGAND_FEATURE_COLUMNS,
    PROJECT_DIRS,
)


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    message: str


ARTIFACT_SCHEMAS = {
    "02_curated_data/jak2_curated_ligands.csv": CURATED_LIGAND_COLUMNS,
    "03_features/ligand_features.csv": LIGAND_FEATURE_COLUMNS,
    "05_generated_candidates/merged/generated_merged_filtered.csv": GENERATED_CANDIDATE_COLUMNS,
    "06_docking/scores/docking_scores.csv": DOCKING_SCORE_COLUMNS,
    "08_final_ranking/final_ranked_candidates.csv": FINAL_RANKING_COLUMNS,
}


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install pandas to validate CSV artifacts") from exc
    return pd


def validate_project_dirs(base: str | Path) -> list[ValidationResult]:
    root = Path(base)
    results = []
    for rel in PROJECT_DIRS:
        path = root / rel
        results.append(
            ValidationResult(
                name=f"dir:{rel}",
                passed=path.is_dir(),
                message="exists" if path.is_dir() else f"missing directory {path}",
            )
        )
    return results


def validate_csv_schema(path: str | Path, required_columns: list[str]) -> ValidationResult:
    pd = _pandas()
    csv_path = Path(path)
    if not csv_path.exists():
        return ValidationResult(name=f"schema:{csv_path.name}", passed=False, message=f"missing {csv_path}")
    try:
        frame = pd.read_csv(csv_path, nrows=5)
    except Exception as exc:
        return ValidationResult(name=f"schema:{csv_path.name}", passed=False, message=f"read failed: {exc}")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        return ValidationResult(
            name=f"schema:{csv_path.name}",
            passed=False,
            message=f"missing columns: {', '.join(missing)}",
        )
    return ValidationResult(name=f"schema:{csv_path.name}", passed=True, message="schema ok")


def validate_existing_artifacts(base: str | Path) -> list[ValidationResult]:
    root = Path(base)
    results = []
    for rel, columns in ARTIFACT_SCHEMAS.items():
        path = root / rel
        if path.exists():
            results.append(validate_csv_schema(path, columns))
    return results


def validate_sprint_gates(base: str | Path) -> list[ValidationResult]:
    root = Path(base)
    checks = {
        "gate:m0_environment": root / "00_project_registry" / "environment_versions.txt",
        "gate:m1_curated_ligands": root / "02_curated_data" / "jak2_curated_ligands.csv",
        "gate:m1_receptor_pdb": root / "01_raw_data" / "pdb" / "5AEP.pdb",
        "gate:m2_graph_tensors": root / "03_features" / "ligand_graphs" / "se3_graphs.pt",
        "gate:m3_debug_checkpoint": root / "04_models_checkpoints" / "se3_flow" / "se3_latest_checkpoint.pt",
        "gate:m3_debug_json": root / "04_models_checkpoints" / "se3_flow" / "se3_dataloader_test_passed.json",
        "gate:m4_candidates": root / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv",
        "gate:m5_docking_grid": root / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json",
        "gate:m5_ligand_sdf": root / "06_docking" / "ligands_sdf" / "candidates_for_docking.sdf",
        "gate:m5_receptor_pdbqt": root / "06_docking" / "receptor" / "jak2_prepared.pdbqt",
        "gate:m5_ligand_pdbqt_manifest": root / "06_docking" / "ligands_pdbqt" / "ligand_pdbqt_manifest.csv",
        "gate:m5_vina_manifest": root / "06_docking" / "scores" / "vina_command_manifest.csv",
        "gate:m5_docking_scores": root / "06_docking" / "scores" / "docking_scores.csv",
        "gate:m6_admet": root / "07_admet_synthesis" / "admet_scores.csv",
        "gate:m7_ranking": root / "08_final_ranking" / "final_ranked_candidates.csv",
    }
    results = []
    for name, path in checks.items():
        if name == "gate:m5_docking_scores":
            results.append(validate_docking_scores(path))
            continue
        passed = path.exists() and path.stat().st_size > 0
        results.append(
            ValidationResult(
                name=name,
                passed=passed,
                message=str(path) if passed else f"not ready: {path}",
            )
        )
    return results


def validate_docking_scores(path: str | Path) -> ValidationResult:
    pd = _pandas()
    docking_path = Path(path)
    if not docking_path.exists() or docking_path.stat().st_size == 0:
        return ValidationResult(
            name="gate:m5_docking_scores",
            passed=False,
            message=f"not ready: {docking_path}",
        )
    try:
        frame = pd.read_csv(docking_path)
    except Exception as exc:
        return ValidationResult(
            name="gate:m5_docking_scores",
            passed=False,
            message=f"could not read docking scores: {exc}",
        )
    if "best_score" not in frame.columns:
        return ValidationResult(
            name="gate:m5_docking_scores",
            passed=False,
            message="docking_scores.csv missing best_score column",
        )
    parsed = pd.to_numeric(frame["best_score"], errors="coerce").notna().sum()
    if parsed <= 0:
        return ValidationResult(
            name="gate:m5_docking_scores",
            passed=False,
            message="docking_scores.csv exists but contains no parsed numeric scores",
        )
    return ValidationResult(
        name="gate:m5_docking_scores",
        passed=True,
        message=f"{parsed} numeric docking scores parsed",
    )


def validation_summary(results: list[ValidationResult]) -> dict:
    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    return {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "failures": [result.__dict__ for result in results if not result.passed],
    }
