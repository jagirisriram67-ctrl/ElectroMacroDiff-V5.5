"""Resume-safe project registry and progress helpers."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .schemas import ARTIFACT_COLUMNS, PROJECT_DIRS, REGISTRY_COLUMNS


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(stage: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in stage.lower()).strip("_")
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{cleaned}_{uuid.uuid4().hex[:8]}"


def ensure_project_tree(base: str | Path) -> list[Path]:
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for rel in PROJECT_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    ensure_csv(root / "00_project_registry" / "run_registry.csv", REGISTRY_COLUMNS)
    ensure_csv(root / "00_project_registry" / "artifact_registry.csv", ARTIFACT_COLUMNS)
    return created


def ensure_csv(path: str | Path, columns: list[str]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)


def append_row(path: str | Path, columns: list[str], row: dict[str, Any]) -> None:
    ensure_csv(path, columns)
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writerow({column: row.get(column, "") for column in columns})


def register_run(
    base: str | Path,
    stage: str,
    status: str,
    input_path: str = "",
    output_path: str = "",
    molecules_in: int | str = "",
    molecules_out: int | str = "",
    random_seed: int | str = 42,
    notes: str = "",
    run_id: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> str:
    run_id = run_id or make_run_id(stage)
    row = {
        "run_id": run_id,
        "stage": stage,
        "started_at": started_at or utc_timestamp(),
        "finished_at": finished_at or utc_timestamp(),
        "status": status,
        "input_path": input_path,
        "output_path": output_path,
        "molecules_in": molecules_in,
        "molecules_out": molecules_out,
        "random_seed": random_seed,
        "notes": notes,
    }
    append_row(Path(base) / "00_project_registry" / "run_registry.csv", REGISTRY_COLUMNS, row)
    return run_id


def register_artifact(
    base: str | Path,
    stage: str,
    path: str | Path,
    artifact_type: str,
    owner: str = "",
    status: str = "created",
    notes: str = "",
) -> str:
    artifact_id = f"art_{uuid.uuid4().hex[:10]}"
    row = {
        "artifact_id": artifact_id,
        "stage": stage,
        "path": str(path),
        "artifact_type": artifact_type,
        "owner": owner,
        "created_at": utc_timestamp(),
        "status": status,
        "notes": notes,
    }
    append_row(Path(base) / "00_project_registry" / "artifact_registry.csv", ARTIFACT_COLUMNS, row)
    return artifact_id


def load_progress(path: str | Path) -> dict[str, Any]:
    progress_path = Path(path)
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"last_completed_index": -1}


def save_progress(path: str | Path, progress: dict[str, Any]) -> None:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(progress)
    payload["updated_at"] = utc_timestamp()
    with progress_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def environment_report() -> str:
    lines = [
        f"timestamp_utc: {utc_timestamp()}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"platform: {platform.platform()}",
        f"executable: {sys.executable}",
    ]
    for module in ["numpy", "pandas", "rdkit", "selfies", "torch", "e3nn", "meeko", "gemmi", "vina"]:
        lines.append(f"{module}: {module_version(module)}")
    return "\n".join(lines) + "\n"


def module_version(module: str) -> str:
    try:
        imported = __import__(module)
    except Exception as exc:  # pragma: no cover - depends on local environment
        return f"not_available ({exc.__class__.__name__})"
    return str(getattr(imported, "__version__", "available_unknown_version"))


def write_environment_report(base: str | Path) -> Path:
    output = Path(base) / "00_project_registry" / "environment_versions.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(environment_report(), encoding="utf-8")
    return output


def git_revision(base: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(base),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "git_not_available"
    if result.returncode != 0:
        return "not_a_git_repository"
    return result.stdout.strip()
