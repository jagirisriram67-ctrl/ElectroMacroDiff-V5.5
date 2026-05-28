"""Data collection helpers for JAK2 ligand and structure inputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .chemistry import canonicalize_smiles, mol_id_from_smiles, p_activity_from_nm, summarize_molecule

CHEMBL_ACTIVITY_TYPES = {"IC50", "KI", "KD"}
LIKELY_JAK2_TARGET_IDS = ["CHEMBL2971"]


def _requests():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Install requests for online data collection") from exc
    return requests


def convert_activity_to_nm(value: Any, units: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    unit = (units or "").strip().lower().replace("µ", "u")
    if unit in {"nm", "nanomolar"}:
        return numeric
    if unit in {"um", "µm", "micromolar"}:
        return numeric * 1000.0
    if unit in {"mm", "millimolar"}:
        return numeric * 1_000_000.0
    if unit in {"m", "molar"}:
        return numeric * 1_000_000_000.0
    return None


def fetch_chembl_target_ids(query: str = "JAK2") -> list[str]:
    """Return likely ChEMBL target IDs for human JAK2."""

    requests = _requests()
    url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
    params = {"pref_name__icontains": query, "limit": 20}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    targets = payload.get("targets", [])
    ids: list[str] = []
    for target in targets:
        name = str(target.get("pref_name", "")).lower()
        organism = str(target.get("organism", "")).lower()
        target_id = target.get("target_chembl_id")
        if target_id and "jak2" in name and ("homo sapiens" in organism or organism == ""):
            ids.append(str(target_id))
    return ids or LIKELY_JAK2_TARGET_IDS


def fetch_chembl_activities(
    target_ids: list[str] | None = None,
    limit: int = 1000,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch JAK2 activity records from the public ChEMBL API."""

    requests = _requests()
    target_ids = target_ids or fetch_chembl_target_ids("JAK2")
    all_records: list[dict[str, Any]] = []
    per_target_limit = max(1, limit // max(len(target_ids), 1))
    for target_id in target_ids:
        url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        offset = 0
        while len(all_records) < limit:
            params = {
                "target_chembl_id": target_id,
                "standard_type__in": ",".join(sorted(CHEMBL_ACTIVITY_TYPES)),
                "limit": min(1000, per_target_limit),
                "offset": offset,
            }
            response = requests.get(url, params=params, timeout=90)
            response.raise_for_status()
            payload = response.json()
            records = payload.get("activities", [])
            if not records:
                break
            all_records.extend(records)
            next_url = payload.get("page_meta", {}).get("next")
            if not next_url or len(all_records) >= limit:
                break
            offset += len(records)

    trimmed = all_records[:limit]
    if output_path:
        write_dicts_csv(output_path, trimmed)
    return trimmed


def download_pdb(pdb_id: str, output_path: str | Path) -> Path:
    """Download a PDB structure from RCSB."""

    requests = _requests()
    pdb_id = pdb_id.upper()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    output.write_text(response.text, encoding="utf-8")
    return output


def write_dicts_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        output.write_text("", encoding="utf-8")
        return
    columns = sorted({key for record in records for key in record})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def curate_chembl_records(records: list[dict[str, Any]], max_records: int | None = None) -> list[dict[str, Any]]:
    """Convert raw ChEMBL activity records into the V5.2 curated schema."""

    curated: dict[str, dict[str, Any]] = {}
    for record in records:
        smiles = record.get("canonical_smiles")
        if not smiles:
            molecule = record.get("molecule_chembl_id")
            smiles = record.get("molecule_structures", {}).get("canonical_smiles") if molecule else None
        if not smiles:
            continue

        canonical = canonicalize_smiles(str(smiles))
        if canonical is None:
            continue
        summary = summarize_molecule(canonical)
        if summary is None:
            continue

        value_nm = convert_activity_to_nm(record.get("standard_value"), record.get("standard_units"))
        p_activity = p_activity_from_nm(value_nm)
        if value_nm is None or p_activity is None:
            continue

        source_id = str(record.get("molecule_chembl_id") or mol_id_from_smiles(canonical, "CHEMBL"))
        existing = curated.get(summary.inchikey)
        candidate = {
            "mol_id": mol_id_from_smiles(canonical, "JAK2"),
            "source": "ChEMBL",
            "source_id": source_id,
            "canonical_smiles": canonical,
            "inchikey": summary.inchikey,
            "activity_type": record.get("standard_type", ""),
            "activity_value_nM": round(value_nm, 6),
            "p_activity": round(p_activity, 6),
            "target": "JAK2",
            "assay_id": record.get("assay_chembl_id", ""),
            "confidence_score": record.get("confidence_score", ""),
            "max_ring_size": summary.max_ring_size,
            "has_macrocycle_12_20": summary.has_macrocycle_12_20,
            "has_constrained_ring_8_11": summary.has_constrained_ring_8_11,
            "split": "",
            "notes": "curated_from_chembl_api",
        }
        if existing is None or float(candidate["p_activity"]) > float(existing["p_activity"]):
            curated[summary.inchikey] = candidate

    rows = list(curated.values())
    rows.sort(key=lambda row: float(row["p_activity"]), reverse=True)
    return rows[:max_records] if max_records else rows
