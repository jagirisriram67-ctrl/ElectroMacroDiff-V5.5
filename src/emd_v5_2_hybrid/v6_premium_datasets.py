"""V6 premium dataset discovery and planning helpers.

The V6 upgrade is intentionally dataset-first.  These utilities do not try to
download licensed or very large structure datasets.  Instead, they inspect
user-provided local/Kaggle mounts, write an auditable manifest, and decide
which V6 training lanes are ready to run.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one V6 premium dataset lane."""

    key: str
    label: str
    priority: int
    role: str
    ready_signal: str
    notes: str


@dataclass
class DatasetScanResult:
    """Compact scan result for one dataset."""

    dataset_key: str
    label: str
    status: str
    root: str
    discovered_files: int
    discovered_records: int | None
    ready_for_training: bool
    training_lane: str
    notes: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "chembl_bindingdb_jak": DatasetSpec(
        key="chembl_bindingdb_jak",
        label="ChEMBL + BindingDB JAK/JAK2 activity table",
        priority=1,
        role="Activity and reward-model supervision for JAK/JAK2 ligands.",
        ready_signal="A CSV/TSV with SMILES plus activity values for JAK/JAK2.",
        notes="Small enough for Kaggle; this should be the first V6 data lane.",
    ),
    "pdbbind": DatasetSpec(
        key="pdbbind",
        label="PDBbind v2020 refined/general structures",
        priority=2,
        role="Protein-ligand affinity and pose-quality supervision.",
        ready_signal="PDBbind INDEX file plus complex folders containing protein and ligand files.",
        notes="Higher quality than CrossDocked, but licensing and user download are required.",
    ),
    "plinder": DatasetSpec(
        key="plinder",
        label="PLINDER protein-ligand interaction systems",
        priority=3,
        role="Residue-level protein-ligand interaction embedding and split control.",
        ready_signal="PLINDER index parquet files and/or systems directory.",
        notes="Best long-term route for learned pocket interaction embeddings.",
    ),
    "biolip2": DatasetSpec(
        key="biolip2",
        label="BioLiP2 biologically relevant ligand-protein interactions",
        priority=4,
        role="Biologically relevant contact and binding-site labels.",
        ready_signal="BioLiP/BioLiP2 annotation table or downloaded structure bundle.",
        notes="Useful for residue-contact labels; format varies by download choice.",
    ),
    "crossdocked2020": DatasetSpec(
        key="crossdocked2020",
        label="CrossDocked2020 large-scale docked poses",
        priority=5,
        role="Large-scale pocket-conditioned generation and pose robustness.",
        ready_signal="CrossDocked split/types files or receptor/ligand archives.",
        notes="Large and noisy; defer until PDBbind/PLINDER lanes are stable.",
    ),
}


def default_dataset_roots(base: str | Path) -> dict[str, list[Path]]:
    """Return conservative default locations to inspect inside the project."""

    base = Path(base)
    raw = base / "01_raw_data"
    premium = raw / "v6_premium"
    return {
        "chembl_bindingdb_jak": [
            raw / "chembl",
            raw / "bindingdb",
            premium / "chembl_bindingdb_jak",
        ],
        "pdbbind": [raw / "pdbbind", premium / "pdbbind"],
        "plinder": [raw / "plinder", premium / "plinder"],
        "biolip2": [raw / "biolip2", raw / "BioLiP", premium / "biolip2"],
        "crossdocked2020": [raw / "crossdocked2020", premium / "crossdocked2020"],
    }


def scan_all_datasets(
    base: str | Path,
    dataset_roots: dict[str, Iterable[str | Path]] | None = None,
) -> list[DatasetScanResult]:
    """Scan all V6 dataset lanes and return manifest rows."""

    roots = default_dataset_roots(base)
    if dataset_roots:
        for key, values in dataset_roots.items():
            roots.setdefault(key, [])
            roots[key].extend(Path(v) for v in values)

    return [
        _scan_dataset(key, [Path(p) for p in roots.get(key, [])])
        for key in sorted(DATASET_SPECS, key=lambda k: DATASET_SPECS[k].priority)
    ]


def write_v6_manifest(
    base: str | Path,
    results: list[DatasetScanResult],
    output_csv: str | Path | None = None,
    summary_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write the V6 dataset manifest and summary JSON."""

    base = Path(base)
    output_csv = Path(output_csv) if output_csv else base / "03_features" / "v6_premium_dataset_manifest.csv"
    summary_json = (
        Path(summary_json) if summary_json else base / "03_features" / "v6_premium_dataset_summary.json"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(results[0]).keys()) if results else list(DatasetScanResult.__annotations__)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    ready = [row.dataset_key for row in results if row.ready_for_training]
    missing = [row.dataset_key for row in results if not row.ready_for_training]
    summary = {
        "phase": "v6_premium_dataset_preparation",
        "manifest_csv": str(output_csv),
        "ready_dataset_count": len(ready),
        "missing_dataset_count": len(missing),
        "ready_datasets": ready,
        "missing_or_incomplete_datasets": missing,
        "recommended_next_lane": _recommend_next_lane(results),
        "training_lanes": v6_training_lanes(),
        "claim_boundary": (
            "V6 preparation enables learned residue-level protein-ligand conditioning; "
            "it does not claim that V6 training has completed until checkpoints and metrics exist."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def v6_training_lanes() -> list[dict[str, str]]:
    """Return the sequential V6 training lanes."""

    return [
        {
            "lane": "account_1_activity_reward",
            "dataset": "chembl_bindingdb_jak",
            "goal": "Train JAK/JAK2 activity and affinity-prior reward models.",
            "acceptance": "ROC-AUC or PR-AUC improves over V5.5 validity-only reward evidence.",
        },
        {
            "lane": "account_2_pdbbind_pose_affinity",
            "dataset": "pdbbind",
            "goal": "Train protein-ligand pose/affinity encoder on high-quality complexes.",
            "acceptance": "Validation Pearson/Spearman and pose-quality classification are reported.",
        },
        {
            "lane": "account_3_plinder_biolip_contacts",
            "dataset": "plinder, biolip2",
            "goal": "Train residue-level contact and pocket interaction encoder.",
            "acceptance": "Residue-contact AUC/F1 beats pocket-count baseline.",
        },
        {
            "lane": "account_4_v6_generator_distillation",
            "dataset": "crossdocked2020 plus prior lanes",
            "goal": "Distill learned pocket encoder into anchor/linker/generation policies.",
            "acceptance": "Logged generation improves raw validity, docking, and pocket-fit metrics.",
        },
    ]


def _scan_dataset(key: str, roots: list[Path]) -> DatasetScanResult:
    spec = DATASET_SPECS[key]
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        return DatasetScanResult(
            dataset_key=key,
            label=spec.label,
            status="missing",
            root=";".join(str(root) for root in roots),
            discovered_files=0,
            discovered_records=None,
            ready_for_training=False,
            training_lane=_lane_for_dataset(key),
            notes=f"Not found. Expected: {spec.ready_signal}",
        )

    scanners = {
        "chembl_bindingdb_jak": _scan_activity_tables,
        "pdbbind": _scan_pdbbind,
        "plinder": _scan_plinder,
        "biolip2": _scan_biolip2,
        "crossdocked2020": _scan_crossdocked,
    }
    status, file_count, record_count, ready, notes = scanners[key](existing_roots)
    return DatasetScanResult(
        dataset_key=key,
        label=spec.label,
        status=status,
        root=";".join(str(root) for root in existing_roots),
        discovered_files=file_count,
        discovered_records=record_count,
        ready_for_training=ready,
        training_lane=_lane_for_dataset(key),
        notes=notes,
    )


def _scan_activity_tables(roots: list[Path]) -> tuple[str, int, int | None, bool, str]:
    patterns = ["*.csv", "*.tsv", "*.txt"]
    files = _find_files(roots, patterns, max_depth=3)
    jak_like = [
        path for path in files
        if any(token in path.name.lower() for token in ("jak", "bindingdb", "chembl", "activity"))
    ]
    record_count = _count_table_rows(jak_like[0]) if jak_like else None
    ready = bool(jak_like)
    return (
        "ready" if ready else "incomplete",
        len(jak_like),
        record_count,
        ready,
        "Found activity-like table(s)." if ready else "No JAK/BindingDB/ChEMBL table found.",
    )


def _scan_pdbbind(roots: list[Path]) -> tuple[str, int, int | None, bool, str]:
    files = _find_files(roots, ["INDEX*data*", "*.pdb", "*.sdf", "*.mol2"], max_depth=4)
    index_files = [path for path in files if path.name.upper().startswith("INDEX")]
    protein_files = [path for path in files if path.name.endswith("_protein.pdb") or "protein" in path.name.lower()]
    ligand_files = [path for path in files if path.suffix.lower() in {".sdf", ".mol2"}]
    record_count = _count_pdbbind_index(index_files[0]) if index_files else None
    ready = bool(index_files and protein_files and ligand_files)
    notes = (
        f"index={len(index_files)}, protein_files={len(protein_files)}, ligand_files={len(ligand_files)}"
    )
    return ("ready" if ready else "incomplete", len(files), record_count, ready, notes)


def _scan_plinder(roots: list[Path]) -> tuple[str, int, int | None, bool, str]:
    files = _find_files(roots, ["*.parquet", "*.csv", "*.json", "*.zip"], max_depth=5)
    index_like = [
        path for path in files
        if any(token in path.name.lower() for token in ("annotation_table", "ligands_per_system"))
    ]
    systems_dirs = [root for root in roots if (root / "systems").exists() or (root / "v2" / "systems").exists()]
    record_count = _count_parquet_rows(index_like[0]) if index_like else None
    ready = bool(index_like or systems_dirs)
    notes = f"index_like={len(index_like)}, systems_dirs={len(systems_dirs)}"
    return ("ready" if ready else "incomplete", len(files), record_count, ready, notes)


def _scan_biolip2(roots: list[Path]) -> tuple[str, int, int | None, bool, str]:
    files = _find_files(roots, ["*.txt", "*.csv", "*.tsv", "*.pdb", "*.cif", "*.gz"], max_depth=4)
    annotation_like = [
        path for path in files
        if any(token in path.name.lower() for token in ("biolip", "ligand", "receptor", "annotation"))
    ]
    record_count = _count_table_rows(annotation_like[0]) if annotation_like else None
    ready = bool(annotation_like)
    return (
        "ready" if ready else "incomplete",
        len(files),
        record_count,
        ready,
        f"annotation_like={len(annotation_like)}",
    )


def _scan_crossdocked(roots: list[Path]) -> tuple[str, int, int | None, bool, str]:
    files = _find_files(roots, ["*.types", "*.sdf", "*.sdf.gz", "*.pdb", "*.tar", "*.tar.gz"], max_depth=4)
    split_files = [path for path in files if path.suffix == ".types" or "split" in path.name.lower()]
    receptor_files = [path for path in files if path.suffix.lower() == ".pdb" and "rec" in path.name.lower()]
    ligand_files = [path for path in files if ".sdf" in path.name.lower()]
    ready = bool((split_files or receptor_files) and ligand_files)
    record_count = _count_table_rows(split_files[0]) if split_files else None
    notes = f"split_files={len(split_files)}, receptor_files={len(receptor_files)}, ligand_files={len(ligand_files)}"
    return ("ready" if ready else "incomplete", len(files), record_count, ready, notes)


def _find_files(roots: Iterable[Path], patterns: Iterable[str], max_depth: int = 4) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            found.append(root)
            continue
        if not root.exists():
            continue
        root_depth = len(root.parts)
        for pattern in patterns:
            for path in root.rglob(pattern):
                if len(path.parts) - root_depth <= max_depth:
                    found.append(path)
    return sorted(set(found))


def _count_table_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return None


def _count_pdbbind_index(path: Path) -> int | None:
    try:
        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    count += 1
        return count
    except OSError:
        return None


def _count_parquet_rows(path: Path) -> int | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        return int(len(pd.read_parquet(path, columns=[])))
    except Exception:
        try:
            return int(len(pd.read_parquet(path)))
        except Exception:
            return None


def _lane_for_dataset(key: str) -> str:
    for lane in v6_training_lanes():
        if key in lane["dataset"]:
            return lane["lane"]
    return "unassigned"


def _recommend_next_lane(results: list[DatasetScanResult]) -> str:
    for result in results:
        if result.ready_for_training:
            return result.training_lane
    return "prepare_account_1_activity_reward_dataset"
