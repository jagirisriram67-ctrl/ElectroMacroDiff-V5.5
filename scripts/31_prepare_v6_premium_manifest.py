"""Prepare the V6 premium dataset manifest.

This script scans project folders or Kaggle-mounted dataset paths and writes a
small manifest describing which V6 training lanes are ready.  It never downloads
large/licensed datasets and never mutates V5.5 outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan V6 premium datasets and write manifest.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--dataset-root", action="append", default=[], help="Generic root to scan for all datasets.")
    parser.add_argument("--activity-root", action="append", default=[], help="ChEMBL/BindingDB JAK activity root.")
    parser.add_argument("--pdbbind-root", action="append", default=[], help="PDBbind root or extracted folder.")
    parser.add_argument("--plinder-root", action="append", default=[], help="PLINDER root or v2 folder.")
    parser.add_argument("--biolip2-root", action="append", default=[], help="BioLiP2 root.")
    parser.add_argument("--crossdocked-root", action="append", default=[], help="CrossDocked2020 root.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    from emd_v5_2_hybrid.v6_premium_datasets import scan_all_datasets, write_v6_manifest

    base = Path(args.base).resolve()
    shared_roots = [Path(p).resolve() for p in args.dataset_root]
    dataset_roots = {
        "chembl_bindingdb_jak": [*shared_roots, *[Path(p).resolve() for p in args.activity_root]],
        "pdbbind": [*shared_roots, *[Path(p).resolve() for p in args.pdbbind_root]],
        "plinder": [*shared_roots, *[Path(p).resolve() for p in args.plinder_root]],
        "biolip2": [*shared_roots, *[Path(p).resolve() for p in args.biolip2_root]],
        "crossdocked2020": [*shared_roots, *[Path(p).resolve() for p in args.crossdocked_root]],
    }

    results = scan_all_datasets(base, dataset_roots=dataset_roots)
    summary = write_v6_manifest(
        base,
        results,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
    )

    print("V6 premium dataset manifest written.")
    print(json.dumps(summary, indent=2))
    for row in results:
        status = "READY" if row.ready_for_training else "MISSING"
        print(f"{status:7s} {row.dataset_key:22s} {row.notes}")


if __name__ == "__main__":
    main()
