from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run
from emd_v5_2_hybrid.validation import (
    validate_existing_artifacts,
    validate_project_dirs,
    validate_sprint_gates,
    validation_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate project tree, artifact schemas, and sprint gates.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any validation check fails.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    results = []
    results.extend(validate_project_dirs(base))
    results.extend(validate_existing_artifacts(base))
    results.extend(validate_sprint_gates(base))
    summary = validation_summary(results)
    output = base / "00_project_registry" / "validation_report.json"
    output.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [result.__dict__ for result in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    register_artifact(base, "M0_validation", output, "validation_report", owner="Student 1")
    register_run(
        base,
        stage="M0_validation",
        status="completed" if summary["failed"] == 0 else "completed_with_open_gates",
        output_path=str(output),
        notes=f"{summary['passed']} passed, {summary['failed']} open/failed",
    )
    print(json.dumps(summary, indent=2))
    print(f"Validation report: {output}")
    if args.strict and summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
