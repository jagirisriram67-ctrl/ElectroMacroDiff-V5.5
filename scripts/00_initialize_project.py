from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import ensure_project_tree, register_artifact, register_run, write_environment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the EMD V5.2 Hybrid project tree.")
    parser.add_argument("--base", default=str(ROOT), help="Project base directory")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    ensure_project_tree(base)

    root_requirements = ROOT / "requirements_v5_2_hybrid_colab.txt"
    registry_requirements = base / "00_project_registry" / "requirements_v5_2_hybrid_colab.txt"
    if root_requirements.exists() and root_requirements.resolve() != registry_requirements.resolve():
        shutil.copy2(root_requirements, registry_requirements)
        register_artifact(
            base,
            stage="M0_registry",
            path=registry_requirements,
            artifact_type="requirements",
            owner="Student 1",
            notes="Colab requirements copied into registry",
        )

    env_path = write_environment_report(base)
    register_artifact(
        base,
        stage="M0_registry",
        path=env_path,
        artifact_type="environment_report",
        owner="Student 1",
    )
    register_run(
        base,
        stage="M0_registry",
        status="completed",
        output_path=str(env_path),
        notes="Project tree and registry initialized",
    )
    print(f"Initialized project at: {base}")
    print(f"Environment report: {env_path}")


if __name__ == "__main__":
    main()
