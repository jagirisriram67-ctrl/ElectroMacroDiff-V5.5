"""Write a V6 sequential Kaggle training plan from local evidence.

The output is a handover-ready Markdown plan that uses the current V5.5 metrics
and the V6 premium dataset manifest.  It is deliberately conservative: training
claims are marked pending until real V6 checkpoints and benchmark outputs exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write V6 Kaggle sequential training plan.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--manifest-summary", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    from emd_v5_2_hybrid.v6_premium_datasets import DATASET_SPECS, v6_training_lanes

    base = Path(args.base).resolve()
    summary_path = (
        Path(args.manifest_summary)
        if args.manifest_summary
        else base / "03_features" / "v6_premium_dataset_summary.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else base / "docs" / "V6_KAGGLE_SEQUENTIAL_TRAINING_PLAN.md"
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)

    manifest_summary = _read_json(summary_path, default={})
    v5_metrics = _collect_v5_5_metrics(base)

    lines = [
        "# ElectroMacroDiff V6 Sequential Premium-Dataset Training Plan",
        "",
        "## Locked Direction",
        "",
        "V6 is the premium-data upgrade after the V5.5 pocket/electronic-conditioned proof.",
        "It must be trained one lane at a time, with the four Kaggle accounts used as clean",
        "role-based environments rather than uncontrolled parallel jobs.",
        "",
        "V5.5 stays frozen. V6 writes new artifacts under `v6_*` paths only.",
        "",
        "## Current V5.5 Evidence Baseline",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Anchor GNN F1 | {_fmt(v5_metrics.get('anchor_f1'))} |",
        f"| Anchor GNN precision | {_fmt(v5_metrics.get('anchor_precision'))} |",
        f"| Linker exact accuracy | {_fmt(v5_metrics.get('linker_exact_accuracy'))} |",
        f"| Linker within-one accuracy | {_fmt(v5_metrics.get('linker_within_one_accuracy'))} |",
        f"| Validity reward ROC-AUC | {_fmt(v5_metrics.get('reward_roc_auc'))} |",
        f"| SE(3) continuation val loss | {_fmt(v5_metrics.get('se3_val_loss'))} |",
        f"| V5.5 generated candidates | {_fmt(v5_metrics.get('generated_candidates'))} |",
        f"| V5.5 raw-attempt validity | {_fmt(v5_metrics.get('raw_attempt_validity_percent'))} |",
        f"| V5.5 linker novelty | {_fmt(v5_metrics.get('linker_novelty_percent'))} |",
        "",
        "## Dataset Readiness",
        "",
        "| Dataset | Role | Status |",
        "|---|---|---|",
    ]

    ready = set(manifest_summary.get("ready_datasets", []))
    for key in sorted(DATASET_SPECS, key=lambda k: DATASET_SPECS[k].priority):
        spec = DATASET_SPECS[key]
        status = "ready" if key in ready else "not mounted / incomplete"
        lines.append(f"| `{key}` | {spec.role} | {status} |")

    lines.extend([
        "",
        "## Four Sequential Lanes",
        "",
    ])

    for lane in v6_training_lanes():
        lines.extend([
            f"### {lane['lane']}",
            "",
            f"- Dataset: `{lane['dataset']}`",
            f"- Goal: {lane['goal']}",
            f"- Acceptance: {lane['acceptance']}",
            "- Run style: smoke test first, then long training if the smoke output is valid.",
            "- Output rule: checkpoint and metrics must use `v6_` names and must not overwrite V5.5.",
        ])
        if lane["lane"] == "account_1_activity_reward":
            lines.extend([
                "",
                "Smoke command:",
                "",
                "```bash",
                "!python scripts/33_train_v6_activity_reward.py --base . --device auto --smoke-test --epochs 2 --max-rows 300",
                "```",
                "",
                "Long command:",
                "",
                "```bash",
                "!python scripts/33_train_v6_activity_reward.py --base . --device auto --epochs 800 --batch-size 64 --learning-rate 5e-4 --patience 120",
                "```",
            ])
        lines.append("")

    lines.extend([
        "## Kaggle Notebook Contract",
        "",
        "Each notebook should contain only these sections:",
        "",
        "1. Setup and mounted dataset check.",
        "2. `python scripts/31_prepare_v6_premium_manifest.py --base . ...`.",
        "3. Smoke run for that lane.",
        "4. Long run for that lane.",
        "5. Zip only the lane outputs and metrics.",
        "",
        "Do not put giant dataset downloads in the notebook. Mount/upload the dataset as a Kaggle",
        "dataset first, then pass the mounted path to the manifest script.",
        "",
        "## V6 Claim Boundary",
        "",
        "Allowed after successful V6 training:",
        "",
        "- V6 uses learned protein-ligand and residue-contact evidence to condition generation.",
        "- V6 improves the V5.5 pocket/electronic-conditioned policy with premium structural data.",
        "",
        "Not allowed until a true generative model is trained and benchmarked:",
        "",
        "- Full protein-conditioned diffusion has beaten MED.",
        "- Quantum electron density was used during generation.",
        "- Experimental potency or biological activity is proven.",
        "",
    ])

    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_md}")


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_v5_5_metrics(base: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    acceptance = _read_json(base / "04_models_checkpoints" / "v5_5_model_acceptance_report.json", {})
    if acceptance:
        accepted = acceptance.get("acceptance", acceptance)
        metrics["anchor_f1"] = _nested_get(accepted, "pocket_anchor_gnn", "test_f1")
        metrics["anchor_precision"] = _nested_get(accepted, "pocket_anchor_gnn", "test_precision")
        metrics["linker_exact_accuracy"] = _nested_get(
            accepted, "pocket_linker_policy", "test_size_exact_accuracy"
        )
        metrics["linker_within_one_accuracy"] = _nested_get(
            accepted, "pocket_linker_policy", "test_size_within_one_accuracy"
        )
        metrics["reward_roc_auc"] = _nested_get(accepted, "validity_reward", "test_roc_auc")
        metrics["se3_val_loss"] = _nested_get(accepted, "se3_continuation", "best_val_loss")

    generation = _read_json(
        base / "05_generated_candidates" / "v5_5_pocket_guided" / "v5_5_pocket_guided_summary.json",
        {},
    )
    metrics["generated_candidates"] = (
        generation.get("valid_outputs")
        or generation.get("valid_candidates")
        or generation.get("candidate_count")
    )
    metrics["raw_attempt_validity_percent"] = generation.get("raw_attempt_validity_percent")
    gap_summary = _read_json(
        base / "05_generated_candidates" / "v5_5_pocket_guided" / "v5_5_pocket_guided_benchmark_gap_summary.json",
        {},
    )
    metrics["linker_novelty_percent"] = (
        generation.get("linker_novelty_percent")
        or gap_summary.get("linker_novelty_percent")
    )
    return metrics


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
