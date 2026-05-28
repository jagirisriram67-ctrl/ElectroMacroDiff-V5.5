from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.admet_synthesis import profile_to_dict, score_admet_from_descriptors
from emd_v5_2_hybrid.chemistry import summarize_molecule
from emd_v5_2_hybrid.registry import register_artifact, register_run


def safe_float(value, fallback: float) -> float:
    import math

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return numeric if math.isfinite(numeric) else fallback


def safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generated candidates for ADMET and synthesis proxies.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--candidates-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--flags-csv", default=None)
    parser.add_argument("--notes-csv", default=None)
    parser.add_argument("--stage", default="M6_admet_synthesis")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    candidates_path = (
        Path(args.candidates_csv).resolve()
        if args.candidates_csv
        else base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
    )
    output_path = Path(args.output_csv).resolve() if args.output_csv else base / "07_admet_synthesis" / "admet_scores.csv"
    flags_path = Path(args.flags_csv).resolve() if args.flags_csv else base / "07_admet_synthesis" / "filter_flags.csv"
    notes_path = Path(args.notes_csv).resolve() if args.notes_csv else base / "07_admet_synthesis" / "safety_proxy_notes.csv"

    candidates = pd.read_csv(candidates_path)
    rows = []
    for row in candidates.to_dict(orient="records"):
        summary = summarize_molecule(str(row.get("canonical_smiles", row.get("smiles", ""))))
        hbd = row.get("hbd", None)
        hba = row.get("hba", None)
        rotatable = row.get("rotatable_bonds", None)
        if summary is not None:
            hbd = summary.hbd if hbd in (None, "") or pd.isna(hbd) else hbd
            hba = summary.hba if hba in (None, "") or pd.isna(hba) else hba
            rotatable = summary.rotatable_bonds if rotatable in (None, "") or pd.isna(rotatable) else rotatable
        profile = score_admet_from_descriptors(
            qed=safe_float(row.get("qed"), summary.qed if summary else 0.5),
            sa_score=safe_float(row.get("sa_score"), summary.sa_score if summary else 5.0),
            mw=safe_float(row.get("mw"), summary.mw if summary else 0.0),
            logp=safe_float(row.get("logp"), summary.logp if summary else 0.0),
            tpsa=safe_float(row.get("tpsa"), summary.tpsa if summary else 0.0),
            hbd=safe_int(hbd, summary.hbd if summary else 0),
            hba=safe_int(hba, summary.hba if summary else 0),
            rotatable_bonds=safe_int(rotatable, summary.rotatable_bonds if summary else 0),
        )
        item = {"candidate_id": row["candidate_id"], **profile_to_dict(profile)}
        rows.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored = pd.DataFrame(rows)
    scored.to_csv(output_path, index=False)
    scored[["candidate_id", "lipinski_violations", "veber_pass", "main_risk"]].to_csv(flags_path, index=False)
    scored[["candidate_id", "safety_proxy_score", "main_risk"]].to_csv(notes_path, index=False)

    register_artifact(base, args.stage, output_path, "admet_scores", owner="Student 5")
    register_artifact(base, args.stage, flags_path, "filter_flags", owner="Student 5")
    register_artifact(base, args.stage, notes_path, "safety_proxy_notes", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=str(candidates_path),
        output_path=str(output_path),
        molecules_in=len(candidates),
        molecules_out=len(scored),
        notes="Descriptor proxy scoring; not biological safety proof",
    )
    print(f"Wrote ADMET scores: {output_path}")


if __name__ == "__main__":
    main()
