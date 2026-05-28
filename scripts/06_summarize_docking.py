from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize parsed Vina docking scores.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    docking_path = base / "06_docking" / "scores" / "docking_scores.csv"
    ranking_path = base / "08_final_ranking" / "final_ranked_candidates.csv"
    out_dir = base / "06_docking" / "scores"
    summary_path = out_dir / "docking_summary.json"
    top_scores_path = out_dir / "top_docking_scores.csv"
    final_docked_path = base / "08_final_ranking" / "top_docked_ranked_candidates.csv"

    docking = pd.read_csv(docking_path)
    docking["best_score"] = pd.to_numeric(docking["best_score"], errors="coerce")
    scored = docking[docking["best_score"].notna()].copy()
    top_scores = scored.sort_values("best_score", ascending=True).head(args.top_n)
    top_scores.to_csv(top_scores_path, index=False)

    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        ranked_docked = ranking[pd.to_numeric(ranking.get("docking_score"), errors="coerce").notna()].copy()
        ranked_docked = ranked_docked.sort_values("rank", ascending=True).head(args.top_n)
        ranked_docked.to_csv(final_docked_path, index=False)

    summary = {
        "num_manifest_ligands": int(len(docking)),
        "num_scored": int(len(scored)),
        "num_missing_or_pending": int(len(docking) - len(scored)),
        "best_score_min": None if scored.empty else float(scored["best_score"].min()),
        "best_score_median": None if scored.empty else float(scored["best_score"].median()),
        "best_score_max": None if scored.empty else float(scored["best_score"].max()),
        "top_candidate_id": None if scored.empty else str(top_scores.iloc[0]["candidate_id"]),
        "top_candidate_score": None if scored.empty else float(top_scores.iloc[0]["best_score"]),
        "notes": "Vina scores are docking proxies and require pose inspection plus experimental validation.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    register_artifact(base, "M5_docking", summary_path, "docking_summary", owner="Student 5")
    register_artifact(base, "M5_docking", top_scores_path, "top_docking_scores", owner="Student 5")
    if final_docked_path.exists():
        register_artifact(base, "M7_ranking", final_docked_path, "top_docked_ranked_candidates", owner="Student 1")
    register_run(
        base,
        stage="M5_docking_summary",
        status="completed",
        input_path=str(docking_path),
        output_path=str(summary_path),
        molecules_in=len(docking),
        molecules_out=len(scored),
        notes=f"Top score {summary['top_candidate_score']} for {summary['top_candidate_id']}",
    )
    print(json.dumps(summary, indent=2))
    print(f"Top docking scores: {top_scores_path}")


if __name__ == "__main__":
    main()
