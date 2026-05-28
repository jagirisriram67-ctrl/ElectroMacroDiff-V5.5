from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run
from emd_v5_2_hybrid.reporting import write_limitations_section, write_markdown_candidate_card


def main() -> None:
    parser = argparse.ArgumentParser(description="Create TPP-style markdown assets for report drafting.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    ranking_path = base / "08_final_ranking" / "final_ranked_candidates.csv"
    output_dir = base / "09_reports" / "candidate_cards"
    limitations_path = base / "09_reports" / "limitations.md"

    ranking_all = pd.read_csv(ranking_path).sort_values("rank")
    selected = ranking_all[ranking_all.get("selection_tier", "") == "final_candidate"].copy()
    if len(selected) >= args.top_n:
        ranking = selected.head(args.top_n)
    else:
        ranking = pd.concat([selected, ranking_all]).drop_duplicates("candidate_id").head(args.top_n)
    card_paths = []
    for row in ranking.to_dict(orient="records"):
        card_paths.append(write_markdown_candidate_card(row, output_dir))
    write_limitations_section(limitations_path)

    for path in card_paths:
        register_artifact(base, "M8_reporting", path, "candidate_card_markdown", owner="Student 1")
    register_artifact(base, "M8_reporting", limitations_path, "limitations_section", owner="Student 1")
    register_run(
        base,
        stage="M8_reporting_assets",
        status="completed",
        input_path=str(ranking_path),
        output_path=str(output_dir),
        molecules_in=len(ranking),
        molecules_out=len(card_paths),
        notes="Markdown candidate cards generated; docking-pending status retained if no scores exist",
    )
    print(f"Candidate cards: {output_dir}")
    print(f"Limitations: {limitations_path}")


if __name__ == "__main__":
    main()
