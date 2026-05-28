from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.ranking import docking_to_score, rank_dataframe
from emd_v5_2_hybrid.registry import register_artifact, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge scores and create final ranked candidates.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--ignore-docking", action="store_true", help="Rank without existing docking scores.")
    parser.add_argument("--candidates-csv", default=None)
    parser.add_argument("--docking-csv", default=None)
    parser.add_argument("--pose-sanity-csv", default=None)
    parser.add_argument("--admet-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--stage", default="M7_ranking")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    candidates_path = (
        Path(args.candidates_csv).resolve()
        if args.candidates_csv
        else base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
    )
    docking_path = Path(args.docking_csv).resolve() if args.docking_csv else base / "06_docking" / "scores" / "docking_scores.csv"
    pose_sanity_path = (
        Path(args.pose_sanity_csv).resolve()
        if args.pose_sanity_csv
        else base / "06_docking" / "scores" / "pose_sanity_scores.csv"
    )
    admet_path = Path(args.admet_csv).resolve() if args.admet_csv else base / "07_admet_synthesis" / "admet_scores.csv"
    output_path = Path(args.output_csv).resolve() if args.output_csv else base / "08_final_ranking" / "final_ranked_candidates.csv"

    candidates = pd.read_csv(candidates_path)
    docking = (
        pd.DataFrame(columns=["candidate_id", "best_score"])
        if args.ignore_docking or not docking_path.exists()
        else pd.read_csv(docking_path)
    )
    admet = pd.read_csv(admet_path) if admet_path.exists() else pd.DataFrame(columns=["candidate_id"])

    merged = candidates.merge(
        docking[["candidate_id", "best_score"]] if "best_score" in docking.columns else docking,
        on="candidate_id",
        how="left",
    )
    if not admet.empty:
        merged = merged.merge(admet, on="candidate_id", how="left", suffixes=("", "_admet"))
    if pose_sanity_path.exists():
        pose = pd.read_csv(pose_sanity_path)
        keep = [column for column in ["candidate_id", "pose_score", "pose_decision", "contacts_within_4A", "hard_clashes_lt_1_8A"] if column in pose.columns]
        if keep:
            merged = merged.merge(pose[keep], on="candidate_id", how="left", suffixes=("", "_pose_sanity"))

    if "smiles" not in merged.columns and "canonical_smiles" in merged.columns:
        merged["smiles"] = merged["canonical_smiles"]
    if "best_score" in merged.columns and "docking_score" not in merged.columns:
        merged["docking_score"] = merged["best_score"]
    if "diversity_cluster" not in merged.columns:
        cluster_source = merged.get("murcko_scaffold", merged.get("smiles", merged["candidate_id"]))
        merged["diversity_cluster"] = [
            "scaffold_" + hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]
            for value in cluster_source.fillna(merged["candidate_id"])
        ]
    merged["docking_score_scaled"] = docking_to_score(merged.get("best_score", pd.Series([None] * len(merged))))
    merged["pose_score"] = merged.get("pose_score", 0.5).fillna(0.5)
    merged["admet_score"] = merged.get("admet_score", merged.get("qed", 0.5)).fillna(0.5)
    merged["synthesis_score"] = merged.get("synthesis_score", 1.0 - ((merged.get("sa_score", 5.0).fillna(5.0) - 1.0) / 9.0))
    merged["novelty_score"] = merged.get("novelty_score", merged.get("novel_flag", True).astype(float))
    merged["safety_proxy_score"] = merged.get("safety_proxy_score", 0.5)

    ranked = rank_dataframe(merged)
    docking_available = "best_score" in ranked.columns and ranked["best_score"].notna().any()
    ranked["diverse_rank"] = 0
    ranked["selection_tier"] = "await_docking" if not docking_available else "not_selected"
    seen_clusters = set()
    diverse_rank = 1
    for idx, row in ranked.iterrows():
        cluster = row.get("diversity_cluster", "")
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        ranked.at[idx, "diverse_rank"] = diverse_rank
        if docking_available:
            ranked.at[idx, "selection_tier"] = "final_candidate" if diverse_rank <= 5 else "backup_candidate"
        diverse_rank += 1
    if not docking_available:
        ranked["decision"] = "await_docking"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)
    register_artifact(base, args.stage, output_path, "final_ranked_candidates", owner="Student 1")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=f"{candidates_path}; {docking_path}; {admet_path}",
        output_path=str(output_path),
        molecules_in=len(merged),
        molecules_out=len(ranked),
        notes="Consensus ranking generated",
    )
    print(f"Wrote ranked candidates: {output_path}")


if __name__ == "__main__":
    main()
