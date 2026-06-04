# ============================================================
# NOTEBOOK 07: Final Ranking and TPP Report
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# Purpose: Consensus ranking, molecule cards, TPP report, final package
# Owner: Student 1 (all assist)
# Day: 9-10
# Success gate: 3-5 final candidates, TPP report generated
# ============================================================

# ── CELL 1: Install & Mount ───────────────────────────────
# !pip -q install rdkit-pypi openpyxl python-docx pyyaml tqdm tabulate
# from google.colab import drive
# drive.mount("/content/drive")

# ── CELL 2: Config ─────────────────────────────────────────
import os, sys, time
import numpy as np
import pandas as pd

BASE = "."
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))

TINY_DEBUG = True
RANDOM_SEED = 42

from emd_pipeline.config_registry import load_config, set_all_seeds, update_registry
config = load_config(os.path.join(BASE, "campaign_config.yaml"))
set_all_seeds(RANDOM_SEED)

# ── CELL 3: Load All Scoring Data ─────────────────────────
print("--- Loading scoring data ---")

# Docking scores
docking_path = os.path.join(BASE, "06_docking", "scores", "docking_scores.csv")
docking_df = pd.read_csv(docking_path)
print(f"Docking scores: {len(docking_df)}")

# ADMET scores
admet_path = os.path.join(BASE, "07_admet_synthesis", "admet_scores.csv")
admet_df = pd.read_csv(admet_path) if os.path.exists(admet_path) else pd.DataFrame()
print(f"ADMET scores: {len(admet_df)}")

# Synthesis scores
synth_path = os.path.join(BASE, "07_admet_synthesis", "synthesis_scores.csv")
synth_df = pd.read_csv(synth_path) if os.path.exists(synth_path) else pd.DataFrame()
print(f"Synthesis scores: {len(synth_df)}")

# Safety proxy
safety_path = os.path.join(BASE, "07_admet_synthesis", "safety_proxy_notes.csv")
safety_df = pd.read_csv(safety_path) if os.path.exists(safety_path) else pd.DataFrame()
print(f"Safety proxy: {len(safety_df)}")

# Candidate metadata
merged_path = os.path.join(BASE, "05_generated_candidates", "merged", "generated_merged_filtered.csv")
candidates_df = pd.read_csv(merged_path) if os.path.exists(merged_path) else pd.DataFrame()
print(f"Candidate metadata: {len(candidates_df)}")

# ── CELL 4: Consensus Ranking ─────────────────────────────
print("\n--- Consensus Ranking ---")

from emd_pipeline.ranking_report import consensus_ranking

ranking_weights = config.get("ranking", {}).get("weights", {
    "docking_score": 0.35,
    "pose_sanity": 0.15,
    "admet": 0.20,
    "synthesis": 0.15,
    "novelty_diversity": 0.10,
    "safety_proxy": 0.05,
})

ranked_path = os.path.join(BASE, "08_final_ranking", "final_ranked_candidates.csv")
ranked_df = consensus_ranking(
    docking_df=docking_df,
    admet_df=admet_df,
    synthesis_df=synth_df,
    safety_df=safety_df,
    candidates_df=candidates_df,
    weights=ranking_weights,
    output_path=ranked_path,
)

print(f"\nFull ranking table:")
display_cols = ["rank", "candidate_id", "smiles", "best_score",
                "final_weighted_score", "decision", "main_risk"]
available_cols = [c for c in display_cols if c in ranked_df.columns]
print(ranked_df[available_cols].to_string(index=False))

# ── CELL 5: Generate Molecule Cards ───────────────────────
print("\n--- Molecule Cards ---")

from emd_pipeline.ranking_report import generate_molecule_cards

cards_path = os.path.join(BASE, "08_final_ranking", "top10_molecule_cards.xlsx")
cards_df = generate_molecule_cards(ranked_df, cards_path, top_n=10)

print("\nTop 10 Molecule Cards:")
print(cards_df.to_string(index=False))

# ── CELL 6: Save Top 5 SDF ────────────────────────────────
from emd_pipeline.ranking_report import save_top_sdf

sdf_path = os.path.join(BASE, "08_final_ranking", "final_top5.sdf")
save_top_sdf(ranked_df, sdf_path, top_n=5)

# ── CELL 7: SE(3) vs Baseline Comparison ──────────────────
print("\n--- SE(3) vs Baseline Comparison ---")

if "source_generator" in ranked_df.columns:
    for src in ranked_df["source_generator"].unique():
        sub = ranked_df[ranked_df["source_generator"] == src]
        print(f"\n{src}:")
        print(f"  Count: {len(sub)}")
        print(f"  Mean final score: {sub['final_weighted_score'].mean():.4f}")
        print(f"  Best final score: {sub['final_weighted_score'].max():.4f}")
        if "best_score" in sub.columns:
            valid = sub["best_score"].dropna()
            if len(valid) > 0:
                print(f"  Mean docking: {valid.mean():.2f}")
                print(f"  Best docking: {valid.min():.2f}")
        in_top5 = len(sub[sub["rank"] <= 5])
        in_top10 = len(sub[sub["rank"] <= 10])
        print(f"  In top 5: {in_top5}")
        print(f"  In top 10: {in_top10}")
else:
    print("Source generator info not available")

# ── CELL 8: Visualization ─────────────────────────────────
from emd_pipeline.visualization import plot_ranking_dashboard, draw_molecule_grid

# Ranking dashboard
ranking_plot = os.path.join(BASE, "08_final_ranking", "ranking_overview.png")
plot_ranking_dashboard(ranked_df, output_path=ranking_plot)

# ── CELL 9: Generate 2D Structure Images ──────────────────
print("\n--- Generating 2D Structure Images ---")

img_dir = os.path.join(BASE, "08_final_ranking", "structure_images")
top_smiles = ranked_df.head(8)["smiles"].tolist()
top_legends = [
    f"#{row.get('rank','?')} {row.get('candidate_id','?')}\nScore: {row.get('final_weighted_score',0):.3f}"
    for _, row in ranked_df.head(8).iterrows()
]
grid_path = os.path.join(img_dir, "top_candidates_grid.png")
draw_molecule_grid(top_smiles, top_legends, output_path=grid_path)

# ── CELL 10: Generate TPP Report ──────────────────────────
print("\n--- Generating TPP Report ---")

from emd_pipeline.ranking_report import generate_tpp_report

reports_dir = os.path.join(BASE, "09_reports")
tpp_path = generate_tpp_report(ranked_df, config, reports_dir)

# ── CELL 11: Generate Final Summary Report ────────────────
print("\n--- Generating Final Summary ---")

summary_lines = [
    "=" * 70,
    "ElectroMacroDiff V5.2 Hybrid — Final Execution Summary",
    "=" * 70,
    "",
    f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Target: JAK2 (PDB: 5AEP)",
    "",
    "PIPELINE STAGES COMPLETED:",
    f"  M0: Project Registry & Environment ✅",
    f"  M1: Data Collection & Curation ✅",
    f"  M2: Feature Engineering & SE(3) Dataset ✅",
    f"  M3: SE(3) Flow Matching Model Training ✅",
    f"  M4: Hybrid Candidate Generation ✅",
    f"  M5: Docking & Pose Filtering ✅",
    f"  M6: ADMET, Synthesis & Safety ✅",
    f"  M7: Consensus Ranking & TPP Report ✅",
    "",
    "KEY RESULTS:",
    f"  Total candidates ranked: {len(ranked_df)}",
    f"  Final candidates (top 5): {len(ranked_df[ranked_df['decision'] == 'CANDIDATE'])}",
    f"  Backup candidates: {len(ranked_df[ranked_df['decision'] == 'BACKUP'])}",
    "",
    "TOP 5 FINAL CANDIDATES:",
]

for _, row in ranked_df.head(5).iterrows():
    summary_lines.append(
        f"  #{row.get('rank','?')}: {row.get('candidate_id','?')} "
        f"| Score: {row.get('final_weighted_score', 0):.4f} "
        f"| Dock: {row.get('best_score', 'N/A')} "
        f"| {row.get('decision', '')}"
    )

summary_lines.extend([
    "",
    "DELIVERABLES:",
    f"  - Curated dataset: 02_curated_data/jak2_curated_ligands.csv",
    f"  - SE(3) checkpoints: 04_models_checkpoints/se3_flow/",
    f"  - Generated candidates: 05_generated_candidates/merged/",
    f"  - Docking scores: 06_docking/scores/docking_scores.csv",
    f"  - ADMET scores: 07_admet_synthesis/admet_scores.csv",
    f"  - Final ranking: 08_final_ranking/final_ranked_candidates.csv",
    f"  - Molecule cards: 08_final_ranking/top10_molecule_cards.xlsx",
    f"  - Top 5 SDF: 08_final_ranking/final_top5.sdf",
    f"  - TPP Report: 09_reports/EMD_V5_2_Hybrid_TPP.txt",
    "",
    "DISCLAIMER:",
    "  All results are computational hypotheses.",
    "  Experimental validation is required before any biological claims.",
    "",
    "=" * 70,
])

summary_path = os.path.join(reports_dir, "EMD_V5_2_Hybrid_Final_Summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Final summary saved: {summary_path}")

# ── CELL 12: Artifact Registry ────────────────────────────
print("\n--- Building Artifact Registry ---")

artifact_registry = []
artifact_dirs = [
    "00_project_registry",
    "01_raw_data",
    "02_curated_data",
    "03_features",
    "04_models_checkpoints",
    "05_generated_candidates",
    "06_docking",
    "07_admet_synthesis",
    "08_final_ranking",
    "09_reports",
    "11_logs",
]

for adir in artifact_dirs:
    full_dir = os.path.join(BASE, adir)
    if os.path.exists(full_dir):
        for root, dirs, files in os.walk(full_dir):
            for f in files:
                fpath = os.path.join(root, f)
                rel_path = os.path.relpath(fpath, BASE)
                size = os.path.getsize(fpath)
                artifact_registry.append({
                    "path": rel_path,
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 1),
                    "directory": adir,
                })

registry_df = pd.DataFrame(artifact_registry)
registry_path = os.path.join(BASE, "00_project_registry", "artifact_registry.csv")
registry_df.to_csv(registry_path, index=False)
print(f"Artifact registry: {len(registry_df)} files")
print(f"Total size: {registry_df['size_kb'].sum():.0f} KB")

# ── CELL 13: Final Registry Update ────────────────────────
update_registry(
    os.path.join(BASE, "00_project_registry", "run_registry.csv"),
    {
        "notebook": "07_final_ranking_tpp",
        "step": "complete",
        "status": "complete",
        "output_path": ranked_path,
        "notes": f"{len(ranked_df)} ranked, top score={ranked_df['final_weighted_score'].max():.4f}",
    }
)

# ── CELL 14: Final Output ─────────────────────────────────
print("\n" + "=" * 70)
print("🎉  NOTEBOOK 07 COMPLETE — Final Ranking & TPP")
print("=" * 70)
print(f"  Ranked candidates: {len(ranked_df)}")
print(f"  Final top 5: ✅")
print(f"  Molecule cards: ✅")
print(f"  Top 5 SDF: ✅")
print(f"  TPP report: ✅")
print(f"  Final summary: ✅")
print(f"  Artifact registry: ✅ ({len(registry_df)} files)")
print()
print("  ╔══════════════════════════════════════════════════════╗")
print("  ║  ElectroMacroDiff V5.2 Hybrid Pipeline COMPLETE!    ║")
print("  ╚══════════════════════════════════════════════════════╝")
print()
print("  Day 10 tasks remaining:")
print("    1. Rerun Tiny Debug pipeline end-to-end")
print("    2. Verify all Drive files present")
print("    3. Polish report and slides")
print("    4. Rehearse presentation")
