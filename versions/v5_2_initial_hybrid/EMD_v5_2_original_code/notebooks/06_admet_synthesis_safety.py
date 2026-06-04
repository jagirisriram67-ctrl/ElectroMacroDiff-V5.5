# ============================================================
# NOTEBOOK 06: ADMET, Synthesis, and Safety
# ElectroMacroDiff V5.2 Hybrid
# ============================================================
# Purpose: Compute ADMET/safety/synthesis scores for docked candidates
# Owner: Student 5 + Student 1
# Day: 8
# Success gate: Top 10 candidates fully scored
# ============================================================

# ── CELL 1: Install & Mount ───────────────────────────────
# !pip -q install rdkit-pypi pyyaml tqdm
# from google.colab import drive
# drive.mount("/content/drive")

# ── CELL 2: Config ─────────────────────────────────────────
import os, sys
import pandas as pd

BASE = "."
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.dirname(BASE))

TINY_DEBUG = True

from emd_pipeline.config_registry import load_config, set_all_seeds, update_registry
config = load_config(os.path.join(BASE, "campaign_config.yaml"))
set_all_seeds(42)

# ── CELL 3: Load Docking Scores ───────────────────────────
scores_path = os.path.join(BASE, "06_docking", "scores", "docking_scores.csv")
scores_df = pd.read_csv(scores_path)
print(f"Loaded {len(scores_df)} docking results")

# Focus on generated (non-control) candidates
gen_mask = scores_df.get("control_or_generated", pd.Series("generated", index=scores_df.index)) != "control"
candidates_for_admet = scores_df[gen_mask].copy()
print(f"Generated candidates for ADMET: {len(candidates_for_admet)}")

# ── CELL 4: Compute ADMET Scores ──────────────────────────
from emd_pipeline.admet_synthesis import compute_admet_scores

admet_path = os.path.join(BASE, "07_admet_synthesis", "admet_scores.csv")
admet_df = compute_admet_scores(
    candidates_for_admet,
    smiles_col="smiles",
    output_path=admet_path,
)

print(f"\nADMET Summary:")
print(admet_df[["qed", "admet_score", "lipinski_violations"]].describe())

# ── CELL 5: Apply Filter Flags ────────────────────────────
from emd_pipeline.admet_synthesis import apply_filter_flags

filter_path = os.path.join(BASE, "07_admet_synthesis", "filter_flags.csv")
filter_df = apply_filter_flags(
    admet_df,
    config=config.get("admet", {}),
    output_path=filter_path,
)

# ── CELL 6: Compute Synthesis Scores ──────────────────────
from emd_pipeline.admet_synthesis import compute_synthesis_scores

synth_path = os.path.join(BASE, "07_admet_synthesis", "synthesis_scores.csv")
synth_df = compute_synthesis_scores(
    candidates_for_admet,
    smiles_col="smiles",
    output_path=synth_path,
)

# ── CELL 7: Compute Safety Proxy ──────────────────────────
from emd_pipeline.admet_synthesis import compute_safety_proxy

safety_path = os.path.join(BASE, "07_admet_synthesis", "safety_proxy_notes.csv")
safety_df = compute_safety_proxy(
    candidates_for_admet,
    smiles_col="smiles",
    output_path=safety_path,
)

# ── CELL 8: Combined Summary ──────────────────────────────
print("\n--- Combined ADMET/Synthesis/Safety Summary ---")

# Merge all scores
combined = admet_df[["candidate_id", "qed", "admet_score", "lipinski_violations", "pains_flag"]].copy()

if len(synth_df) > 0:
    combined = combined.merge(
        synth_df[["candidate_id", "sa_score", "synthesis_score"]],
        on="candidate_id", how="left"
    )

if len(safety_df) > 0:
    combined = combined.merge(
        safety_df[["candidate_id", "safety_proxy_score", "num_alerts"]],
        on="candidate_id", how="left"
    )

print(combined.describe())

# Identify top candidates by composite quality
combined["composite_quality"] = (
    combined.get("admet_score", 0.5).fillna(0.5) * 0.4
    + combined.get("synthesis_score", 0.5).fillna(0.5) * 0.3
    + combined.get("safety_proxy_score", 0.5).fillna(0.5) * 0.3
)

combined = combined.sort_values("composite_quality", ascending=False)
print(f"\nTop 10 by composite quality:")
print(combined.head(10).to_string(index=False))

# ── CELL 9: Visualization ─────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(combined["qed"].dropna(), bins=20, color="#009688", alpha=0.7)
axes[0, 0].set_title("QED Distribution")

if "sa_score" in combined.columns:
    axes[0, 1].hist(combined["sa_score"].dropna(), bins=20, color="#FF5722", alpha=0.7)
    axes[0, 1].set_title("SA Score Distribution")

if "admet_score" in combined.columns:
    axes[1, 0].hist(combined["admet_score"].dropna(), bins=20, color="#3F51B5", alpha=0.7)
    axes[1, 0].set_title("ADMET Score Distribution")

if "safety_proxy_score" in combined.columns:
    axes[1, 1].hist(combined["safety_proxy_score"].dropna(), bins=10, color="#795548", alpha=0.7)
    axes[1, 1].set_title("Safety Proxy Score")

plt.suptitle("ADMET / Synthesis / Safety Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "07_admet_synthesis", "admet_overview.png"), dpi=150)
plt.close()

# ── CELL 10: Summary ──────────────────────────────────────
scored_count = len(combined)
gate_pass = scored_count >= (5 if TINY_DEBUG else 10)

update_registry(
    os.path.join(BASE, "00_project_registry", "run_registry.csv"),
    {
        "notebook": "06_admet_synthesis",
        "step": "complete",
        "status": "complete",
        "output_path": admet_path,
        "notes": f"{scored_count} scored, mean_qed={combined['qed'].mean():.3f}",
    }
)

print("\n" + "=" * 60)
print("NOTEBOOK 06 COMPLETE — ADMET, Synthesis, Safety")
print("=" * 60)
print(f"  Scored: {scored_count}")
print(f"  Mean QED: {combined['qed'].mean():.3f}")
print(f"  PAINS flagged: {combined['pains_flag'].sum()}")
print(f"  Next: Run Notebook 07 (Final Ranking & TPP)")
