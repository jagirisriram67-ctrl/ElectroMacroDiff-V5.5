"""
EMD V5.2 Hybrid — Campaign Dashboard & Logging
================================================
Campaign progress tracking, daily log generation,
and team progress dashboard for the 10-day workflow.
"""

import os
import json
import datetime
import pandas as pd


def generate_daily_log(base_dir, output_path=None):
    """Generate a daily progress log from all registry and checkpoint files.
    
    Scans the project directory for completion markers and generates
    a human-readable progress report.
    """
    if output_path is None:
        output_path = os.path.join(base_dir, "11_logs", "daily_log.md")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    now = datetime.datetime.now()
    
    lines = [
        f"# ElectroMacroDiff V5.2 Hybrid — Daily Log",
        f"",
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        "---",
        "",
    ]
    
    # Check each module's completion status
    modules = [
        ("M1: Data Collection", [
            ("02_curated_data/jak2_curated_ligands.csv", "Curated ligands"),
            ("01_raw_data/pdb/5AEP.pdb", "PDB structure"),
        ]),
        ("M2: Feature Engineering", [
            ("03_features/ligand_features.csv", "Molecular features"),
            ("03_features/conformers/curated_conformers.sdf", "3D conformers"),
            ("03_features/ligand_graphs/se3_graphs.pt", "SE(3) graph tensors"),
        ]),
        ("M3: SE(3) Training", [
            ("04_models_checkpoints/se3_flow/se3_best_checkpoint.pt", "Best model"),
            ("04_models_checkpoints/se3_flow/training_log.csv", "Training log"),
            ("04_models_checkpoints/se3_flow/training_curve.png", "Training curve"),
        ]),
        ("M4: Candidate Generation", [
            ("05_generated_candidates/selfies/generated_selfies_filtered.csv", "SELFIES candidates"),
            ("05_generated_candidates/rdkit/generated_rdkit_filtered.csv", "RDKit candidates"),
            ("05_generated_candidates/se3/generated_se3_filtered.csv", "SE(3) candidates"),
            ("05_generated_candidates/merged/generated_merged_filtered.csv", "Merged candidates"),
        ]),
        ("M5: Docking", [
            ("06_docking/scores/docking_scores.csv", "Docking scores"),
        ]),
        ("M6: ADMET & Synthesis", [
            ("07_admet_synthesis/admet_scores.csv", "ADMET scores"),
            ("07_admet_synthesis/synthesis_scores.csv", "Synthesis scores"),
            ("07_admet_synthesis/safety_proxy_notes.csv", "Safety scores"),
        ]),
        ("M7: Ranking & Report", [
            ("08_final_ranking/final_ranked_candidates.csv", "Final ranking"),
            ("09_reports/EMD_V5_2_Hybrid_TPP.txt", "TPP report (TXT)"),
            ("09_reports/EMD_V5_2_Hybrid_TPP.docx", "TPP report (DOCX)"),
        ]),
    ]
    
    total_files = 0
    completed_files = 0
    
    for module_name, files in modules:
        module_done = 0
        module_total = len(files)
        
        file_status = []
        for rel_path, desc in files:
            full_path = os.path.join(base_dir, rel_path)
            exists = os.path.exists(full_path)
            if exists:
                size = os.path.getsize(full_path)
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(full_path))
                file_status.append(f"  - [x] {desc} ({size:,} bytes, {mtime.strftime('%H:%M')})")
                module_done += 1
                completed_files += 1
            else:
                file_status.append(f"  - [ ] {desc}")
            total_files += 1
        
        status = "COMPLETE" if module_done == module_total else f"{module_done}/{module_total}"
        lines.append(f"## {module_name} — {status}")
        lines.append("")
        lines.extend(file_status)
        lines.append("")
    
    # Summary
    pct = (completed_files / total_files * 100) if total_files > 0 else 0
    lines.extend([
        "---",
        "",
        f"## Overall Progress: {completed_files}/{total_files} files ({pct:.0f}%)",
        "",
    ])
    
    # Training metrics if available
    train_log = os.path.join(base_dir, "04_models_checkpoints", "se3_flow", "training_log.csv")
    if os.path.exists(train_log):
        try:
            log_df = pd.read_csv(train_log)
            last = log_df.iloc[-1]
            best_val = log_df["val_loss"].dropna().min()
            lines.extend([
                "### Training Summary",
                f"- Epochs completed: {int(last['epoch']) + 1}",
                f"- Final train loss: {last['train_loss']:.6f}",
                f"- Best val loss: {best_val:.6f}" if not pd.isna(best_val) else "- Best val loss: N/A",
                f"- Final LR: {last['lr']:.2e}",
                "",
            ])
        except:
            pass
    
    # Generation metrics if available
    gen_metrics = os.path.join(base_dir, "05_generated_candidates", "merged",
                               "generation_comparison_metrics.csv")
    if os.path.exists(gen_metrics):
        try:
            metrics_df = pd.read_csv(gen_metrics)
            m = metrics_df.iloc[0]
            lines.extend([
                "### Generation Summary",
                f"- Total candidates: {int(m.get('total_candidates', 0))}",
                f"- Validity: {m.get('validity_pct', 0):.1f}%",
                f"- Novelty: {m.get('novelty_pct', 0):.1f}%",
                f"- Internal diversity: {m.get('internal_diversity', 'N/A')}",
                "",
            ])
        except:
            pass
    
    # Ranking summary if available
    ranked_path = os.path.join(base_dir, "08_final_ranking", "final_ranked_candidates.csv")
    if os.path.exists(ranked_path):
        try:
            ranked_df = pd.read_csv(ranked_path)
            lines.extend([
                "### Ranking Summary",
                f"- Candidates ranked: {len(ranked_df)}",
                f"- Top score: {ranked_df['final_weighted_score'].max():.4f}",
            ])
            for _, row in ranked_df.head(3).iterrows():
                lines.append(
                    f"  - #{int(row['rank'])}: {row['candidate_id']} "
                    f"(score={row['final_weighted_score']:.4f})"
                )
            lines.append("")
        except:
            pass
    
    report = "\n".join(lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Daily log saved to {output_path}")
    return report


def check_campaign_status(base_dir):
    """Quick status check for pipeline progress.
    
    Returns a dict with module completion status.
    """
    status = {}
    
    checks = {
        "M1_data": "02_curated_data/jak2_curated_ligands.csv",
        "M2_features": "03_features/ligand_graphs/se3_graphs.pt",
        "M3_model": "04_models_checkpoints/se3_flow/se3_best_checkpoint.pt",
        "M4_generation": "05_generated_candidates/merged/generated_merged_filtered.csv",
        "M5_docking": "06_docking/scores/docking_scores.csv",
        "M6_admet": "07_admet_synthesis/admet_scores.csv",
        "M7_ranking": "08_final_ranking/final_ranked_candidates.csv",
        "M8_report": "09_reports/EMD_V5_2_Hybrid_TPP.txt",
    }
    
    for key, path in checks.items():
        full_path = os.path.join(base_dir, path)
        status[key] = os.path.exists(full_path)
    
    completed = sum(status.values())
    total = len(status)
    status["overall_pct"] = completed / total * 100
    
    return status
