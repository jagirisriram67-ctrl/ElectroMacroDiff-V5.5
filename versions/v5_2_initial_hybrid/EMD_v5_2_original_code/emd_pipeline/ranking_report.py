"""
EMD V5.2 Hybrid — Final Ranking & Report Module (M7/M8)
=======================================================
Consensus ranking, molecule card generation, and TPP report writing.
"""

import os
import numpy as np
import pandas as pd


def consensus_ranking(docking_df, admet_df, synthesis_df, safety_df,
                       candidates_df=None, weights=None, output_path=None):
    """Compute final consensus ranking across all scoring dimensions.
    
    Default weights:
        docking: 35%, pose_sanity: 15%, admet: 20%,
        synthesis: 15%, novelty/diversity: 10%, safety: 5%
    """
    if weights is None:
        weights = {
            "docking_score": 0.35,
            "pose_sanity": 0.15,
            "admet": 0.20,
            "synthesis": 0.15,
            "novelty_diversity": 0.10,
            "safety_proxy": 0.05,
        }
    
    # Start with docking scores
    merged = docking_df[["candidate_id", "smiles", "best_score"]].copy()
    merged = merged.dropna(subset=["best_score"])
    
    # Merge ADMET
    if admet_df is not None and len(admet_df) > 0:
        merged = merged.merge(
            admet_df[["candidate_id", "qed", "admet_score", "lipinski_violations"]],
            on="candidate_id", how="left"
        )
    
    # Merge synthesis
    if synthesis_df is not None and len(synthesis_df) > 0:
        merged = merged.merge(
            synthesis_df[["candidate_id", "sa_score", "synthesis_score"]],
            on="candidate_id", how="left"
        )
    
    # Merge safety
    if safety_df is not None and len(safety_df) > 0:
        merged = merged.merge(
            safety_df[["candidate_id", "safety_proxy_score"]],
            on="candidate_id", how="left"
        )
    
    # Merge candidate metadata
    if candidates_df is not None and len(candidates_df) > 0:
        meta_cols = ["candidate_id", "source_generator", "inchikey", "novel_flag",
                     "has_macrocycle_12_20", "max_ring_size"]
        available = [c for c in meta_cols if c in candidates_df.columns]
        merged = merged.merge(candidates_df[available], on="candidate_id", how="left")
    
    # Fill missing scores
    merged["admet_score"] = merged.get("admet_score", pd.Series(0.5, index=merged.index)).fillna(0.5)
    merged["synthesis_score"] = merged.get("synthesis_score", pd.Series(0.5, index=merged.index)).fillna(0.5)
    merged["safety_proxy_score"] = merged.get("safety_proxy_score", pd.Series(0.5, index=merged.index)).fillna(0.5)
    
    # Normalize docking score (more negative = better -> higher normalized)
    if len(merged) > 0 and "best_score" in merged.columns:
        dmin = merged["best_score"].min()
        dmax = merged["best_score"].max()
        if dmax != dmin:
            merged["docking_score_norm"] = (merged["best_score"] - dmax) / (dmin - dmax)
        else:
            merged["docking_score_norm"] = 0.5
    else:
        merged["docking_score_norm"] = 0.5
    
    # Pose sanity (proxy: better docking = better pose)
    merged["pose_sanity_norm"] = merged["docking_score_norm"]
    
    # Novelty/diversity (use novel_flag if available)
    merged["novelty_diversity_norm"] = merged.get("novel_flag", pd.Series(True, index=merged.index)).astype(float).fillna(0.5)
    
    # Blueprint §11.6 column aliases
    merged["docking_score"] = merged["best_score"]
    merged["pose_score"] = merged["pose_sanity_norm"]
    merged["novelty_score"] = merged["novelty_diversity_norm"]
    
    # Compute final weighted score
    merged["final_weighted_score"] = (
        weights["docking_score"] * merged["docking_score_norm"]
        + weights["pose_sanity"] * merged["pose_sanity_norm"]
        + weights["admet"] * merged["admet_score"]
        + weights["synthesis"] * merged["synthesis_score"]
        + weights["novelty_diversity"] * merged["novelty_diversity_norm"]
        + weights["safety_proxy"] * merged["safety_proxy_score"]
    )
    
    # Rank
    merged = merged.sort_values("final_weighted_score", ascending=False).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)
    
    # Diversity clustering (Butina, Tanimoto, radius=2 Morgan FPs)
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        from rdkit.ML.Cluster import Butina
        
        fps = []
        valid_indices = []
        for i, smi in enumerate(merged["smiles"]):
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                valid_indices.append(i)
        
        if len(fps) >= 2:
            # Compute distance matrix
            n = len(fps)
            dists = []
            for i in range(1, n):
                for j in range(i):
                    dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            
            clusters = Butina.ClusterData(dists, n, distThresh=0.4, isDistData=True)
            
            cluster_assignments = [0] * len(merged)
            for cluster_id, cluster_members in enumerate(clusters):
                for member_idx in cluster_members:
                    orig_idx = valid_indices[member_idx]
                    cluster_assignments[orig_idx] = cluster_id
            
            merged["diversity_cluster"] = cluster_assignments
        else:
            merged["diversity_cluster"] = 0
    except Exception:
        merged["diversity_cluster"] = 0
    
    # Decision logic
    decisions = []
    for i, row in merged.iterrows():
        if row["rank"] <= 5:
            decisions.append("CANDIDATE")
        elif row["rank"] <= 10:
            decisions.append("BACKUP")
        else:
            decisions.append("ARCHIVE")
    merged["decision"] = decisions
    
    # Main risk assessment
    risks = []
    for _, row in merged.iterrows():
        risk_items = []
        if row.get("admet_score", 1) < 0.4:
            risk_items.append("poor ADMET")
        if row.get("synthesis_score", 1) < 0.3:
            risk_items.append("hard synthesis")
        if row.get("safety_proxy_score", 1) < 0.5:
            risk_items.append("safety alerts")
        risks.append("; ".join(risk_items) if risk_items else "low risk")
    merged["main_risk"] = risks
    
    print(f"Final ranking: {len(merged)} candidates ranked")
    print(f"  Top 5 scores: {merged['final_weighted_score'].head().tolist()}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return merged


def generate_molecule_cards(ranked_df, output_path, top_n=10):
    """Generate molecule cards for top candidates as Excel file."""
    from rdkit import Chem
    
    top = ranked_df.head(top_n).copy()
    
    # Add 2D depiction info
    card_data = []
    for _, row in top.iterrows():
        card = {
            "Rank": row.get("rank", ""),
            "Candidate ID": row.get("candidate_id", ""),
            "SMILES": row.get("smiles", ""),
            "Source": row.get("source_generator", "unknown"),
            "Docking Score": f"{row.get('best_score', 'N/A')}",
            "QED": f"{row.get('qed', 'N/A'):.3f}" if pd.notna(row.get('qed')) else "N/A",
            "SA Score": f"{row.get('sa_score', 'N/A'):.2f}" if pd.notna(row.get('sa_score')) else "N/A",
            "ADMET Score": f"{row.get('admet_score', 'N/A'):.3f}" if pd.notna(row.get('admet_score')) else "N/A",
            "Safety": f"{row.get('safety_proxy_score', 'N/A'):.3f}" if pd.notna(row.get('safety_proxy_score')) else "N/A",
            "Final Score": f"{row.get('final_weighted_score', 'N/A'):.4f}" if pd.notna(row.get('final_weighted_score')) else "N/A",
            "Decision": row.get("decision", ""),
            "Main Risk": row.get("main_risk", ""),
            "InChIKey": row.get("inchikey", ""),
        }
        card_data.append(card)
    
    cards_df = pd.DataFrame(card_data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        cards_df.to_excel(output_path, index=False, sheet_name="Molecule Cards")
        print(f"Molecule cards saved to {output_path}")
    except:
        csv_path = output_path.replace(".xlsx", ".csv")
        cards_df.to_csv(csv_path, index=False)
        print(f"Molecule cards saved to {csv_path} (Excel failed)")
    
    return cards_df


def save_top_sdf(ranked_df, output_path, top_n=5, smiles_col="smiles"):
    """Save top N molecules as SDF file."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = Chem.SDWriter(output_path)
    saved = 0
    
    for _, row in ranked_df.head(top_n).iterrows():
        smi = str(row.get(smiles_col, ""))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if mol.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            
            mol.SetProp("candidate_id", str(row.get("candidate_id", "")))
            mol.SetProp("rank", str(row.get("rank", "")))
            mol.SetProp("docking_score", str(row.get("best_score", "")))
            mol.SetProp("final_score", str(row.get("final_weighted_score", "")))
            
            writer.write(mol)
            saved += 1
    
    writer.close()
    print(f"Saved {saved} molecules to {output_path}")


def generate_tpp_report(ranked_df, config, output_dir):
    """Generate TPP (Target Product Profile) style report."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = [
        "=" * 70,
        "ElectroMacroDiff V5.2 Hybrid — Target Product Profile (TPP) Report",
        "=" * 70,
        "",
        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"Target: JAK2 (Janus kinase 2)",
        f"Primary PDB: 5AEP",
        "",
        "-" * 70,
        "EXECUTIVE SUMMARY",
        "-" * 70,
        "",
        "This report presents computationally prioritized candidates for",
        "JAK2 inhibition, generated using a hybrid AI pipeline combining",
        "a custom SE(3) flow-matching model with classical RDKit/SELFIES",
        "baselines, followed by docking, ADMET filtering, and consensus ranking.",
        "",
        "IMPORTANT: These are computational hypotheses only.",
        "Experimental validation is required before any biological claims.",
        "",
    ]
    
    # Pipeline statistics section
    report_lines.extend([
        "-" * 70,
        "PIPELINE STATISTICS",
        "-" * 70,
        "",
        f"  Total candidates ranked: {len(ranked_df)}",
        f"  Final candidates (top 5): {len(ranked_df[ranked_df['decision'] == 'CANDIDATE'])}",
        f"  Backup candidates: {len(ranked_df[ranked_df['decision'] == 'BACKUP'])}",
    ])
    
    if "source_generator" in ranked_df.columns:
        report_lines.append("")
        report_lines.append("  Source Generator Comparison:")
        for src in ranked_df["source_generator"].dropna().unique():
            sub = ranked_df[ranked_df["source_generator"] == src]
            top5_count = len(sub[sub["rank"] <= 5])
            report_lines.append(
                f"    {src}: {len(sub)} candidates, "
                f"{top5_count} in top 5, "
                f"mean score={sub['final_weighted_score'].mean():.4f}"
            )
    report_lines.append("")
    
    # Top candidates section with InChIKey
    report_lines.extend([
        "-" * 70,
        "TOP CANDIDATE MOLECULES",
        "-" * 70,
        "",
    ])
    
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
        can_gen_inchi = True
    except:
        can_gen_inchi = False
    
    for _, row in ranked_df.head(5).iterrows():
        smi = row.get("smiles", "N/A")
        
        # Generate InChIKey if possible
        inchikey = row.get("inchikey", "")
        if (not inchikey or pd.isna(inchikey)) and can_gen_inchi:
            try:
                mol = Chem.MolFromSmiles(str(smi))
                if mol:
                    inchikey = InchiToInchiKey(MolToInchi(mol))
            except:
                inchikey = "N/A"
        
        report_lines.extend([
            f"Rank {row.get('rank', 'N/A')}:",
            f"  Candidate ID: {row.get('candidate_id', 'N/A')}",
            f"  SMILES: {smi}",
            f"  InChIKey: {inchikey}",
            f"  Source: {row.get('source_generator', 'N/A')}",
            f"  Docking Score: {row.get('best_score', 'N/A')}",
            f"  Final Score: {row.get('final_weighted_score', 'N/A'):.4f}" if pd.notna(row.get('final_weighted_score')) else "  Final Score: N/A",
            f"  Decision: {row.get('decision', 'N/A')}",
            f"  Risk: {row.get('main_risk', 'N/A')}",
            "",
        ])
    
    report_lines.extend([
        "-" * 70,
        "LIMITATIONS AND DISCLAIMERS",
        "-" * 70,
        "",
        "1. All scores are computational approximations.",
        "2. Docking scores do not guarantee binding affinity.",
        "3. ADMET predictions are proxy estimates, not experimental measurements.",
        "4. Safety assessments are structural alerts only, not toxicology.",
        "5. Synthesizability scores are heuristic, not verified routes.",
        "6. These candidates require experimental follow-up.",
        "",
        "-" * 70,
        "METHODS",
        "-" * 70,
        "",
        "1. Data: ChEMBL 36 JAK2 bioactivity records, curated and deduplicated.",
        "2. SE(3) Model: Custom flow-matching generator trained on 3D ligand graphs.",
        "3. Baselines: SELFIES mutation and RDKit substituent replacement.",
        "4. Docking: AutoDock Vina against JAK2 structure 5AEP.",
        "5. ADMET: Lipinski, Veber, QED, PAINS, Brenk filters.",
        "6. Synthesis: SA Score, with AiZynthFinder for top candidates if available.",
        "7. Ranking: Weighted consensus across docking, ADMET, synthesis, novelty, safety.",
        "",
        "=" * 70,
        "END OF TPP REPORT",
        "=" * 70,
    ])
    
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "EMD_V5_2_Hybrid_TPP.txt")
    
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"TPP report saved to {report_path}")
    
    # Try to create DOCX version
    try:
        from docx import Document
        doc = Document()
        doc.add_heading("ElectroMacroDiff V5.2 Hybrid — TPP Report", 0)
        
        for line in report_lines:
            if line.startswith("=") or line.startswith("-"):
                continue
            elif line.strip() == "":
                doc.add_paragraph("")
            elif line.startswith("Rank "):
                doc.add_heading(line, level=2)
            elif line.startswith("  "):
                doc.add_paragraph(line.strip(), style="List Bullet")
            else:
                doc.add_paragraph(line)
        
        docx_path = os.path.join(output_dir, "EMD_V5_2_Hybrid_TPP.docx")
        doc.save(docx_path)
        print(f"TPP DOCX saved to {docx_path}")
    except ImportError:
        print("python-docx not available, DOCX not generated")
    
    return report_path
