"""
EMD V5.2 Hybrid — ADMET, Safety & Synthesis Module (M6)
=======================================================
Drug-likeness filters, ADMET property computation,
safety proxy scoring, and synthesizability assessment.
"""

import os
import numpy as np
import pandas as pd


def compute_admet_scores(candidates_df, smiles_col="canonical_smiles", output_path=None):
    """Compute ADMET-related descriptors and drug-likeness flags.
    
    Computes: Lipinski, Veber, QED, PAINS, Brenk, and combined ADMET score.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED as QEDModule
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    
    # Set up PAINS filter
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(params)
    
    # Try Brenk filter
    try:
        brenk_params = FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        brenk_catalog = FilterCatalog(brenk_params)
        has_brenk = True
    except:
        has_brenk = False
    
    records = []
    
    for _, row in candidates_df.iterrows():
        smi = str(row.get(smiles_col, ""))
        cid = row.get("candidate_id", "")
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
        qed = QEDModule.qed(mol)
        
        # Lipinski violations
        lipinski_violations = sum([
            mw > 500, logp > 5, hbd > 5, hba > 10
        ])
        
        # Veber violations
        veber_violations = sum([
            tpsa > 140, rotb > 10
        ])
        
        # PAINS
        pains_flag = pains_catalog.HasMatch(mol)
        
        # Brenk
        brenk_flag = brenk_catalog.HasMatch(mol) if has_brenk else False
        
        # Combined ADMET score (0-1, higher = better)
        admet_score = 1.0
        if lipinski_violations > 1:
            admet_score -= 0.3
        if veber_violations > 0:
            admet_score -= 0.2
        if pains_flag:
            admet_score -= 0.2
        if brenk_flag:
            admet_score -= 0.1
        admet_score = max(0.0, min(1.0, admet_score * qed))
        
        records.append({
            "candidate_id": cid,
            "smiles": smi,
            "mw": mw,
            "logp": logp,
            "hbd": hbd,
            "hba": hba,
            "tpsa": tpsa,
            "rotatable_bonds": rotb,
            "qed": qed,
            "lipinski_violations": lipinski_violations,
            "veber_violations": veber_violations,
            "pains_flag": pains_flag,
            "brenk_flag": brenk_flag,
            "admet_score": admet_score,
        })
    
    df = pd.DataFrame(records)
    print(f"ADMET scores computed for {len(df)} candidates")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return df


def compute_synthesis_scores(candidates_df, smiles_col="canonical_smiles", output_path=None):
    """Compute synthesizability scores for candidates.
    
    Uses SA Score as primary metric, with AiZynthFinder as optional stretch.
    """
    from rdkit import Chem
    
    records = []
    
    for _, row in candidates_df.iterrows():
        smi = str(row.get(smiles_col, ""))
        cid = row.get("candidate_id", "")
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # SA Score
        try:
            from rdkit.Chem import RDConfig
            import sys
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            from sascorer import calculateScore
            sa_score = calculateScore(mol)
        except:
            # Fallback: estimate from complexity
            sa_score = min(10, max(1, mol.GetNumHeavyAtoms() / 10.0 + 2))
        
        # Normalize to 0-1 (lower SA = better synthesis = higher score)
        synthesis_score = max(0, min(1, (10 - sa_score) / 9.0))
        
        records.append({
            "candidate_id": cid,
            "smiles": smi,
            "sa_score": sa_score,
            "synthesis_score": synthesis_score,
            "route_found": "not_attempted",
            "synthesis_notes": "SA score only" if sa_score <= 4 else "moderate complexity",
        })
    
    df = pd.DataFrame(records)
    print(f"Synthesis scores computed for {len(df)} candidates")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return df


def compute_safety_proxy(candidates_df, smiles_col="canonical_smiles", output_path=None):
    """Compute safety proxy scores.
    
    Notes:
    - This is NOT a real safety assessment
    - Uses structural alerts and similarity as proxies
    - Must be clearly stated as computational hypothesis
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    records = []
    
    # Known problematic substructures (comprehensive set)
    alert_smarts = [
        ("[N+](=O)[O-]",   "nitro"),
        ("[SH]",            "thiol"),
        ("C(=O)Cl",         "acyl_chloride"),
        ("[N;R]=[N;R]",     "ring_azo"),
        ("C#N",             "nitrile"),
        ("[N]=[N]=[N]",     "azide"),
        ("C=CC(=O)",        "michael_acceptor"),
        ("C1OC1",           "epoxide"),
        ("NN",              "hydrazine"),
        ("[CH]=O",          "aldehyde"),
        ("OO",              "peroxide"),
        ("[n+][O-]",        "n_oxide"),
    ]
    
    alert_patterns = []
    for sm, name in alert_smarts:
        pat = Chem.MolFromSmarts(sm)
        if pat:
            alert_patterns.append((name, pat))
    
    for _, row in candidates_df.iterrows():
        smi = str(row.get(smiles_col, ""))
        cid = row.get("candidate_id", "")
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # Count structural alerts
        alerts = []
        for name, pat in alert_patterns:
            if mol.HasSubstructMatch(pat):
                alerts.append(name)
        
        # Safety proxy score (0-1, higher = safer)
        safety_score = 1.0 - min(1.0, len(alerts) * 0.25)
        
        records.append({
            "candidate_id": cid,
            "smiles": smi,
            "structural_alerts": "; ".join(alerts) if alerts else "none",
            "num_alerts": len(alerts),
            "safety_proxy_score": safety_score,
            "slc19a3_note": "Not assessed - requires experimental validation",
            "safety_notes": "Computational proxy only - not a safety determination",
        })
    
    df = pd.DataFrame(records)
    print(f"Safety proxy scores computed for {len(df)} candidates")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return df


def apply_filter_flags(admet_df, config=None, output_path=None):
    """Apply go/no-go filter flags based on ADMET thresholds."""
    if config is None:
        config = {
            "qed_threshold": 0.3,
            "sa_score_threshold": 6.0,
            "mw_max": 900,
            "logp_max": 7.0,
            "lipinski_max_violations": 1,
        }
    
    df = admet_df.copy()
    
    df["pass_qed"] = df["qed"] >= config.get("qed_threshold", 0.3)
    df["pass_lipinski"] = df["lipinski_violations"] <= config.get("lipinski_max_violations", 1)
    df["pass_mw"] = df["mw"] <= config.get("mw_max", 900)
    df["pass_logp"] = df["logp"] <= config.get("logp_max", 7.0)
    df["pass_pains"] = ~df["pains_flag"]
    
    df["passes_all_filters"] = (
        df["pass_qed"] & df["pass_lipinski"] & df["pass_mw"] &
        df["pass_logp"] & df["pass_pains"]
    )
    
    passed = df["passes_all_filters"].sum()
    print(f"Filter results: {passed}/{len(df)} passed all filters")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
    
    return df
