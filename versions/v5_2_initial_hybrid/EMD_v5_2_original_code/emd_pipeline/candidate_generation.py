"""
EMD V5.2 Hybrid — Candidate Generation Module (M4)
===================================================
Baseline generation using RDKit and SELFIES, plus
post-generation filtering and merging with SE(3) outputs.
"""

import os
import numpy as np
import pandas as pd


def generate_selfies_candidates(seed_df, smiles_col="canonical_smiles",
                                 mutations_per_seed=10, seed=42, output_path=None):
    """Generate candidate molecules by SELFIES mutation.
    
    Process:
    1. Convert seed SMILES to SELFIES
    2. Apply random symbol mutations
    3. Convert back to SMILES
    4. Validate with RDKit
    """
    import selfies as sf
    from rdkit import Chem
    
    np.random.seed(seed)
    
    # Get SELFIES alphabet
    alphabet = list(sf.get_semantic_robust_alphabet())
    
    results = []
    total_attempted = 0
    
    for _, row in seed_df.iterrows():
        smi = str(row.get(smiles_col, ""))
        parent_id = row.get("mol_id", "unknown")
        
        try:
            selfies_str = sf.encoder(smi)
            if selfies_str is None:
                continue
            
            tokens = list(sf.split_selfies(selfies_str))
            
            for mut_idx in range(mutations_per_seed):
                total_attempted += 1
                mutated = tokens.copy()
                
                # Random mutation strategy
                strategy = np.random.choice(["replace", "insert", "delete"])
                
                if strategy == "replace" and len(mutated) > 0:
                    pos = np.random.randint(len(mutated))
                    mutated[pos] = np.random.choice(alphabet)
                
                elif strategy == "insert":
                    pos = np.random.randint(len(mutated) + 1)
                    mutated.insert(pos, np.random.choice(alphabet))
                
                elif strategy == "delete" and len(mutated) > 2:
                    pos = np.random.randint(len(mutated))
                    mutated.pop(pos)
                
                new_selfies = "".join(mutated)
                
                try:
                    new_smi = sf.decoder(new_selfies)
                    if new_smi:
                        mol = Chem.MolFromSmiles(new_smi)
                        if mol:
                            can_smi = Chem.MolToSmiles(mol)
                            results.append({
                                "candidate_id": f"SELFIES_{len(results):05d}",
                                "source_generator": "SELFIES",
                                "parent_mol_id": parent_id,
                                "canonical_smiles": can_smi,
                                "valid_rdkit": True,
                                "generation_notes": f"{strategy} mutation",
                            })
                except:
                    continue
        except:
            continue
    
    df = pd.DataFrame(results)
    print(f"SELFIES: {total_attempted} attempted, {len(df)} valid")
    
    if output_path and len(df) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return df


def generate_rdkit_candidates(seed_df, smiles_col="canonical_smiles",
                               subs_per_seed=5, seed=42, output_path=None):
    """Generate candidate molecules by RDKit molecular modification.
    
    Uses three complementary strategies to maximize hit rate:
    1. Substituent swap: replace matched functional groups
    2. Random atom mutation: change atom types at random positions
    3. Ring bioisostere: swap ring systems
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
    
    np.random.seed(seed)
    
    # Strategy 1: Substituent replacement (expanded pattern set)
    substituents = [
        ("[NH2]", "NC"),           # primary amine -> methylamine
        ("[OH]", "OC"),            # hydroxyl -> methoxy
        ("[F]", "[Cl]"),           # fluorine -> chlorine
        ("[Cl]", "[F]"),           # chlorine -> fluorine
        ("[NH2]", "N"),            # amine -> nitrogen
        ("[OH]", "O"),             # hydroxyl -> ether
        ("[Br]", "[Cl]"),          # bromine -> chlorine
        ("[CH3]", "C(F)(F)F"),     # methyl -> trifluoromethyl
        ("[c]1[cH][cH][cH][cH][c]1", "c1ccncc1"),  # phenyl -> pyridine
    ]
    
    # Strategy 2: Random atom mutation targets
    atom_swaps = [
        (6, 7),   # C -> N (carbon to nitrogen in aromatic rings)
        (7, 6),   # N -> C
        (8, 7),   # O -> N
        (7, 8),   # N -> O
        (16, 8),  # S -> O
        (8, 16),  # O -> S
    ]
    
    results = []
    total_attempted = 0
    seen_smiles = set()
    
    for _, row in seed_df.iterrows():
        smi = str(row.get(smiles_col, ""))
        parent_id = row.get("mol_id", "unknown")
        
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            can_parent = Chem.MolToSmiles(mol)
            seen_smiles.add(can_parent)
            
            # --- Strategy 1: Substituent swap ---
            for old_smarts, new_smiles in substituents:
                if len(results) >= len(seed_df) * subs_per_seed:
                    break
                total_attempted += 1
                try:
                    query = Chem.MolFromSmarts(old_smarts)
                    replacement = Chem.MolFromSmiles(new_smiles)
                    if query is None or replacement is None:
                        continue
                    if mol.HasSubstructMatch(query):
                        products = AllChem.ReplaceSubstructs(mol, query, replacement)
                        if products:
                            for nm in products[:1]:
                                try:
                                    Chem.SanitizeMol(nm)
                                    can_smi = Chem.MolToSmiles(nm)
                                    if can_smi not in seen_smiles and Chem.MolFromSmiles(can_smi) is not None:
                                        mw = Descriptors.MolWt(Chem.MolFromSmiles(can_smi))
                                        if 100 < mw < 1000:
                                            seen_smiles.add(can_smi)
                                            results.append({
                                                "candidate_id": f"RDKIT_{len(results):05d}",
                                                "source_generator": "RDKit",
                                                "parent_mol_id": parent_id,
                                                "canonical_smiles": can_smi,
                                                "valid_rdkit": True,
                                                "generation_notes": f"sub {old_smarts}->{new_smiles}",
                                            })
                                except:
                                    pass
                except:
                    continue
            
            # --- Strategy 2: Random atom mutation ---
            rw_mol = Chem.RWMol(mol)
            for _ in range(min(3, subs_per_seed)):
                total_attempted += 1
                try:
                    mut_mol = Chem.RWMol(Chem.MolFromSmiles(smi))
                    if mut_mol is None or mut_mol.GetNumAtoms() < 2:
                        continue
                    
                    # Pick a random atom to mutate
                    atom_idx = np.random.randint(mut_mol.GetNumAtoms())
                    old_num = mut_mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
                    
                    # Find valid swap
                    valid_swaps = [(o, n) for o, n in atom_swaps if o == old_num]
                    if not valid_swaps:
                        continue
                    
                    _, new_num = valid_swaps[np.random.randint(len(valid_swaps))]
                    mut_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(new_num)
                    
                    # Clear aromaticity flags and re-sanitize
                    try:
                        Chem.SanitizeMol(mut_mol)
                        can_smi = Chem.MolToSmiles(mut_mol)
                        if can_smi not in seen_smiles and Chem.MolFromSmiles(can_smi) is not None:
                            seen_smiles.add(can_smi)
                            results.append({
                                "candidate_id": f"RDKIT_{len(results):05d}",
                                "source_generator": "RDKit",
                                "parent_mol_id": parent_id,
                                "canonical_smiles": can_smi,
                                "valid_rdkit": True,
                                "generation_notes": f"atom_mutate idx={atom_idx} {old_num}->{new_num}",
                            })
                    except:
                        pass
                except:
                    continue
            
            # --- Strategy 3: Add/remove small groups ---
            total_attempted += 1
            try:
                # Add methyl group to a random nitrogen
                n_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetTotalNumHs() > 0]
                if n_indices:
                    target = n_indices[np.random.randint(len(n_indices))]
                    ed_mol = Chem.RWMol(mol)
                    new_c = ed_mol.AddAtom(Chem.Atom(6))
                    ed_mol.AddBond(target, new_c, Chem.BondType.SINGLE)
                    try:
                        Chem.SanitizeMol(ed_mol)
                        can_smi = Chem.MolToSmiles(ed_mol)
                        if can_smi not in seen_smiles and Chem.MolFromSmiles(can_smi) is not None:
                            seen_smiles.add(can_smi)
                            results.append({
                                "candidate_id": f"RDKIT_{len(results):05d}",
                                "source_generator": "RDKit",
                                "parent_mol_id": parent_id,
                                "canonical_smiles": can_smi,
                                "valid_rdkit": True,
                                "generation_notes": f"N-methylation at idx={target}",
                            })
                    except:
                        pass
            except:
                pass
                
        except:
            continue
    
    df = pd.DataFrame(results)
    print(f"RDKit: {total_attempted} attempted, {len(df)} valid")
    
    if output_path and len(df) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return df


def filter_candidates(df, smiles_col="canonical_smiles", training_smiles=None,
                       mw_range=(250, 900), logp_max=7.0, output_path=None):
    """Filter generated candidates for validity, uniqueness, novelty, and drug-likeness."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as QEDModule
    
    records = []
    seen_smiles = set()
    training_set = set(training_smiles) if training_smiles else set()
    
    for _, row in df.iterrows():
        smi = str(row.get(smiles_col, ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        can_smi = Chem.MolToSmiles(mol)
        
        # Uniqueness
        if can_smi in seen_smiles:
            continue
        seen_smiles.add(can_smi)
        
        # Novelty
        novel = can_smi not in training_set
        
        # Properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        qed = QEDModule.qed(mol)
        
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        max_rs = max(ring_sizes) if ring_sizes else 0
        
        # SA Score (approximate)
        try:
            from rdkit.Chem import RDConfig
            import sys
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            from sascorer import calculateScore
            sa_score = calculateScore(mol)
        except:
            sa_score = 5.0  # default mid-range
        
        # InChIKey
        try:
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
            ik = InchiToInchiKey(MolToInchi(mol))
        except:
            ik = ""
        
        # Filters
        if mw < mw_range[0] or mw > mw_range[1]:
            continue
        if logp > logp_max:
            continue
        
        records.append({
            "candidate_id": row.get("candidate_id", f"FILT_{len(records):05d}"),
            "source_generator": row.get("source_generator", "unknown"),
            "parent_mol_id": row.get("parent_mol_id", ""),
            "canonical_smiles": can_smi,
            "inchikey": ik,
            "valid_rdkit": True,
            "unique_flag": True,
            "novel_flag": novel,
            "mw": mw,
            "logp": logp,
            "tpsa": tpsa,
            "qed": qed,
            "sa_score": sa_score,
            "max_ring_size": max_rs,
            "has_macrocycle_12_20": 12 <= max_rs <= 20,
            "generation_notes": row.get("generation_notes", ""),
        })
    
    result = pd.DataFrame(records)
    print(f"Filtering: {len(df)} input -> {len(result)} passed")
    
    if output_path and len(result) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return result


def merge_candidates(*dfs, output_path=None):
    """Merge candidate DataFrames from different generators."""
    merged = pd.concat(dfs, ignore_index=True)
    
    # Re-deduplicate across generators
    merged = merged.drop_duplicates(subset=["canonical_smiles"], keep="first")
    merged = merged.reset_index(drop=True)
    merged["candidate_id"] = [f"MERGED_{i:05d}" for i in range(len(merged))]
    
    print(f"Merged candidates: {len(merged)} unique")
    
    # Source stats
    if "source_generator" in merged.columns:
        print(f"  By source: {dict(merged['source_generator'].value_counts())}")
    
    if output_path and len(merged) > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return merged


def compute_generation_metrics(merged_df, training_smiles=None, output_path=None):
    """Compute generation quality metrics including Tanimoto diversity.
    
    Metrics computed:
    - Validity, uniqueness, novelty (standard generative model metrics)
    - Mean QED and SA score
    - Macrocycle count
    - Internal Tanimoto diversity (pairwise Morgan FP similarity)
    - Per-source breakdown
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    total = len(merged_df)
    valid = merged_df.get("valid_rdkit", pd.Series(dtype=bool)).sum()
    unique = merged_df.get("unique_flag", pd.Series(dtype=bool)).sum()
    novel = merged_df.get("novel_flag", pd.Series(dtype=bool)).sum()
    
    metrics = {
        "total_candidates": total,
        "valid_count": int(valid),
        "validity_pct": float(valid / total * 100) if total > 0 else 0,
        "unique_count": int(unique),
        "uniqueness_pct": float(unique / total * 100) if total > 0 else 0,
        "novel_count": int(novel),
        "novelty_pct": float(novel / total * 100) if total > 0 else 0,
        "mean_qed": float(merged_df.get("qed", pd.Series(dtype=float)).mean()),
        "mean_sa_score": float(merged_df.get("sa_score", pd.Series(dtype=float)).mean()),
        "macrocycle_count": int(merged_df.get("has_macrocycle_12_20", pd.Series(dtype=bool)).sum()),
    }
    
    # Tanimoto internal diversity (1 - mean_pairwise_similarity)
    try:
        smiles_list = merged_df["canonical_smiles"].dropna().tolist()
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        
        if len(fps) >= 2:
            sims = []
            # Sample pairwise similarities (cap at 500 pairs for speed)
            n_pairs = min(len(fps) * (len(fps) - 1) // 2, 500)
            indices = np.random.choice(len(fps), size=(n_pairs, 2), replace=True)
            for i, j in indices:
                if i != j:
                    sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            
            if sims:
                mean_sim = float(np.mean(sims))
                metrics["mean_tanimoto_similarity"] = round(mean_sim, 4)
                metrics["internal_diversity"] = round(1.0 - mean_sim, 4)
    except Exception as e:
        metrics["internal_diversity"] = "N/A"
    
    # Per-source breakdown
    if "source_generator" in merged_df.columns:
        for src in merged_df["source_generator"].unique():
            sub = merged_df[merged_df["source_generator"] == src]
            metrics[f"{src}_count"] = len(sub)
            metrics[f"{src}_mean_qed"] = float(sub.get("qed", pd.Series(dtype=float)).mean())
            
            # Per-source diversity
            try:
                src_smiles = sub["canonical_smiles"].dropna().tolist()
                src_fps = []
                for smi in src_smiles:
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol:
                        src_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                if len(src_fps) >= 2:
                    src_sims = []
                    for ii in range(min(len(src_fps), 20)):
                        for jj in range(ii + 1, min(len(src_fps), 20)):
                            src_sims.append(DataStructs.TanimotoSimilarity(src_fps[ii], src_fps[jj]))
                    if src_sims:
                        metrics[f"{src}_diversity"] = round(1.0 - float(np.mean(src_sims)), 4)
            except:
                pass
    
    print(f"Generation Metrics: {metrics}")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame([metrics]).to_csv(output_path, index=False)
    
    return metrics
