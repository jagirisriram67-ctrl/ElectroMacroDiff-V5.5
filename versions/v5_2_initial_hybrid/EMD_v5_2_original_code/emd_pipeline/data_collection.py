"""
EMD V5.2 Hybrid — Data Collection & Curation Module (M1)
========================================================
Handles ChEMBL/BindingDB/PDB data retrieval, curation, deduplication,
and macrocycle/constrained-ring labeling for JAK2 ligands.
"""

import os
import time
import pandas as pd
import numpy as np
import requests
from pathlib import Path


def fetch_chembl_jak2_activities(output_path, max_records=10000, activity_types=None):
    """Fetch JAK2 bioactivity data from ChEMBL API."""
    if activity_types is None:
        activity_types = ["IC50", "Ki", "Kd"]
    
    from chembl_webresource_client.new_client import new_client
    
    target_api = new_client.target
    jak2_targets = target_api.search("JAK2").filter(
        organism="Homo sapiens", target_type="SINGLE PROTEIN"
    )
    jak2_targets_list = list(jak2_targets)
    print(f"Found {len(jak2_targets_list)} JAK2 target record(s)")
    
    if not jak2_targets_list:
        jak2_targets = target_api.search("Janus kinase 2")
        jak2_targets_list = list(jak2_targets)
    
    target_ids = [t["target_chembl_id"] for t in jak2_targets_list]
    print(f"Target IDs: {target_ids}")
    
    activity_api = new_client.activity
    all_activities = []
    
    for tid in target_ids:
        for atype in activity_types:
            print(f"  Fetching {atype} for {tid}...")
            try:
                acts = activity_api.filter(
                    target_chembl_id=tid,
                    standard_type=atype,
                    standard_value__isnull=False,
                ).only([
                    "molecule_chembl_id", "canonical_smiles",
                    "standard_type", "standard_relation",
                    "standard_value", "standard_units",
                    "assay_chembl_id", "assay_description",
                    "target_chembl_id", "target_pref_name",
                    "pchembl_value", "document_chembl_id",
                ])
                batch = list(acts)
                all_activities.extend(batch)
                print(f"    Got {len(batch)} records")
                if len(all_activities) >= max_records:
                    break
            except Exception as e:
                print(f"    Error: {e}")
                continue
        if len(all_activities) >= max_records:
            break
    
    if not all_activities:
        print("No activities retrieved.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_activities[:max_records])
    rename_map = {
        "molecule_chembl_id": "chembl_id",
        "standard_type": "activity_type",
        "standard_value": "activity_value",
        "standard_units": "activity_units",
        "assay_chembl_id": "assay_id",
        "target_chembl_id": "target_id",
        "target_pref_name": "target_name",
        "pchembl_value": "p_activity",
        "document_chembl_id": "reference",
    }
    df = df.rename(columns=rename_map)
    df["source"] = "ChEMBL"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} raw ChEMBL records to {output_path}")
    return df


def fetch_pdb_structure(pdb_id, output_dir, fmt="pdb"):
    """Download a PDB structure file from RCSB."""
    pdb_id = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.{fmt}"
    filename = f"{pdb_id}.{fmt}"
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    print(f"Downloading {pdb_id} from RCSB PDB...")
    response = requests.get(url, timeout=60)
    
    if response.status_code == 200:
        with open(output_path, "w") as f:
            f.write(response.text)
        print(f"Saved {pdb_id} to {output_path}")
        return output_path
    else:
        raise ConnectionError(f"Failed to download {pdb_id}: HTTP {response.status_code}")


def curate_activities(raw_df, mw_min=250, mw_max=900, output_path=None):
    """Curate raw activity data: validate SMILES, convert units, deduplicate, label rings."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    
    df = raw_df.copy()
    print(f"Starting curation with {len(df)} records")
    
    df = df.dropna(subset=["canonical_smiles"])
    
    # Validate SMILES and compute InChIKey
    records = []
    for _, row in df.iterrows():
        smi = str(row.get("canonical_smiles", ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        can_smi = Chem.MolToSmiles(mol)
        try:
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
            inchi_str = MolToInchi(mol)
            ik = InchiToInchiKey(inchi_str) if inchi_str else ""
        except:
            ik = ""
        
        mw = Descriptors.MolWt(mol)
        if mw < mw_min or mw > mw_max:
            continue
        
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        max_rs = max(ring_sizes) if ring_sizes else 0
        
        rec = dict(row)
        rec["canonical_smiles"] = can_smi
        rec["inchikey"] = ik
        rec["mw"] = mw
        rec["max_ring_size"] = max_rs
        rec["has_macrocycle_12_20"] = 12 <= max_rs <= 20
        rec["has_constrained_ring_8_11"] = 8 <= max_rs <= 11
        rec["num_rings"] = len(ring_sizes)
        rec["num_rotatable_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        records.append(rec)
    
    df = pd.DataFrame(records)
    print(f"After validation: {len(df)}")
    
    # Convert to nM
    df["activity_value"] = pd.to_numeric(df.get("activity_value", 0), errors="coerce")
    df["activity_value_nM"] = df["activity_value"].copy()
    if "activity_units" in df.columns:
        mask_um = df["activity_units"].str.lower().str.contains("um|µm", na=False)
        df.loc[mask_um, "activity_value_nM"] = df.loc[mask_um, "activity_value"] * 1000
    
    df["p_activity_computed"] = df["activity_value_nM"].apply(
        lambda x: -np.log10(x * 1e-9) if pd.notna(x) and x > 0 else np.nan
    )
    if "p_activity" in df.columns:
        df["p_activity"] = pd.to_numeric(df["p_activity"], errors="coerce")
        df["p_activity"] = df["p_activity"].fillna(df["p_activity_computed"])
    else:
        df["p_activity"] = df["p_activity_computed"]
    
    # Deduplicate
    df = df.sort_values("p_activity", ascending=False)
    df = df.drop_duplicates(subset=["inchikey"], keep="first")
    df = df.reset_index(drop=True)
    df["mol_id"] = [f"JAK2_{i:05d}" for i in range(len(df))]
    
    # Add Murcko scaffold (blueprint requirement)
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = []
    for smi in df["canonical_smiles"]:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                scaf = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol)
                )
                scaffolds.append(Chem.MolToSmiles(scaf))
            else:
                scaffolds.append("")
        except:
            scaffolds.append("")
    df["murcko_scaffold"] = scaffolds
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} curated ligands to {output_path}")
    return df


def scaffold_split(df, smiles_col="canonical_smiles", train_frac=0.7, val_frac=0.15, seed=42):
    """Split molecules by Murcko scaffold."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    
    np.random.seed(seed)
    scaffold_to_mols = defaultdict(list)
    
    for idx, smi in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(
                    MurckoScaffold.GetScaffoldForMol(mol)
                )
                scaffold_to_mols[Chem.MolToSmiles(scaffold)].append(idx)
            else:
                scaffold_to_mols["_invalid_"].append(idx)
        except:
            scaffold_to_mols["_error_"].append(idx)
    
    scaffolds = list(scaffold_to_mols.keys())
    np.random.shuffle(scaffolds)
    
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    splits = [""] * n
    count = 0
    
    for scaffold in scaffolds:
        for idx in scaffold_to_mols[scaffold]:
            if count < n_train:
                splits[idx] = "train"
            elif count < n_train + n_val:
                splits[idx] = "val"
            else:
                splits[idx] = "test"
            count += 1
    
    df = df.copy()
    df["split"] = splits
    print(f"Scaffold split: {dict(df['split'].value_counts())}")
    return df


def analyze_macrocycles(df):
    """Generate macrocycle/constrained-ring statistics."""
    stats = {
        "total_ligands": len(df),
        "macrocycles_12_20": int(df.get("has_macrocycle_12_20", pd.Series(dtype=bool)).sum()),
        "constrained_8_11": int(df.get("has_constrained_ring_8_11", pd.Series(dtype=bool)).sum()),
        "max_ring_size_mean": float(df.get("max_ring_size", pd.Series(dtype=float)).mean()),
    }
    print(f"Macrocycle Analysis: {stats}")
    return stats
