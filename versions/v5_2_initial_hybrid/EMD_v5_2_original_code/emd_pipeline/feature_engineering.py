"""
EMD V5.2 Hybrid — Feature Engineering Module (M2)
=================================================
Computes molecular descriptors, generates 3D conformers,
and builds SE(3)-ready graph tensors for the flow matching model.
"""

import os
import numpy as np
import pandas as pd


def compute_descriptors(df, smiles_col="canonical_smiles", output_path=None):
    """Compute RDKit molecular descriptors for all molecules.
    
    Computes: MW, LogP, TPSA, HBD, HBA, RotBonds, QED, ring info, formal charge.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED as QEDModule
    
    records = []
    for _, row in df.iterrows():
        smi = str(row.get(smiles_col, ""))
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        mol_id = row.get("mol_id", "")
        ring_info = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ring_info.AtomRings()]
        
        rec = {
            "mol_id": mol_id,
            "canonical_smiles": smi,
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "qed": QEDModule.qed(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_rings": len(ring_sizes),
            "max_ring_size": max(ring_sizes) if ring_sizes else 0,
            "formal_charge": Chem.GetFormalCharge(mol),
            "passes_rdkit": True,
            "passes_basic_filters": True,
        }
        
        # Basic filter checks
        if rec["mw"] < 250 or rec["mw"] > 900:
            rec["passes_basic_filters"] = False
        if rec["logp"] > 7:
            rec["passes_basic_filters"] = False
        
        records.append(rec)
    
    result = pd.DataFrame(records)
    print(f"Computed descriptors for {len(result)} molecules")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"Saved features to {output_path}")
    
    return result


def generate_conformers(df, smiles_col="canonical_smiles", output_sdf=None,
                        max_conformers=1, optimize=True):
    """Generate 3D conformers for molecules using RDKit.
    
    Uses ETKDG for initial geometry, then MMFF/UFF for optimization.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    mols_3d = []
    failed = 0
    
    for _, row in df.iterrows():
        smi = str(row.get(smiles_col, ""))
        mol_id = row.get("mol_id", "unknown")
        
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                failed += 1
                continue
            
            mol = Chem.AddHs(mol)
            
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 1
            
            result = AllChem.EmbedMolecule(mol, params)
            if result == -1:
                # Fallback: random coords
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            if optimize and mol.GetNumConformers() > 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                except:
                    try:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
                    except:
                        pass
            
            if mol.GetNumConformers() > 0:
                mol.SetProp("mol_id", str(mol_id))
                mol.SetProp("SMILES", smi)
                mols_3d.append(mol)
            else:
                failed += 1
        except Exception:
            failed += 1
    
    print(f"Generated conformers: {len(mols_3d)} success, {failed} failed")
    
    if output_sdf and mols_3d:
        os.makedirs(os.path.dirname(output_sdf), exist_ok=True)
        writer = Chem.SDWriter(output_sdf)
        for mol in mols_3d:
            writer.write(mol)
        writer.close()
        print(f"Saved conformers to {output_sdf}")
    
    return mols_3d


def build_se3_graph_tensors(mols_3d, output_pt=None, output_index=None):
    """Convert 3D molecules to SE(3)-ready graph tensors.
    
    Each graph contains:
    - atom_features: [N, F] atom-level features
    - coords: [N, 3] 3D coordinates
    - edge_index: [2, E] bond connectivity
    - edge_features: [E, F_edge] bond features
    """
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Atom feature encoding dimensions
    ATOM_TYPES = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7}
    HYBRID_TYPES = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
    }
    
    graphs = []
    index_records = []
    
    for mol_idx, mol in enumerate(mols_3d):
        try:
            mol_noH = Chem.RemoveHs(mol)
            conf = mol.GetConformer(0)
            num_atoms = mol_noH.GetNumAtoms()
            
            if num_atoms < 3:
                continue
            
            # Compute Gasteiger charges
            try:
                AllChem.ComputeGasteigerCharges(mol_noH)
            except:
                pass
            
            # Atom features
            atom_feats = []
            coords = []
            ring_mask = []
            macrocycle_mask = []
            ring_info = mol_noH.GetRingInfo()
            
            # Precompute donor/acceptor atoms
            try:
                from rdkit.Chem import rdMolChemicalFeatures, ChemicalFeatures
                from rdkit import RDConfig
                fdef_path = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
                factory = ChemicalFeatures.BuildFeatureFactory(fdef_path)
                features = factory.GetFeaturesForMol(mol_noH)
                donor_atoms = set()
                acceptor_atoms = set()
                for feat in features:
                    if feat.GetFamily() == 'Donor':
                        for idx in feat.GetAtomIds():
                            donor_atoms.add(idx)
                    elif feat.GetFamily() == 'Acceptor':
                        for idx in feat.GetAtomIds():
                            acceptor_atoms.add(idx)
            except:
                donor_atoms = set()
                acceptor_atoms = set()
            
            for i, atom in enumerate(mol_noH.GetAtoms()):
                # One-hot atomic number
                atom_type = ATOM_TYPES.get(atom.GetAtomicNum(), len(ATOM_TYPES))
                one_hot_atom = [0] * (len(ATOM_TYPES) + 1)
                one_hot_atom[atom_type] = 1
                
                # Hybridization
                hyb = HYBRID_TYPES.get(atom.GetHybridization(), len(HYBRID_TYPES))
                one_hot_hyb = [0] * (len(HYBRID_TYPES) + 1)
                one_hot_hyb[hyb] = 1
                
                # Scalar features
                aromatic = 1.0 if atom.GetIsAromatic() else 0.0
                degree = atom.GetDegree() / 6.0
                formal_charge = float(atom.GetFormalCharge())
                in_ring = 1.0 if atom.IsInRing() else 0.0
                valence = atom.GetTotalValence() / 6.0
                
                # Donor / Acceptor flags
                is_donor = 1.0 if i in donor_atoms else 0.0
                is_acceptor = 1.0 if i in acceptor_atoms else 0.0
                
                # Chirality
                chiral_tag = atom.GetChiralTag()
                is_chiral = 1.0 if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0.0
                
                # Gasteiger charge
                try:
                    gasteiger = float(atom.GetDoubleProp("_GasteigerCharge"))
                    if np.isnan(gasteiger) or np.isinf(gasteiger):
                        gasteiger = 0.0
                except:
                    gasteiger = 0.0
                
                # Ring membership flags
                in_macrocycle = 0.0
                for ring in ring_info.AtomRings():
                    if i in ring and len(ring) >= 12:
                        in_macrocycle = 1.0
                        break
                
                ring_mask.append(in_ring)
                macrocycle_mask.append(in_macrocycle)
                
                feat = one_hot_atom + one_hot_hyb + [
                    aromatic, degree, formal_charge, in_ring,
                    valence, is_donor, is_acceptor, is_chiral,
                    gasteiger, in_macrocycle
                ]
                atom_feats.append(feat)
                
                # 3D coords (get from H-containing mol by index mapping)
                pos = conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            
            # Edge features (bonds)
            edge_index = [[], []]
            edge_feats = []
            
            BOND_TYPES = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3,
            }
            
            for bond in mol_noH.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                
                bt = BOND_TYPES.get(bond.GetBondType(), 4)
                one_hot_bond = [0] * 5
                one_hot_bond[bt] = 1
                
                conj = 1.0 if bond.GetIsConjugated() else 0.0
                in_ring = 1.0 if bond.IsInRing() else 0.0
                stereo = 1.0 if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE else 0.0
                
                feat = one_hot_bond + [conj, in_ring, stereo]
                
                # Bidirectional edges
                edge_index[0].extend([i, j])
                edge_index[1].extend([j, i])
                edge_feats.extend([feat, feat])
            
            # Get mol_id
            mol_id = mol.GetProp("mol_id") if mol.HasProp("mol_id") else f"mol_{mol_idx}"
            smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else ""
            
            graph = {
                "atom_features": torch.tensor(atom_feats, dtype=torch.float32),
                "coords": torch.tensor(coords, dtype=torch.float32),
                "edge_index": torch.tensor(edge_index, dtype=torch.long),
                "edge_features": torch.tensor(edge_feats, dtype=torch.float32) if edge_feats else torch.zeros(0, 8),
                "ring_mask": torch.tensor(ring_mask, dtype=torch.float32),
                "macrocycle_mask": torch.tensor(macrocycle_mask, dtype=torch.float32),
                "num_atoms": num_atoms,
                "mol_id": mol_id,
                "smiles": smiles,
            }
            graphs.append(graph)
            index_records.append({
                "graph_idx": len(graphs) - 1,
                "mol_id": mol_id,
                "smiles": smiles,
                "num_atoms": num_atoms,
            })
            
        except Exception as e:
            print(f"  Warning: graph build failed for mol {mol_idx}: {e}")
            continue
    
    print(f"Built {len(graphs)} SE(3) graph tensors")
    
    if output_pt and graphs:
        os.makedirs(os.path.dirname(output_pt), exist_ok=True)
        torch.save(graphs, output_pt)
        print(f"Saved graphs to {output_pt}")
    
    if output_index and index_records:
        idx_df = pd.DataFrame(index_records)
        os.makedirs(os.path.dirname(output_index), exist_ok=True)
        idx_df.to_csv(output_index, index=False)
        print(f"Saved graph index to {output_index}")
    
    return graphs


def save_splits(df, output_dir, id_col="mol_id", split_col="split"):
    """Save train/val/test split ID files."""
    os.makedirs(output_dir, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        ids = df[df[split_col] == split_name][id_col].tolist()
        path = os.path.join(output_dir, f"{split_name}_ids.txt")
        with open(path, "w") as f:
            f.write("\n".join(str(i) for i in ids))
        print(f"  {split_name}: {len(ids)} molecules -> {path}")
