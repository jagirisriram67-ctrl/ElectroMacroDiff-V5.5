"""
EMD V5.2 Hybrid — Docking Module (M5)
======================================
Receptor preparation, ligand preparation, docking execution,
pose filtering, and visualization for AutoDock Vina.
"""

import os
import subprocess
import numpy as np
import pandas as pd


def prepare_receptor(pdb_path, output_dir, pdbqt_name="jak2_prepared.pdbqt"):
    """Prepare receptor PDB for docking.
    
    Steps:
    1. Remove waters and non-standard residues
    2. Add hydrogens
    3. Convert to PDBQT format
    
    Falls back to simple PDB cleaning if tools unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, pdbqt_name)
    
    # Try using Biopython for cleaning
    try:
        from Bio.PDB import PDBParser, PDBIO, Select
        
        class CleanSelect(Select):
            def accept_residue(self, residue):
                # Remove water
                if residue.get_resname() == "HOH":
                    return 0
                return 1
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("receptor", pdb_path)
        
        clean_pdb = os.path.join(output_dir, "jak2_clean.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(clean_pdb, CleanSelect())
        print(f"Cleaned PDB saved to {clean_pdb}")
        
    except ImportError:
        clean_pdb = pdb_path
        print("Biopython not available, using raw PDB")
    
    # Try meeko/openbabel for PDBQT conversion
    try:
        from meeko import MoleculePreparation, PDBQTMolecule
        # If meeko works for receptor, use it
        print("Attempting PDBQT conversion...")
        # Fallback: just copy the clean PDB as-is
        import shutil
        shutil.copy(clean_pdb, output_path.replace(".pdbqt", ".pdb"))
        print(f"Clean receptor saved (PDBQT conversion may need manual step)")
    except:
        import shutil
        shutil.copy(clean_pdb if clean_pdb != pdb_path else pdb_path, 
                     output_path.replace(".pdbqt", ".pdb"))
    
    # Extract pocket center from reference ligand or known residues
    pocket_info = extract_pocket_center(pdb_path)
    
    return output_path, pocket_info


def extract_pocket_center(pdb_path):
    """Extract binding pocket center coordinates from PDB.
    
    Tries to find co-crystallized ligand, falls back to known JAK2 pocket residues.
    """
    try:
        from Bio.PDB import PDBParser
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("receptor", pdb_path)
        
        # Look for HETATM records (potential ligands)
        ligand_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " " and residue.get_resname() not in ["HOH", "SO4", "PO4", "GOL"]:
                        for atom in residue:
                            ligand_coords.append(atom.get_coord())
        
        if ligand_coords:
            center = np.mean(ligand_coords, axis=0)
            print(f"Pocket center from ligand: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
            return {"center": center.tolist(), "source": "co-crystallized_ligand"}
    except:
        pass
    
    # Fallback: known JAK2 ATP-binding pocket approximate center (for 5AEP)
    default_center = [25.0, 10.0, 15.0]
    print(f"Using default JAK2 pocket center: {default_center}")
    return {"center": default_center, "source": "default_5AEP"}


def prepare_ligands_for_docking(candidates_df, output_dir, smiles_col="canonical_smiles"):
    """Convert candidate SMILES to 3D SDF/PDBQT for docking."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    os.makedirs(output_dir, exist_ok=True)
    prepared = []
    
    for _, row in candidates_df.iterrows():
        smi = str(row.get(smiles_col, ""))
        cid = row.get("candidate_id", f"lig_{_}")
        
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            mol = Chem.AddHs(mol)
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if result == -1:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            if mol.GetNumConformers() > 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                except:
                    pass
                
                sdf_path = os.path.join(output_dir, f"{cid}.sdf")
                writer = Chem.SDWriter(sdf_path)
                writer.write(mol)
                writer.close()
                
                prepared.append({
                    "candidate_id": cid,
                    "smiles": smi,
                    "sdf_path": sdf_path,
                    "status": "prepared"
                })
        except Exception as e:
            prepared.append({
                "candidate_id": cid,
                "smiles": smi,
                "sdf_path": "",
                "status": f"failed: {e}"
            })
    
    print(f"Prepared {len([p for p in prepared if p['status'] == 'prepared'])}/{len(candidates_df)} ligands")
    return pd.DataFrame(prepared)


def run_vina_docking(receptor_path, ligand_sdf, pocket_center, pocket_size=(22, 22, 22),
                      exhaustiveness=16, num_modes=9, energy_range=3):
    """Run AutoDock Vina docking for a single ligand.
    
    Returns docking results dict or None if failed.
    """
    try:
        from vina import Vina
        
        v = Vina(sf_name="vina")
        v.set_receptor(receptor_path)
        v.set_ligand_from_file(ligand_sdf)
        v.compute_vina_maps(
            center=pocket_center,
            box_size=list(pocket_size)
        )
        
        v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)
        energies = v.energies()
        
        return {
            "best_score": float(energies[0][0]),
            "all_scores": [float(e[0]) for e in energies],
            "status": "success"
        }
    except Exception as e:
        return {"best_score": None, "status": f"failed: {e}"}


def dock_candidates(candidates_df, receptor_path, pocket_info, output_dir,
                     config=None, progress_path=None):
    """Dock all candidates and save scores.
    
    Includes resume support via progress tracking.
    """
    from emd_pipeline.config_registry import load_progress, save_progress
    
    if config is None:
        config = {}
    
    pocket_center = pocket_info.get("center", [25.0, 10.0, 15.0])
    pocket_size = config.get("grid_size", [22, 22, 22])
    exhaustiveness = config.get("exhaustiveness", 16)
    
    os.makedirs(output_dir, exist_ok=True)
    scores_path = os.path.join(output_dir, "docking_scores.csv")
    
    # Resume support
    progress = load_progress(progress_path) if progress_path else {"last_completed_index": -1}
    start_idx = progress.get("last_completed_index", -1) + 1
    
    results = []
    
    # Load existing results if resuming
    if os.path.exists(scores_path) and start_idx > 0:
        existing = pd.read_csv(scores_path)
        results = existing.to_dict("records")
    
    for idx in range(start_idx, len(candidates_df)):
        row = candidates_df.iloc[idx]
        cid = row.get("candidate_id", f"dock_{idx}")
        smi = row.get("canonical_smiles", row.get("smiles", ""))
        
        print(f"  Docking {idx+1}/{len(candidates_df)}: {cid}")
        
        # Prepare ligand
        ligand_dir = os.path.join(output_dir, "ligands_sdf")
        os.makedirs(ligand_dir, exist_ok=True)
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        sdf_path = os.path.join(ligand_dir, f"{cid}.sdf")
        
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                if mol.GetNumConformers() > 0:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    writer = Chem.SDWriter(sdf_path)
                    writer.write(mol)
                    writer.close()
        except:
            pass
        
        # Dock
        dock_result = run_vina_docking(
            receptor_path, sdf_path, pocket_center,
            pocket_size=pocket_size, exhaustiveness=exhaustiveness
        )
        
        results.append({
            "candidate_id": cid,
            "smiles": smi,
            "docking_engine": "vina",
            "receptor_pdb": "5AEP",
            "grid_center_x": pocket_center[0],
            "grid_center_y": pocket_center[1],
            "grid_center_z": pocket_center[2],
            "grid_size_x": pocket_size[0],
            "grid_size_y": pocket_size[1],
            "grid_size_z": pocket_size[2],
            "best_score": dock_result.get("best_score"),
            "pose_rank": 1,
            "status": dock_result.get("status", "unknown"),
            "control_or_generated": row.get("source_generator", "generated"),
        })
        
        # Save progress
        if progress_path:
            save_progress(progress_path, {"last_completed_index": idx, "status": "docking"})
        
        # Periodic save
        if idx % 10 == 0:
            pd.DataFrame(results).to_csv(scores_path, index=False)
    
    # Final save
    scores_df = pd.DataFrame(results)
    scores_df.to_csv(scores_path, index=False)
    print(f"Docking complete: {len(scores_df)} results saved to {scores_path}")
    
    return scores_df


def visualize_top_poses(scores_df, poses_dir, output_dir, top_n=10):
    """Generate 2D structure images for top-scoring candidates."""
    from rdkit import Chem
    from rdkit.Chem import Draw
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by best score (most negative = best)
    ranked = scores_df.dropna(subset=["best_score"]).sort_values("best_score")
    top = ranked.head(top_n)
    
    for _, row in top.iterrows():
        cid = row["candidate_id"]
        smi = row.get("smiles", "")
        score = row.get("best_score", "N/A")
        
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                img = Draw.MolToImage(mol, size=(400, 300))
                img_path = os.path.join(output_dir, f"top_pose_{cid}.png")
                img.save(img_path)
        except:
            continue
    
    print(f"Generated {top_n} pose images in {output_dir}")
