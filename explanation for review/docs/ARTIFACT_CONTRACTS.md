# Artifact Contracts

This project survives Colab disconnects by treating every stage as an artifact contract.

## Curated Ligands

Path:

```text
02_curated_data/jak2_curated_ligands.csv
```

Required columns:

```text
mol_id,source,source_id,canonical_smiles,inchikey,activity_type,activity_value_nM,p_activity,target,assay_id,confidence_score,max_ring_size,has_macrocycle_12_20,has_constrained_ring_8_11,split,notes
```

## Ligand Features

Path:

```text
03_features/ligand_features.csv
```

Required columns:

```text
mol_id,canonical_smiles,mw,logp,tpsa,hbd,hba,rotatable_bonds,qed,num_atoms,num_heavy_atoms,num_rings,max_ring_size,formal_charge,passes_rdkit,passes_basic_filters
```

## Generated Candidates

Path:

```text
05_generated_candidates/merged/generated_merged_filtered.csv
```

Required columns:

```text
candidate_id,source_generator,parent_mol_id,canonical_smiles,inchikey,valid_rdkit,unique_flag,novel_flag,mw,logp,tpsa,hbd,hba,rotatable_bonds,qed,sa_score,murcko_scaffold,max_ring_size,has_macrocycle_12_20,has_constrained_ring_8_11,passes_basic_filters,generation_notes
```

## Docking Scores

Path:

```text
06_docking/scores/docking_scores.csv
```

Required columns:

```text
candidate_id,smiles,docking_engine,receptor_pdb,grid_center_x,grid_center_y,grid_center_z,grid_size_x,grid_size_y,grid_size_z,best_score,pose_rank,pose_file,pose_image,control_or_generated,notes
```

## Final Ranking

Path:

```text
08_final_ranking/final_ranked_candidates.csv
```

Required columns:

```text
rank,candidate_id,source_generator,smiles,inchikey,docking_score,pose_score,qed,sa_score,admet_score,safety_proxy_score,novelty_score,diversity_cluster,final_weighted_score,decision,main_risk
```
