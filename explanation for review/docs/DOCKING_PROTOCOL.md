# Docking Protocol

## Current Prepared State

- Receptor source: `01_raw_data/pdb/5AEP.pdb`
- Reference ligand detected from PDB: `QUP:A:2000`
- Grid file: `06_docking/receptor/docking_grid_5AEP_QUP.json`
- Candidate SDF bundle: `06_docking/ligands_sdf/candidates_for_docking.sdf`
- Candidate SDF manifest: `06_docking/ligands_sdf/docking_input_manifest.csv`
- Cleaned receptor PDB: `06_docking/receptor/5AEP_receptor_clean.pdb`
- Prepared receptor PDBQT: `06_docking/receptor/jak2_prepared.pdbqt`
- Ligand PDBQT files: `06_docking/ligands_pdbqt/*.pdbqt`
- Vina manifest: `06_docking/scores/vina_command_manifest.csv`

## Grid

The grid is inferred from the co-crystallized ligand `QUP` in PDB `5AEP`:

```text
center_x = 32.530
center_y = 13.271
center_z = -3.863
size_x = 18.0
size_y = 18.0
size_z = 18.0
```

Visually verify this pocket before production docking.

## Required Next Conversion

Before Vina docking, these local preparation steps are now automated:

```powershell
python scripts/06_prepare_pdbqt.py --base .
python scripts/06_make_vina_manifest.py --base .
```

Then run Vina where the executable is available:

```powershell
python scripts/06_run_vina_manifest.py --base .
python scripts/06_parse_vina_results.py --base .
```

## Scientific Rule

Do not promote final molecules from ranking alone. Ranking before docking is only a queue for docking.
