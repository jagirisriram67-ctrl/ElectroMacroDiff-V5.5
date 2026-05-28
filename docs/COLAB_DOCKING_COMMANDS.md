# Colab Docking Commands

These commands are the practical next gate after `06_prepare_docking_inputs.py`.

Primary references:

- Meeko ligand preparation docs: <https://meeko.readthedocs.io/en/develop/lig_prep_basic.html>
- Meeko receptor preparation docs: <https://meeko.readthedocs.io/en/develop/rec_cli_options.html>
- AutoDock Vina manual: <https://vina.scripps.edu/manual/>

## Install

```python
!pip -q install vina meeko
```

## Automated Prep Scripts

The repository now provides the preparation scripts directly:

```python
BASE = "/content/drive/MyDrive/EMD_V5_2_Hybrid"
!python "{BASE}/scripts/06_prepare_docking_inputs.py" --base "{BASE}" --top-n 150
!python "{BASE}/scripts/06_prepare_pdbqt.py" --base "{BASE}"
!python "{BASE}/scripts/06_make_vina_manifest.py" --base "{BASE}"
```

## Manual Ligand Prep Reference

Meeko provides `mk_prepare_ligand.py` for ligand PDBQT creation from 3D structures.

```python
from pathlib import Path
import subprocess

BASE = Path("/content/drive/MyDrive/EMD_V5_2_Hybrid")
sdf_dir = BASE / "06_docking" / "ligands_sdf"
pdbqt_dir = BASE / "06_docking" / "ligands_pdbqt"
pdbqt_dir.mkdir(parents=True, exist_ok=True)

for sdf in sdf_dir.glob("CAND_*.sdf"):
    out = pdbqt_dir / f"{sdf.stem}.pdbqt"
    if out.exists():
        continue
    subprocess.run([
        "mk_prepare_ligand.py",
        "-i", str(sdf),
        "-o", str(out),
    ], check=False)
```

## Prepare Receptor

Receptor preparation is the part most likely to need manual attention because waters, ions, co-crystallized ligand, protonation, and residue templates matter.

```python
BASE = Path("/content/drive/MyDrive/EMD_V5_2_Hybrid")
receptor_pdb = BASE / "01_raw_data" / "pdb" / "5AEP.pdb"
receptor_out = BASE / "06_docking" / "receptor" / "jak2_prepared.pdbqt"

!mk_prepare_receptor.py -i "{receptor_pdb}" -o "{receptor_out}"
```

If receptor preparation fails because of modified residues/templates, use ChimeraX/AutoDockTools/OpenBabel as fallback, then save the final file exactly here:

```text
06_docking/receptor/jak2_prepared.pdbqt
```

## Build Vina Manifest

```python
import json
from pathlib import Path

BASE = Path("/content/drive/MyDrive/EMD_V5_2_Hybrid")
grid = json.loads((BASE / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json").read_text())

!python "{BASE / 'scripts' / '06_make_vina_manifest.py'}" \
  --base "{BASE}" \
  --center {grid["center"]["x"]} {grid["center"]["y"]} {grid["center"]["z"]} \
  --size {grid["size"]["x"]} {grid["size"]["y"]} {grid["size"]["z"]} \
  --exhaustiveness 16
```

## Run Vina Commands

```python
import pandas as pd
import subprocess

manifest = BASE / "06_docking" / "scores" / "vina_command_manifest.csv"
jobs = pd.read_csv(manifest)

for i, row in jobs.iterrows():
    if row.get("status") == "completed":
        continue
    result = subprocess.run(row["command"], shell=True)
    jobs.loc[i, "status"] = "completed" if result.returncode == 0 else "failed"
    jobs.to_csv(manifest, index=False)
```

After docking, parse logs into `06_docking/scores/docking_scores.csv`:

```python
!python "{BASE / 'scripts' / '06_parse_vina_results.py'}" --base "{BASE}"
```

Then rerun:

```python
!python "{BASE / 'scripts' / '07_rank_candidates.py'}" --base "{BASE}"
!python "{BASE / 'scripts' / '08_validate_project.py'}" --base "{BASE}"
!python "{BASE / 'scripts' / '09_make_report_assets.py'}" --base "{BASE}" --top-n 10
```
