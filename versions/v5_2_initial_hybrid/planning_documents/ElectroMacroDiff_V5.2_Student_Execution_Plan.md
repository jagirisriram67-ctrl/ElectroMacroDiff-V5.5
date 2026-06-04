# ElectroMacroDiff V5.2

## Realistic 5-Student, 8-10 Day Execution Plan

### 0. Reality Check

This version is designed for the actual team and compute situation:

- Team: 5 B.Tech second-year students.
- Time: 8-10 days.
- Compute: Google Colab free tier only.
- Accounts: each member has 2-3 Google accounts, but true parallel long-running Colab execution is unreliable and often not possible.
- Storage: Google Drive is the main persistence layer.
- AI tools available: ChatGPT, Codex, Gemini, Claude, DeepSeek, Grok, Qwen, Antigravity, and similar coding/research copilots.

Therefore, V5.2 does not depend on training a giant foundation model from scratch, running 11 modules simultaneously, or assuming 48-hour uninterrupted GPU sessions. It is a checkpointed, resume-first, staged pipeline that produces a credible computational prototype and final research dossier.

The mindset: do not try to out-compute pharma. Out-organize the pipeline, use strong pretrained tools, cache everything, and make every notebook resumable.

### 1. New Strategic Goal

Build a working end-to-end computational drug discovery prototype for JAK2-focused macrocyclic/constrained inhibitor design.

Final deliverable after 8-10 days:

- 20-50 generated or selected candidate molecules.
- 5-10 validated shortlist candidates.
- 3-5 final proposed molecules.
- Docking/pose evidence.
- Basic ADMET and toxicity-risk screening.
- Synthesizability estimation.
- Final Target Product Profile style report.
- All notebooks, checkpoints, intermediate files, and outputs saved to Google Drive.

The final claim should be:

> "We built a low-resource, reproducible computational pipeline that proposes and prioritizes JAK2 macrocyclic/constrained inhibitors for future experimental testing."

Do not claim:

- experimentally proven inhibitors
- clinical safety
- true near-FEP accuracy
- guaranteed synthesizability
- FDA-ready molecules

### 2. Simplified V5.2 Architecture

V5.0/V5.1 had too many heavy modules for your real constraints. V5.2 compresses the system into 7 practical modules.

| Module | Name | Purpose | Compute Level |
|---|---|---|---|
| M0 | Drive Registry and Resume System | Save every artifact, checkpoint, config, and log | CPU |
| M1 | Data Builder | Collect JAK2 ligands, macrocycle/constrained molecules, and target structures | CPU |
| M2 | Ligand Featurizer | Clean SMILES, detect rings, generate conformers, calculate descriptors | CPU |
| M3 | Candidate Generator | Generate/expand molecules using lightweight methods and optional fine-tuning | CPU/GPU optional |
| M4 | Docking and Pose Filter | Dock candidates against JAK2 pocket | CPU/GPU optional |
| M5 | ADMET, Toxicity, and Synth Filter | Drug-likeness, PAINS, toxicity alerts, SLC19A3 proxy, synthesis score | CPU |
| M6 | Ranking and TPP Generator | Final ranking, molecule cards, report, visuals | CPU |

This is deliberately smaller. A finished 7-module pipeline is much more valuable than an unfinished 12-module dream.

### 3. Google Drive First: Required Folder Structure

Create one shared Google Drive folder:

```text
EMD_V5_2/
  00_project_registry/
  01_raw_data/
  02_curated_data/
  03_features/
  04_models_checkpoints/
  05_generated_candidates/
  06_docking/
  07_admet_synthesis/
  08_final_ranking/
  09_reports/
  10_notebooks/
  11_logs/
```

Every notebook must start with:

```python
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/EMD_V5_2"
```

Every long step must save:

- input file path
- output file path
- timestamp
- random seed
- package versions
- number of molecules processed
- number passed
- number failed
- resume index

### 4. Resume-First Rule

No module is allowed to process everything in one fragile run.

Every loop must follow this pattern:

```python
import os, json, time

progress_path = f"{BASE}/00_project_registry/progress_m3_generation.json"

if os.path.exists(progress_path):
    progress = json.load(open(progress_path))
    start_idx = progress["last_completed_index"] + 1
else:
    progress = {"last_completed_index": -1}
    start_idx = 0

for i in range(start_idx, len(items)):
    # process one molecule / batch
    result = process(items[i])
    save_result(result)

    progress["last_completed_index"] = i
    progress["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(progress, open(progress_path, "w"), indent=2)
```

For model training:

- save checkpoint every 100-500 steps
- save best checkpoint separately
- save optimizer state if continuing training
- save LoRA adapter only if using PEFT
- save training config with the checkpoint

### 5. Datasets

#### 5.1 Must-Have Datasets

| Dataset | Purpose | Practical Access |
|---|---|---|
| ChEMBL | JAK2 known inhibitors and activities | CSV/API/export |
| BindingDB | additional JAK2 activity data | downloadable TSV |
| RCSB PDB | JAK2 protein structures | PDB files |
| PubChem | molecule metadata and backup bioactivity | CSV/API |

Primary structure:

- JAK2 structure such as PDB 5AEP.

Target panel if time allows:

- JAK1
- JAK3
- TYK2
- FLT3

Safety proxy:

- SLC19A3 risk is difficult to prove structurally in a student sprint. Use it as a proxy screen using ligand similarity, literature-derived risk notes, and any available predicted off-target model. Clearly label it as a proxy.

#### 5.2 Dataset Sizes

Use three practical levels:

| Set | Size | Use |
|---|---:|---|
| Tiny Debug Set | 20-30 molecules | test every notebook fast |
| Main Sprint Set | 300-700 molecules | actual pipeline |
| Backup Broad Set | 1,000-3,000 molecules | if macrocycles are too few |

Do not wait for a perfect Elite 500. If only 120 high-quality macrocycle/constrained examples exist, use them and augment with related constrained JAK2 inhibitors.

### 6. Model Strategy: What To Actually Use

#### 6.1 Avoid Heavy Training Unless Necessary

With free Colab and 8-10 days, the highest-risk step is training a large 3D generative model. V5.2 should prioritize:

1. retrieve known potent/constrained ligands
2. generate analogs and constrained variants
3. dock/filter/rank
4. present a strong reproducible pipeline

#### 6.2 Candidate Generation Options

Use these in order:

| Option | Method | Why |
|---|---|---|
| A | RDKit analog generation and fragment recombination | most reliable, CPU-friendly |
| B | SELFIES/SMILES mutation around known JAK2 scaffolds | simple, generates novelty |
| C | REINVENT/GuacaMol-style pretrained SMILES generator if easy to run | useful but optional |
| D | Fine-tune small molecule language model with LoRA | only if setup is smooth |
| E | Heavy SE(3) diffusion/flow model | stretch goal only |

Practical generation methods:

- replace substituents on known JAK2 scaffolds
- macrocyclize linker atoms where valence permits
- use SELFIES mutations to keep validity high
- filter by ring size, molecular weight, TPSA, logP, rotatable bonds
- cluster to preserve diversity

#### 6.3 Docking Strategy

Primary:

- DiffDock if install succeeds and runtime is acceptable.

Fallback:

- AutoDock Vina or GNINA.

Minimum acceptable:

- Use one reliable docking method on 50-100 final filtered molecules.
- Save poses and scores.
- Add visual pose snapshots for top 5.

#### 6.4 ADMET and Synthesizability

Use lightweight tools first:

- RDKit descriptors
- QED
- SA score
- PAINS filters
- Brenk filters if available
- SwissADME web/manual for top candidates if necessary
- AiZynthFinder only for final top 5-10 if install works

If AiZynthFinder is too heavy:

- use SA score
- purchasable fragment similarity
- manually propose simple retrosynthetic logic for top molecules
- clearly call it "preliminary synthesizability assessment"

### 7. Team Allocation

| Member | Role | Owns |
|---|---|---|
| Student 1 | Project integrator | Drive structure, registry, final report, merging outputs |
| Student 2 | Data lead | ChEMBL/BindingDB/PDB collection and cleaning |
| Student 3 | Chemistry lead | RDKit features, filters, ring detection, candidate generation |
| Student 4 | Docking lead | protein preparation, docking, pose screenshots |
| Student 5 | Evaluation lead | ADMET, synthesis score, ranking, TPP tables |

Daily rule:

- Everyone pushes outputs to Drive before sleeping.
- Everyone writes a 5-line daily log in `11_logs/`.
- No one keeps important files only inside Colab runtime.

### 8. AI Tool Usage Plan

Use AI tools aggressively, but assign them clear jobs.

| Tool Type | Use |
|---|---|
| ChatGPT / Codex | write and debug notebooks, create report sections, generate pipeline code |
| Gemini | summarize papers, compare target biology, help with Google/Colab ecosystem |
| Claude | long-form report polishing, risk explanation, methods section |
| DeepSeek / Qwen | code alternatives, error debugging, optimization ideas |
| Grok | fast brainstorm, sanity checks, presentation wording |
| Antigravity / Codex | codebase/notebook organization, file generation, automation |

Important rule:

- Never paste AI-generated code directly into the final pipeline without running it on the Tiny Debug Set.
- Every AI answer must be converted into working notebook cells and saved outputs.
- AI tools can help write code, but your project credibility comes from saved artifacts and reproducible runs.

### 9. Notebook Plan

Create these notebooks:

```text
10_notebooks/
  00_environment_check.ipynb
  01_data_collection_curation.ipynb
  02_ligand_features_filters.ipynb
  03_candidate_generation.ipynb
  04_protein_preparation_docking.ipynb
  05_admet_synthesis_filtering.ipynb
  06_final_ranking_tpp_generation.ipynb
```

Each notebook must have:

- install cell
- Drive mount cell
- config cell
- resume/progress cell
- tiny test mode
- main run mode
- output validation cell
- final summary cell

### 10. Module Details

#### M0 - Drive Registry and Resume System

Input:

- project folder
- team member names
- campaign config

Output:

- `campaign_config.yaml`
- `run_registry.csv`
- `progress_*.json`
- `environment_versions.txt`

Success:

- any notebook can disconnect and resume.

#### M1 - Data Builder

Input:

- ChEMBL JAK2 records
- BindingDB JAK2 records
- PDB structures

Process:

- standardize SMILES
- convert activities to nM
- calculate pIC50/pKi where possible
- remove duplicates
- label macrocycles and constrained molecules

Output:

- `02_curated_data/jak2_curated_ligands.csv`
- `02_curated_data/jak2_macrocycle_or_constrained.csv`
- `01_raw_data/jak2_pdb_structures/`

Minimum target:

- 300 usable ligands total
- at least 50 macrocycle/constrained molecules, if available

#### M2 - Ligand Features and Filters

Input:

- curated ligand CSV

Process:

- RDKit sanitize
- ring detection
- descriptors: MW, logP, TPSA, HBD, HBA, rotatable bonds, QED
- macrocycle flags
- PAINS filters
- conformer generation for candidates

Output:

- `03_features/ligand_features.csv`
- `03_features/ligand_conformers.sdf`

Success:

- more than 90% of curated molecules pass RDKit sanitization.

#### M3 - Candidate Generation

Input:

- known potent ligands
- fragment/scaffold table
- generation config

Process:

- SELFIES mutation
- RDKit substituent replacement
- linker/ring-closure attempts
- property filtering
- novelty filtering against training molecules

Output:

- `05_generated_candidates/generated_raw.csv`
- `05_generated_candidates/generated_filtered.csv`
- `05_generated_candidates/generated_filtered.sdf`

Minimum target:

- 500-2,000 raw candidates
- 100-300 filtered candidates

Stretch goal:

- LoRA fine-tune a small SMILES model and compare generated molecules.

#### M4 - Docking and Pose Filter

Input:

- filtered candidates
- prepared JAK2 PDB

Process:

- prepare protein
- define binding pocket
- dock 50-150 candidates
- save scores and poses
- generate images for top 10

Output:

- `06_docking/docking_scores.csv`
- `06_docking/poses/`
- `06_docking/top_pose_images/`

Minimum target:

- at least 50 successfully docked molecules.

#### M5 - ADMET, Toxicity, and Synth Filter

Input:

- top docked candidates

Process:

- QED
- SA score
- PAINS/Brenk
- Lipinski/Veber rules
- hERG/CYP/toxicity predictor if available
- SLC19A3 proxy risk notes
- AiZynthFinder for top 5-10 if possible

Output:

- `07_admet_synthesis/admet_scores.csv`
- `07_admet_synthesis/synthesis_scores.csv`
- `07_admet_synthesis/safety_notes.csv`

Minimum target:

- top 10 candidates fully scored.

#### M6 - Final Ranking and TPP

Input:

- docking scores
- ADMET scores
- synthesis scores
- diversity clusters

Ranking formula:

- docking score: 35%
- pose sanity: 15%
- ADMET/drug-likeness: 20%
- synthesizability: 15%
- novelty/diversity: 10%
- safety proxy: 5%

Output:

- `08_final_ranking/final_ranked_candidates.csv`
- `09_reports/EMD_V5_2_TPP.pdf`
- `09_reports/EMD_V5_2_Final_Report.docx`
- `09_reports/EMD_V5_2_Presentation.pptx`

### 11. 10-Day Execution Schedule

#### Day 1 - Setup and Tiny Pipeline

Goal:

- create Drive folder
- create notebooks
- install/check RDKit
- test 20 molecules through M1-M2

Deliverable:

- Drive registry working
- Tiny Debug Set saved

#### Day 2 - Data Collection

Goal:

- collect ChEMBL/BindingDB JAK2 ligands
- download JAK2 PDB
- curate main ligand table

Deliverable:

- `jak2_curated_ligands.csv`

#### Day 3 - Features and Filtering

Goal:

- compute descriptors
- detect macrocycle/constrained molecules
- produce initial scaffold list

Deliverable:

- `ligand_features.csv`

#### Day 4 - Candidate Generation v1

Goal:

- generate candidates using RDKit/SELFIES
- filter invalid molecules

Deliverable:

- 500-2,000 raw candidates
- 100-300 filtered candidates

#### Day 5 - Protein Preparation and Docking Test

Goal:

- prepare JAK2 protein
- dock 5-10 known ligands first as control
- then dock first 30 generated molecules

Deliverable:

- docking method validated
- first docking scores saved

#### Day 6 - Main Docking Run

Goal:

- dock 50-150 filtered candidates
- save poses and scores

Deliverable:

- `docking_scores.csv`
- top pose files

#### Day 7 - ADMET and Synth Filter

Goal:

- score top 30 docked candidates
- remove bad candidates

Deliverable:

- top 10 shortlist

#### Day 8 - Final Ranking and Molecule Cards

Goal:

- final rank top 10
- create molecule cards for top 3-5

Deliverable:

- `final_ranked_candidates.csv`
- top molecule images

#### Day 9 - Report and TPP

Goal:

- create final report
- create TPP-style document
- prepare slides

Deliverable:

- PDF/docx report draft
- presentation draft

#### Day 10 - Hardening and Demo

Goal:

- rerun tiny pipeline to prove reproducibility
- fix broken links/files
- polish final outputs

Deliverable:

- final Drive package ready for submission/demo.

### 12. 8-Day Compressed Schedule

If only 8 days are available:

- Day 1: setup + data
- Day 2: curation + features
- Day 3: generation
- Day 4: docking setup + controls
- Day 5: main docking
- Day 6: ADMET/synthesis/ranking
- Day 7: report + TPP
- Day 8: verification + presentation

Skip heavy model fine-tuning. Use RDKit/SELFIES generation and docking.

### 13. Optional Stretch Goals

Only attempt these after the core pipeline works:

1. LoRA fine-tune a small SMILES generator.
2. DiffDock instead of Vina/GNINA.
3. AiZynthFinder routes for top 5.
4. Off-target docking panel.
5. Macro-Equi-Diff/flow model experiment on tiny set.

If a stretch goal breaks, abandon it quickly and protect the main deliverable.

### 14. Minimum Viable Success

Even if compute fails, the project succeeds if you deliver:

- curated JAK2 ligand dataset
- generated candidate set
- docking of at least 50 candidates
- ADMET/synthesis filtering
- final ranked top 3-5 candidates
- reproducible notebooks
- saved Drive outputs
- honest final report explaining limitations

### 15. Best Final Story

The strongest final story is not:

> "We trained a pharma-grade foundation model."

The strongest final story is:

> "Under severe compute limits, we engineered a reproducible, checkpointed, low-resource generative screening pipeline for JAK2 macrocyclic/constrained inhibitor discovery. The system integrates curated bioactivity data, rule-guided molecular generation, docking-based prioritization, ADMET filtering, synthesizability estimation, and final TPP-style reporting. Every stage saves restartable artifacts to Google Drive."

That is believable, impressive, and executable by a 5-student team in 8-10 days.

