# ElectroMacroDiff V5.1

## End-to-End Upgradation and Implementation Blueprint

### 0. Executive Upgrade

ElectroMacroDiff V5.1 upgrades the V5.0 hackathon blueprint into a reproducible, checkpointed, audit-ready computational discovery pipeline for JAK2 macrocyclic inhibitor design. V5.0 proved the strategy: split the work across an 11-account Colab array, precompute static chemistry, fine-tune only a small generative adapter, then validate aggressively.

V5.1 keeps that low-cost decentralized idea, but fixes the weak points:

- Replace ChEMBL 33 with the current ChEMBL 36 baseline, plus BindingDB and PDBBind cross-checks.
- Add a real data governance layer: assay confidence, units, duplicates, provenance, leakage control, and train/validation split rules.
- Replace a single "Elite 500" cut with a tiered dataset: Elite 500 for sprint tuning, Elite 2K for robustness, and Decoy/Negative sets for selectivity and toxicity.
- Convert the 11-account setup into named reproducible nodes with artifact contracts.
- Add checkpoint/resume and manifest files for every stage so Colab disconnects do not destroy progress.
- Add selectivity gates against JAK1, JAK3, TYK2, FLT3, and the SLC19A3 safety liability.
- Add a final uncertainty layer so no molecule is promoted on one model's score alone.

The final deliverable is no longer just a visual TPP. It is a full preclinical computational dossier: final molecules, structures, docking/affinity evidence, synthesizability routes, ADMET risk, uncertainty, provenance, and go/no-go decisions.

### 1. Strategic Objective

Goal: generate, validate, and rank 3-5 synthetically plausible, target-fitted macrocyclic JAK2 inhibitor candidates in 5-7 days using free or low-cost infrastructure.

Target product:

- Primary target: human JAK2 kinase domain.
- Desired modality: macrocyclic or constrained small-molecule inhibitor.
- Desired ring size: 12-20 atoms by default, expandable to 10-24 for fallback.
- Desired activity proxy: strong predicted binding to JAK2 ATP/allosteric pocket, depending on selected campaign.
- Desired selectivity: lower predicted binding to JAK1, JAK3, TYK2, FLT3.
- Safety gate: low predicted interaction risk with SLC19A3.
- Output count: 3-5 final candidates, plus 10 backup candidates.

Important scientific boundary: computational affinity, docking, and toxicity predictions are decision-support signals, not proof of biological efficacy or clinical safety. V5.1 is designed to produce prioritized hypotheses for synthesis and testing.

### 2. Infrastructure Architecture

#### 2.1 Compute Layout

V5.1 uses a decentralized 11-node Colab cluster, but each node has a strict role and artifact contract.

| Node | Role | Main Work | Output |
|---|---|---|---|
| 1 | Data curator | ChEMBL, BindingDB, PubChem, PDB metadata ingestion | Canonical assay table |
| 2 | Ligand featurizer | RDKit cleanup, macrocycle detection, conformers | Ligand graph tensors |
| 3 | Pocket featurizer | JAK2/JAK family structures, pocket graphs | Protein/pocket tensors |
| 4 | Site predictor | Attachment-point scoring, linker/cyclization priors | Attachment probability maps |
| 5 | Generative trainer A | QLoRA/adapter fine-tuning | Adapter checkpoints |
| 6 | Generative trainer B | Parallel seed or ablation run | Backup adapter checkpoints |
| 7 | Generator | Batch inference and structure repair | Generated SDF/SMILES |
| 8 | Structure validator | ring strain, clash, geometry, conformer ensemble | Validated structures |
| 9 | Binding validator | DiffDock/Boltz-style scoring, consensus docking | Affinity-ranked table |
| 10 | Synth/ADMET | AiZynthFinder, SA score, ADMET/off-target filters | Synth/ADMET-ranked table |
| 11 | Dossier compiler | Reports, renders, manifests, TPP | Final TPP and audit pack |

#### 2.2 Central Artifact Hub

Recommended: one Google Cloud Storage bucket or Hugging Face Dataset repository.

Required folders:

```text
emd-v5.1/
  00_registry/
  01_raw_data/
  02_curated_data/
  03_ligand_features/
  04_pocket_features/
  05_training_sets/
  06_checkpoints/
  07_generated/
  08_validation/
  09_synthesis_admet/
  10_reports/
```

Every artifact must include:

- `manifest.json`
- `source_versions.json`
- `sha256sums.txt`
- `run_config.yaml`
- `metrics.json`

#### 2.3 Standard Runtime Requirements

Core:

- Python 3.10 or 3.11
- PyTorch with CUDA runtime matching Colab
- torch-geometric
- RDKit
- pandas, numpy, scipy, scikit-learn
- biopython
- networkx
- py3Dmol
- matplotlib, seaborn
- tqdm, rich

Generative and training:

- transformers
- accelerate
- peft
- bitsandbytes
- safetensors
- einops
- e3nn or torchmd/equivariant dependency required by selected generator
- mamba-ssm only if the Mamba predictor is used

Structure and docking:

- DiffDock or DiffDock-L environment
- Boltz-2 or available Boltz inference package
- Open Babel
- Meeko or docking-format helper if using Vina fallback
- AutoDock Vina/GNINA as fallback only

Retrosynthesis and ADMET:

- AiZynthFinder
- stock/building-block database for AiZynthFinder
- PAINS/Brenk filters through RDKit or medicinal chemistry filter package
- Optional: ADMET-AI, pkCSM-style local predictor, or Chemprop ADMET model

Workflow:

- DVC or simple manifest-based versioning
- wandb offline mode or TensorBoard
- pydantic for schema validation
- pytest for contract tests

### 3. Datasets

#### 3.1 Primary Bioactivity Data

| Dataset | Use | Required Fields |
|---|---|---|
| ChEMBL 36 | Primary curated bioactivity source | compound ID, target ID, assay ID, IC50/Ki/Kd, units, confidence score, relation |
| BindingDB | Cross-check potency and patent-derived JAK2 data | SMILES, target, Ki/Kd/IC50, publication/patent source |
| PubChem BioAssay | Supplemental JAK family activity and recent assays | AID, CID, activity, target, source |

ChEMBL 36 is the V5.1 default because the official ChEMBL downloads page lists release 36 as the current release from July 2025.

#### 3.2 Protein Structure Data

| Dataset | Use |
|---|---|
| RCSB PDB JAK2 crystal structures | Pocket extraction, ligand pose references |
| PDBBind refined/general set | Docking and affinity calibration |
| CrossDocked or PoseBusters-style sets | Pose validation and docking failure detection |
| AlphaFold/UniProt structures | Backup structures for off-target selectivity where crystal coverage is weak |

Minimum JAK2 seed structure:

- PDB 5AEP: JAK2 transferase structure, X-ray diffraction, 1.95 A resolution.

Recommended target panel:

- JAK2 primary pocket
- JAK1, JAK3, TYK2 selectivity panel
- FLT3 kinase risk panel
- SLC19A3 safety/off-target panel or homology/structure proxy if no validated ligand-bound structure is available

#### 3.3 Molecular Pretraining and Decoys

| Dataset | Use |
|---|---|
| ZINC / ZINC-like purchasable molecules | Generator pretraining or negative background |
| Enamine REAL / building-block stock | Synthesizability-aware filtering |
| DUD-E / property-matched decoys | Classifier calibration and false-positive control |
| Macrocycle-specific literature/PubChem subset | Macrocycle prior and ring geometry statistics |

#### 3.4 Dataset Tiers

| Tier | Size | Purpose |
|---|---:|---|
| Elite 500 | 500 | Fast QLoRA sprint tuning on highest-confidence JAK2 macrocycle-like binders |
| Elite 2K | 1,500-2,500 | Robustness, validation, diversity, fallback if Elite 500 overfits |
| Selectivity Set | variable | JAK1/JAK2/JAK3/TYK2/FLT3 comparative activity |
| Decoy Set | 5K-20K | Calibration and false-positive reduction |
| Synthesis Stock Set | variable | AiZynthFinder purchasability and route scoring |

### 4. Data Curation Rules

#### 4.1 Assay Standardization

Keep compounds only when:

- Target organism is human or clearly mappable to human JAK2.
- Assay type is biochemical binding or enzymatic inhibition.
- Standard relation is exact or bounded in a usable direction.
- Units can be converted to nM.
- Activity type is IC50, Ki, or Kd.
- Assay confidence score is high enough for the target definition.

Normalize:

- Convert IC50/Ki/Kd to nM.
- Convert to pActivity: `pX = -log10(value_molar)`.
- Canonicalize salts, tautomers, stereochemistry, and protonation state.
- Deduplicate by InChIKey and retain median or best-confidence activity.

Reject or quarantine:

- Ambiguous mixtures.
- Missing stereochemistry where stereochemistry is pharmacologically critical.
- Reactive/toxic structural alerts unless specifically justified.
- Molecules failing basic chemistry sanitization.

#### 4.2 Macrocycle Detection

Primary macrocycle label:

- RDKit ring system contains at least one ring of size 12-20.

Secondary constrained-molecule label:

- Ring size 8-11 or fused/bridged constrained topology with macrocycle-like geometry.

Fallback:

- If true JAK2 macrocycles are too sparse, train on high-potency JAK2 binders plus a macrocycle linker-generation prior from broader macrocycle datasets.

### 5. Module Specification

#### M0 - Orchestrator and Artifact Registry

Purpose: make the decentralized Colab workflow reproducible.

Input:

- `campaign.yaml`
- node assignment table
- cloud credentials

Process:

- Creates folder structure.
- Validates artifact schemas.
- Tracks run IDs, seeds, and versions.

Output:

- `registry.sqlite`
- `manifest.json`
- node-specific run configs

Success metric:

- Every downstream file can be traced back to raw data and code version.

#### M1 - Data Curator

Input:

- ChEMBL 36 dump or API export
- BindingDB export
- PubChem assay exports

Process:

- Query JAK2 and JAK family target records.
- Standardize potency.
- Remove duplicates and low-confidence assays.
- Assign split groups by scaffold and publication to reduce leakage.

Output:

- `curated_activity.parquet`
- `elite_500.csv`
- `elite_2k.csv`
- `selectivity_panel.csv`
- `decoy_panel.csv`

Key metrics:

- number of usable JAK2 compounds
- number of macrocycle/constrained compounds
- pActivity distribution
- scaffold diversity
- assay-source diversity

#### M2 - Ligand Electro-Graph Encoder

Input:

- curated SMILES/SDF

Process:

- RDKit standardization.
- Generate 20-100 conformers per molecule depending on size.
- MMFF/UFF minimization.
- Compute atom and bond features.

Atom features:

- atomic number
- formal charge
- aromaticity
- hybridization
- valence
- ring membership
- macrocycle ring index
- Gasteiger charge
- donor/acceptor flag
- hydrophobic flag
- partial charge proxy
- 3D coordinates

Bond features:

- bond type
- conjugation
- stereochemistry
- ring bond flag
- rotatable bond flag
- estimated bond order

Output:

- `ligand_graphs.pt`
- `ligand_conformers.sdf`
- `ligand_features.parquet`

Success metric:

- more than 95% ligand featurization success for curated set.

#### M3 - Pocket and Target Panel Encoder

Input:

- JAK2 PDB structures
- off-target structures or predicted structures
- reference ligand coordinates

Process:

- Clean structures.
- Resolve chains, missing atoms, protonation assumptions.
- Extract 8-12 A pocket around reference ligand or ATP-binding residues.
- Build residue/atom graph.

Pocket features:

- residue type
- atom type
- charge proxy
- hydrophobicity
- H-bond donor/acceptor
- secondary structure if available
- distance to pocket centroid
- solvent exposure proxy

Output:

- `pocket_graphs.pt`
- `target_panel_graphs.pt`
- `pocket_metadata.json`

Success metric:

- each selected target has a validated pocket graph and visual inspection snapshot.

#### M4 - Attachment and Cyclization Site Predictor

Input:

- ligand graphs
- pocket graphs
- known/candidate linker breakpoints

Model options:

- Primary: GCN-Mamba or lightweight graph transformer.
- Fallback: rule-based RDKit attachment-point scorer with learned classifier.

Process:

- Predict viable fragment attachment atoms.
- Score cyclization feasibility before generation.
- Freeze model during V5.1 sprint unless enough labeled data exists.

Output:

- `attachment_scores.parquet`
- `fragment_pairs.sdf`
- `cyclization_priors.pt`

Success metric:

- attachment sites are chemically valid and do not create impossible valence/ring geometries.

#### M5 - Macrocycle Generator

Input:

- ligand/pocket tensors
- attachment priors
- training split

Primary model:

- PropMolFlow-style geometry-complete SE(3)-equivariant flow matching model.

Alternative/backup:

- FlowMol3-style 3D flow matching for all-atom molecule generation.
- Macro-Equi-Diff-style E(3)-equivariant diffusion for scaffold-based macrocycle generation.

Training strategy:

- Load base model in 4-bit NF4 where supported.
- Inject LoRA/bf16 adapters into geometry and transannular attention layers.
- Freeze base model.
- Use micro-batch 1-4 with gradient accumulation.
- Save checkpoint every 250-500 steps.
- Use early stopping on validity, ring closure, and validation loss.

Generation strategy:

- Generate 2,000-5,000 raw candidates in V5.1, not just 500-1,000.
- Use multiple seeds across accounts.
- Enforce hard constraints: ring size, valence, atom set, max molecular weight, rotatable bonds, synthetic alert limits.

Output:

- `generated_raw.sdf`
- `generated_raw.csv`
- `adapter_checkpoint/`
- `generation_metrics.json`

Success metrics:

- validity above 85%
- macrocycle closure above 70%
- uniqueness above 90%
- novelty above 70% versus training set
- no single scaffold family dominates more than 20% of accepted candidates

#### M6 - Structure Repair and Geometry Filter

Input:

- generated raw SDF/SMILES

Process:

- RDKit sanitization.
- Conformer regeneration if needed.
- MMFF/UFF local minimization.
- Ring strain proxy calculation.
- Clash and impossible bond angle rejection.
- PoseBusters-like checks where available.

Output:

- `generated_clean.sdf`
- `geometry_pass.csv`
- `geometry_fail.csv`

Success metric:

- at least 500 clean, unique, macrocyclic candidates survive for validation.

#### M7 - Diversity and Novelty Filter

Input:

- clean generated molecules
- training molecules

Process:

- ECFP4/Tanimoto clustering.
- Bemis-Murcko scaffold clustering.
- remove near-duplicates.
- keep top candidates per cluster.

Output:

- `diverse_candidates.sdf`
- `diversity_report.html`

Success metric:

- 100-300 diverse candidates remain before heavy validation.

#### M8 - Binding and Structure Validation

Input:

- diverse candidates
- JAK2 pocket structure
- target panel structures

Models:

- DiffDock/DiffDock-L for pose generation.
- Boltz-2-style structure/affinity prediction where available.
- Vina/GNINA only as backup or consensus diversity signal.

Process:

- Dock each molecule into JAK2.
- Run at least 3 seeds per candidate where runtime allows.
- Score pose confidence, predicted affinity, pocket interaction pattern.
- Penalize poses missing hinge or key pocket interactions.

Output:

- `jak2_docking_results.parquet`
- `poses/`
- `binding_ranked.csv`

Success metric:

- 30-50 candidates survive primary JAK2 binding screen.

#### M9 - Selectivity and Safety Gate

Input:

- top JAK2 candidates
- JAK1/JAK3/TYK2/FLT3/SLC19A3 structures or proxies

Process:

- Dock/score against off-targets.
- Compute selectivity margin.
- Flag pan-JAK or FLT3-heavy candidates unless intended.
- Run SLC19A3 interaction-risk proxy.
- Apply ADMET and medicinal chemistry filters.

ADMET filters:

- PAINS
- Brenk alerts
- predicted hERG risk
- CYP inhibition flags
- solubility
- permeability
- molecular weight and polar surface area
- macrocycle-specific permeability warning

Output:

- `selectivity_safety_ranked.csv`
- `admet_report.csv`

Success metric:

- 10-20 candidates pass selectivity/safety triage.

#### M10 - Synthesizability and Retrosynthesis

Input:

- 10-20 triaged candidates
- purchasable building-block stock

Model/tool:

- AiZynthFinder 4.x or current available installation.

Process:

- Run retrosynthesis with multiple search settings.
- Prioritize routes with accessible ring-closure chemistry.
- Score route length, stock availability, reaction confidence, and macrocyclization risk.
- Reject candidates with no plausible route.

Output:

- `synthesis_routes.json`
- `synthesis_ranked.csv`
- route images

Success metric:

- at least 3-5 candidates have plausible synthetic routes.

#### M11 - Consensus Ranker and Uncertainty

Input:

- binding scores
- selectivity scores
- geometry scores
- ADMET scores
- synthesis scores

Process:

- Convert each score to calibrated rank percentile.
- Apply weighted consensus.
- Add uncertainty penalty when models disagree.
- Keep backups from different scaffold clusters.

Default weights:

- JAK2 binding: 30%
- selectivity: 20%
- geometry/strain: 15%
- ADMET/safety: 15%
- synthesizability: 15%
- novelty/diversity: 5%

Output:

- `final_ranked_candidates.csv`
- `decision_matrix.xlsx`

Success metric:

- final 3-5 candidates are not all from the same scaffold and have complete evidence.

#### M12 - TPP and Computational Dossier Generator

Input:

- final candidates
- poses
- routes
- score tables
- manifests

Output:

- `ElectroMacroDiff_V5.1_TPP.pdf`
- `ElectroMacroDiff_V5.1_Computational_Dossier.pdf`
- `final_candidates.sdf`
- `final_candidates.csv`
- `audit_manifest.zip`

TPP sections:

- executive summary
- target rationale
- molecule cards
- 2D structures
- 3D poses
- predicted JAK2 affinity
- selectivity panel
- SLC19A3 safety gate
- ADMET summary
- retrosynthesis route
- uncertainty and risks
- next wet-lab experiments

### 6. End-to-End Flow

#### Phase 0 - Campaign Initialization

Input:

- target: JAK2
- campaign config
- cloud bucket
- team account list

Actions:

- Create artifact hub.
- Freeze package versions.
- Create run registry.
- Assign nodes.

Output:

- ready-to-run cluster with identical environment configs.

#### Phase 1 - Data Acquisition and Curation

Input:

- ChEMBL 36
- BindingDB
- PubChem BioAssay
- RCSB PDB structures

Actions:

- Collect JAK2 and JAK-family data.
- Standardize potencies.
- Label macrocycles and constrained molecules.
- Build Elite 500, Elite 2K, selectivity, and decoy panels.

Output:

- curated training and validation tables.

#### Phase 2 - Static Feature Precomputation

Input:

- curated molecules
- selected structures

Actions:

- Generate ligand conformers.
- Compute electro-graph features.
- Encode JAK2 and off-target pockets.
- Push tensors to cloud.

Output:

- ligand and pocket tensors ready for training.

#### Phase 3 - Attachment Prior and Generator Training

Input:

- ligand/pocket tensors
- Elite 500/Elite 2K splits

Actions:

- Run frozen attachment predictor.
- Train LoRA adapters on selected generator.
- Track validity, loss, macrocycle closure, novelty.

Output:

- best adapter checkpoints.

#### Phase 4 - Candidate Generation

Input:

- adapter checkpoint
- JAK2 pocket condition
- cyclization priors

Actions:

- Generate 2,000-5,000 raw candidates.
- Repair structures.
- Remove invalid molecules.

Output:

- clean generated macrocycle set.

#### Phase 5 - Geometry, Diversity, and Novelty Filtering

Input:

- clean generated set

Actions:

- Remove strained/impossible structures.
- Cluster by scaffold and fingerprints.
- Enforce novelty/diversity.

Output:

- 100-300 diverse candidates.

#### Phase 6 - Binding Validation

Input:

- diverse candidates
- JAK2 structures

Actions:

- DiffDock/Boltz-style pose and affinity scoring.
- Consensus rank.
- Keep top 30-50.

Output:

- JAK2 binding-ranked candidates.

#### Phase 7 - Selectivity, Safety, and ADMET

Input:

- top 30-50 candidates
- off-target target panel

Actions:

- Score JAK1/JAK3/TYK2/FLT3/SLC19A3.
- Apply ADMET and medicinal chemistry filters.
- Keep 10-20.

Output:

- safety-ranked shortlist.

#### Phase 8 - Retrosynthesis

Input:

- 10-20 candidates

Actions:

- Run AiZynthFinder.
- Evaluate route feasibility.
- Reject impossible macrocyclizations.

Output:

- synthesis-qualified candidates.

#### Phase 9 - Final Ranking and Dossier

Input:

- all score tables and final structures

Actions:

- Weighted consensus ranking.
- Uncertainty analysis.
- Generate molecule cards and TPP.

Output:

- final 3-5 candidates and complete dossier.

### 7. Success Gates

| Gate | Minimum Pass |
|---|---|
| Data | at least 500 high-confidence JAK2/constrained examples or justified fallback |
| Featurization | more than 95% curated ligand success |
| Generation | at least 500 valid clean macrocycles |
| Novelty | at least 70% not near-duplicates of training data |
| Diversity | no dominant scaffold over 20% of shortlist |
| Binding | at least 30 candidates with plausible JAK2 poses |
| Selectivity | at least 10 candidates not strongly pan-JAK/FLT3 by model score |
| Safety | no final candidate with high SLC19A3 risk proxy |
| Synthesis | at least 3 route-plausible candidates |
| Dossier | all final molecules traceable through manifests |

### 8. Risk Register and Fallbacks

| Risk | Mitigation |
|---|---|
| Too few true macrocyclic JAK2 binders | Use constrained JAK2 ligands plus broader macrocycle pretraining |
| Colab disconnects | checkpoint every stage; write manifests continuously |
| Generator overfits Elite 500 | train with Elite 2K augmentation and novelty penalty |
| Invalid macrocycles | add structure repair, ring strain filters, and attachment priors |
| Docking false positives | consensus docking, multiple seeds, pose sanity checks |
| Boltz/DiffDock unavailable on Colab | fallback to lighter docking and reserve Boltz for top 50 |
| SLC19A3 structure uncertainty | treat as risk proxy, not proof; use ligand-similarity and model ensemble |
| No synthetic routes | expand building-block stock, relax ring size, regenerate from route-aware priors |

### 9. Day-by-Day Execution Plan

#### Day 0: Setup

- Create bucket/repo.
- Freeze requirements.
- Validate one end-to-end toy run with 5 molecules.

#### Day 1: Data and Features

- Node 1 curates ChEMBL/BindingDB/PubChem.
- Node 2 generates ligand tensors.
- Node 3 generates pocket tensors.
- Gate: Elite 500 + target panel ready.

#### Day 2: Attachment and Training Start

- Node 4 creates attachment priors.
- Nodes 5 and 6 train LoRA adapters with different seeds/settings.
- Gate: valid checkpoint and validation metrics.

#### Day 3: Training Completion and Generation

- Select best checkpoint.
- Node 7 generates 2,000-5,000 candidates.
- Node 8 performs structure repair and geometry filtering.
- Gate: at least 500 clean candidates.

#### Day 4: Binding Validation

- Node 9 performs JAK2 docking/scoring.
- Run consensus ranking.
- Gate: top 30-50 candidates.

#### Day 5: Selectivity, Safety, Synthesis

- Node 9/10 score off-targets and ADMET.
- Node 10 runs AiZynthFinder.
- Gate: 3-5 final candidates plus backups.

#### Day 6-7: Dossier Hardening

- Node 11 compiles TPP and computational dossier.
- Add uncertainty and next-experiment recommendations.
- Final audit package delivered.

### 10. Final Outputs

Required final files:

- `final_candidates.csv`
- `final_candidates.sdf`
- `final_ranked_candidates.csv`
- `synthesis_routes.json`
- `selectivity_safety_ranked.csv`
- `ElectroMacroDiff_V5.1_TPP.pdf`
- `ElectroMacroDiff_V5.1_Computational_Dossier.pdf`
- `audit_manifest.zip`

Each final candidate must include:

- canonical SMILES
- InChIKey
- 2D structure
- 3D conformer
- JAK2 pose
- predicted binding score
- selectivity panel
- SLC19A3 risk status
- ADMET flags
- synthetic route summary
- uncertainty level
- go/no-go recommendation

### 11. V5.1 Upgrade Summary

V5.0 was an aggressive 5-day proof-of-execution plan. V5.1 turns it into a controlled computational discovery system. The main change is not adding more complexity for its own sake; it is adding the missing guardrails: current data, reproducible artifacts, model consensus, selectivity, toxicity caution, synthesis feasibility, and uncertainty. The pipeline remains fast and Colab-compatible, but the outputs become credible enough to hand to a medicinal chemistry or wet-lab validation team as prioritized hypotheses.

