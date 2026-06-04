# ElectroMacroDiff V5.2 Hybrid Addendum

## Keep the Student Plan, Restore the Deep Learning Core

### 1. Final Decision

V5.2 is still the correct execution plan for a 5-student team with 8-10 days and free Google Colab. But M3 must not be only RDKit/SELFIES generation. That would make the project look like a tool-chain integration project instead of a novel AI architecture project.

The corrected strategy is:

> V5.2 remains the full reproducible pipeline, but M3 becomes a hybrid generation module: the custom SE(3) Flow Matching model is the primary research generator, while RDKit/SELFIES is the baseline, backup, and comparison generator.

This gives two defensible stories:

1. Engineering story: low-resource, checkpointed, Google Drive-resumable drug discovery pipeline.
2. Research story: custom SE(3) flow matching architecture with electronic/3D molecular features for macrocycle-aware generation.

### 2. Corrected M3 Module

#### M3 - Hybrid Candidate Generation

Input:

- curated JAK2 ligand table
- ligand conformers
- atom/bond/electronic features
- optional pocket features
- train/validation split

Primary generator:

- custom SE(3) Flow Matching model already written by the team
- target size: approximately 1-2M parameters if that is the current implementation
- trained on Google Colab T4 for a short sprint run

Backup and baseline generator:

- RDKit scaffold/substituent modification
- SELFIES mutation
- simple fragment recombination

Output:

- `generated_se3_raw.csv`
- `generated_se3_raw.sdf`
- `generated_selfies_baseline.csv`
- `generated_rdkit_baseline.csv`
- `generation_comparison_metrics.csv`
- `best_se3_checkpoint.pt`
- `latest_se3_checkpoint.pt`

Minimum success:

- train the SE(3) model for at least one complete run
- save checkpoints and loss curves
- generate at least 100 valid molecules or conformers from the SE(3) model
- compare against RDKit/SELFIES baseline

Ideal success:

- 100-200 epochs on Tiny/Main Sprint set
- 500+ generated candidates
- validity, novelty, uniqueness, ring-size, and descriptor distributions reported

### 3. What the Project readers Should See

Project reader question:

> Where is the deep learning?

Answer:

> The central generation step uses our custom SE(3)-aware flow matching model trained on curated JAK2 macrocycle/constrained ligand data. RDKit and SELFIES are used as controlled baselines and fallback generators. We compare validity, novelty, chemical property distributions, and downstream docking performance between the deep model and rule/string baselines.

Project reader question:

> What is novel?

Answer:

> The novelty is the low-resource adaptation of an SE(3) flow-matching generator with electronic/3D molecular features inside a fully checkpointed student-scale discovery pipeline. The pipeline does not merely screen existing molecules; it generates candidates, ranks them, and produces a TPP-style dossier under severe compute constraints.

Project reader question:

> Did the model actually help?

Answer:

> We include a baseline comparison. The SE(3) generator is evaluated against RDKit/SELFIES candidates using validity, uniqueness, novelty, macrocycle/constrained-geometry retention, docking score distribution, and final shortlist representation.

### 4. Updated 10-Day Schedule

#### Day 1 - Setup and Tiny Pipeline

- Create Drive structure.
- Create registry and checkpoint code.
- Run 20-molecule tiny test through RDKit featurization.
- Confirm model code imports in Colab.

Deliverable:

- environment working
- model forward pass works on tiny batch

#### Day 2 - Data Collection and Curation

- Build JAK2 ligand dataset from ChEMBL/BindingDB.
- Download JAK2 PDB structure.
- Label macrocycle/constrained molecules.

Deliverable:

- `jak2_curated_ligands.csv`
- `jak2_train_val_split.csv`

#### Day 3 - DataLoader and Feature Fix Day

This is the make-or-break day for the custom model.

Tasks:

- fix DataLoader
- confirm tensor shapes
- run one batch through the model
- run one loss computation
- run one optimizer step
- save and reload checkpoint

Deliverable:

- `se3_debug_batch.pt`
- `se3_dataloader_test_passed.json`
- `latest_se3_checkpoint.pt`

Go/no-go rule:

- If one training step works by end of Day 3, continue SE(3) training.
- If not, freeze SE(3) as architecture demonstration and use RDKit/SELFIES for main candidates.

#### Day 4 - SE(3) Training + Baseline Generation

Primary:

- train SE(3) model for 100-200 epochs or as many as Colab allows.
- save every 10-20 epochs.
- save best validation checkpoint.

Parallel human work:

- another member generates RDKit/SELFIES candidates.

Deliverable:

- `best_se3_checkpoint.pt`
- `training_curve.png`
- `generated_selfies_baseline.csv`

#### Day 5 - SE(3) Inference and Candidate Merge

- generate molecules/conformers from best checkpoint.
- sanitize generated candidates.
- merge SE(3), SELFIES, and RDKit outputs.
- remove duplicates.
- compute generation metrics.

Deliverable:

- `generated_se3_filtered.csv`
- `generation_comparison_metrics.csv`
- `merged_candidates_for_docking.csv`

#### Day 6 - Docking Setup and Control Docking

- prepare JAK2 structure.
- dock known ligands as control.
- dock first candidate batch.

Deliverable:

- docking method validated.

#### Day 7 - Main Docking

- dock 50-150 candidates.
- ensure at least some SE(3)-generated candidates are included.
- save poses and scores.

Deliverable:

- `docking_scores.csv`
- `top_pose_images/`

#### Day 8 - ADMET, Synthesizability, and Safety Proxy

- QED, SA score, Lipinski/Veber.
- PAINS/Brenk if available.
- SLC19A3 proxy notes.
- AiZynthFinder only if setup is smooth.

Deliverable:

- `admet_synthesis_scores.csv`

#### Day 9 - Final Ranking and Report

- compare SE(3) vs baseline performance.
- rank final candidates.
- build molecule cards.

Deliverable:

- `final_ranked_candidates.csv`
- draft report.

#### Day 10 - Reproducibility and Presentation

- rerun tiny pipeline.
- verify Drive paths.
- polish TPP and slides.

Deliverable:

- final package.

### 5. SE(3) Training Checklist

Before long training:

- model imports cleanly
- dataset loads from Drive
- batch has expected tensor shapes
- forward pass works
- loss is finite
- backward pass works
- optimizer step works
- checkpoint save works
- checkpoint reload works
- generation/inference function works on a tiny sample

If any item fails, fix it before training.

### 6. Required Checkpoint Format

Save:

```python
checkpoint = {
    "epoch": epoch,
    "global_step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "train_loss": train_loss,
    "val_loss": val_loss,
    "config": config,
    "random_seed": seed,
}
```

Paths:

```text
04_models_checkpoints/
  se3_latest_checkpoint.pt
  se3_best_checkpoint.pt
  se3_epoch_0010.pt
  se3_epoch_0020.pt
  training_log.csv
  training_curve.png
```

### 7. Metrics That Make the SE(3) Story Strong

Report these metrics for SE(3), SELFIES, and RDKit:

- number generated
- RDKit validity
- uniqueness
- novelty versus training set
- macrocycle/constrained ring retention
- molecular weight distribution
- logP distribution
- TPSA distribution
- QED distribution
- SA score distribution
- docking score distribution
- number appearing in final top 10

If the SE(3) model does not outperform every baseline, that is still acceptable. The honest result can be:

> The SE(3) generator produced chemically valid novel candidates under extreme compute limits, while rule-based baselines provided broader fallback diversity. The hybrid system improved robustness.

### 8. Final Claim

Use this wording:

> ElectroMacroDiff V5.2 Hybrid is a low-resource AI drug discovery pipeline centered on a custom SE(3) flow-matching molecular generator, supported by RDKit/SELFIES baselines, docking-based prioritization, ADMET filtering, and synthesizability assessment. The system is designed for resumable execution on free Google Colab using Google Drive checkpoints.

Avoid this wording:

> We discovered clinically safe JAK2 drugs.

Avoid this too:

> RDKit generated all molecules.

The right story is balanced:

> We built and tested a custom deep generative model inside a practical, reproducible screening pipeline, with classical methods as baselines and safety nets.

