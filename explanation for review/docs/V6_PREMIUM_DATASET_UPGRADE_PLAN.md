# ElectroMacroDiff V6 Premium-Dataset Upgrade Plan

## Purpose

V5.5 proved the practical direction: anchor selection, linker choice, and
validity gating can be conditioned on JAK2 pocket/electronic context during
generation.

V6 upgrades the evidence source. Instead of a single 32-dim global pocket vector,
V6 prepares the project for learned protein-ligand and residue-level interaction
embeddings trained from richer structural datasets.

V5.5 remains frozen. V6 must write new `v6_*` artifacts only.

## Why This Is The Correct Next Stair

The missing project aim is not just better docking. The missing aim is:

- protein-guided generation
- electron/electronic context guiding generation decisions
- residue-level pocket fit before molecule construction, not only after docking

V6 attacks that gap by training on protein-ligand complex data:

- activity tables teach the reward model what active JAK/JAK2 chemistry looks like
- PDBbind teaches pose and affinity supervision
- PLINDER/BioLiP2 teach residue-level interaction context
- CrossDocked2020 is reserved for later scale-up after smaller lanes are stable

## Dataset Priority

| Priority | Dataset | Use In V6 | Why Now |
|---:|---|---|---|
| 1 | ChEMBL + BindingDB JAK/JAK2 | Activity/reward labels | Small, fast, directly target-specific |
| 2 | PDBbind v2020 refined/general | Pose and affinity encoder | Higher quality protein-ligand complexes |
| 3 | PLINDER | Learned pocket interaction encoder | Large annotated PLI resource with split/evaluation support |
| 4 | BioLiP2 | Biologically relevant contact labels | Adds binding-site and functional relevance evidence |
| 5 | CrossDocked2020 | Large-scale generator distillation | Big and noisy, use only after the smaller lanes work |

## Four Sequential Kaggle Roles

### Account 1: Activity Reward Lane

Goal:

- build a JAK/JAK2 activity reward model using ChEMBL + BindingDB tables
- improve candidate prioritization beyond validity-only gating

Expected output:

- `04_models_checkpoints/v6_activity_reward/v6_activity_reward_best.pt`
- `03_features/v6_activity_reward_training.csv`
- `04_models_checkpoints/v6_activity_reward/v6_activity_reward_summary.json`

Smoke command:

```bash
!python scripts/33_train_v6_activity_reward.py \
  --base . \
  --device auto \
  --smoke-test \
  --epochs 2 \
  --max-rows 300
```

Long command:

```bash
!python scripts/33_train_v6_activity_reward.py \
  --base . \
  --device auto \
  --epochs 800 \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --patience 120
```

Acceptance:

- report ROC-AUC/PR-AUC
- molecule-level split
- no leakage by duplicate canonical SMILES or InChIKey

### Account 2: PDBbind Pose/Affinity Lane

Goal:

- train a protein-ligand pose/affinity encoder on high-quality complexes
- learn interaction features richer than V5.5's fixed pocket-count vector

Expected output:

- `04_models_checkpoints/v6_pdbbind_pose_affinity/v6_pose_affinity_encoder_best.pt`
- `03_features/v6_pdbbind_complex_manifest.csv`
- `04_models_checkpoints/v6_pdbbind_pose_affinity/v6_pose_affinity_summary.json`

Acceptance:

- validation Pearson/Spearman for affinity
- pose-quality classification metric if decoys are available
- protein-family or cluster-aware split where possible

### Account 3: PLINDER/BioLiP2 Contact Lane

Goal:

- train residue-contact and interaction-type prediction
- convert pocket context from a global vector into residue-level learned evidence

Expected output:

- `04_models_checkpoints/v6_residue_contact_encoder/v6_residue_contact_encoder_best.pt`
- `03_features/v6_residue_contact_training_manifest.csv`
- `04_models_checkpoints/v6_residue_contact_encoder/v6_residue_contact_summary.json`

Acceptance:

- residue-contact AUC/F1
- interaction-type metrics if labels are present
- clear train/validation/test split policy

### Account 4: V6 Generator Distillation Lane

Goal:

- distill the learned protein interaction encoder into anchor/linker/generation policies
- generate a new logged V6 candidate branch

Expected output:

- `05_generated_candidates/v6_pocket_interaction_guided/generated_v6_pocket_interaction_guided.csv`
- `05_generated_candidates/v6_pocket_interaction_guided/v6_pocket_interaction_guided_attempt_log.csv`
- `09_reports/v6_benchmark/EMD_V6_Benchmark_Report.md`

Acceptance:

- raw-attempt validity is logged
- linker novelty is measured
- docking and pocket-electronic fit are measured
- comparison is against frozen V5.5 and MED reference metrics

## First Local Command

Run this before any V6 notebook:

```powershell
python scripts/31_prepare_v6_premium_manifest.py --base .
python scripts/32_write_v6_training_plan.py --base .
```

This writes:

- `03_features/v6_premium_dataset_manifest.csv`
- `03_features/v6_premium_dataset_summary.json`
- `docs/V6_KAGGLE_SEQUENTIAL_TRAINING_PLAN.md`

## Kaggle Rule

Do not download giant datasets inside the notebook. Add each dataset as a Kaggle
dataset or mount path, then pass the mounted path:

```bash
!python scripts/31_prepare_v6_premium_manifest.py \
  --base . \
  --activity-root /kaggle/input/YOUR_JAK_ACTIVITY_DATASET \
  --pdbbind-root /kaggle/input/YOUR_PDBBIND_DATASET \
  --plinder-root /kaggle/input/YOUR_PLINDER_DATASET \
  --biolip2-root /kaggle/input/YOUR_BIOLIP2_DATASET \
  --crossdocked-root /kaggle/input/YOUR_CROSSDOCKED_DATASET
```

Use smoke mode before every long run.

## Claim Boundary

Allowed after V6 preparation:

- ElectroMacroDiff is prepared for premium-data protein-ligand training.
- The V6 data plan targets learned residue-level protein conditioning.

Allowed only after V6 model training succeeds:

- ElectroMacroDiff V6 uses learned protein-ligand interaction embeddings during generation.

Not allowed:

- V6 has beaten MED before the V6 logged benchmark exists.
- V6 performs quantum electron-density generation.
- V6 proves experimental biological activity.

## Scientific References To Cite In V6 Docs

- PLINDER: https://github.com/plinder-org/plinder
- PLINDER dataset documentation: https://plinder-org.github.io/plinder/dataset.html
- BioLiP2 paper: https://academic.oup.com/nar/article/52/D1/D404/7233921
- PDBbind v2020 usage: http://www.pdbbind.org.cn/
- CrossDocked2020 generative-model context: https://pubs.rsc.org/en/content/articlehtml/2022/sc/d1sc05976a
