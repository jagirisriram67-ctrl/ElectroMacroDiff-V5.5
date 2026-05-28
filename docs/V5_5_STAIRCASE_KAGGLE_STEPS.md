# V5.5 Staircase Kaggle Steps

This is the sequential, one-model-at-a-time path for proving V5.5 before any
large premium-dataset V6 work.

## Locked Strategy

- Do not run the four model trainings in parallel.
- Use the four Kaggle accounts as clean role lanes.
- Run a smoke test before every long training job.
- Keep all V5.3 frozen outputs untouched.
- Treat TPU as experimental. Use `--device cuda` on T4/P100 for the current scripts.

## Account 0 Or Any Account: Dataset Manifest Smoke

```bash
python scripts/24_build_v5_5_training_data.py --base . --smoke-test
```

Optional full manifest:

```bash
python scripts/24_build_v5_5_training_data.py --base .
```

This avoids duplicating the huge anchor CSV by default. Trainers load the pocket
vector directly.

## Account 1: Pocket Anchor GNN

Smoke:

```bash
python scripts/25_train_pocket_anchor_gnn.py --base . --device cuda --smoke-test
```

Long JAK2 fine-tune run:

```bash
python scripts/25_train_pocket_anchor_gnn.py \
  --base . \
  --device cuda \
  --epochs 1500 \
  --batch-size 16 \
  --hidden-dim 128 \
  --num-layers 4 \
  --patience 200
```

Target output:

```text
04_models_checkpoints/v5_5_pocket_anchor_gnn/pocket_anchor_gnn_best.pt
```

Acceptance target: better than V5.3 anchor F1 `0.209`; preferred F1 `> 0.55`.

## Account 2: Pocket Linker Policy

Smoke:

```bash
python scripts/26_train_pocket_linker_policy.py --base . --device cuda --smoke-test
```

Long run:

```bash
python scripts/26_train_pocket_linker_policy.py \
  --base . \
  --device cuda \
  --epochs 2000 \
  --batch-size 64 \
  --hidden-dim 192 \
  --patience 250
```

Target output:

```text
04_models_checkpoints/v5_5_pocket_linker_policy/pocket_linker_policy_best.pt
```

Acceptance target: exact linker accuracy `> 50%`, within-one accuracy `> 75%`.

## Account 3: Validity Reward Model

Smoke:

```bash
python scripts/27_train_validity_reward.py --base . --device cuda --smoke-test
```

Long run:

```bash
python scripts/27_train_validity_reward.py \
  --base . \
  --device cuda \
  --epochs 2000 \
  --batch-size 128 \
  --hidden-dim 128 \
  --patience 300
```

Target output:

```text
04_models_checkpoints/v5_5_validity_reward/validity_reward_best.pt
```

Acceptance target: ROC-AUC `>= 0.75`.

## Account 4: SE(3) Continuation

Smoke:

```bash
python scripts/28_train_se3_continuation.py --base . --device cuda --smoke-test
```

Long run:

```bash
python scripts/28_train_se3_continuation.py \
  --base . \
  --device cuda \
  --epochs 500 \
  --batch-size 4 \
  --learning-rate 0.0001 \
  --patience 100
```

Target output:

```text
04_models_checkpoints/v5_5_se3_continuation/se3_v5_5_best_checkpoint.pt
```

SE(3) remains auxiliary geometry evidence unless a later generator directly uses
its coordinate outputs.

## Final V5.5 Generation

Run only after the accepted checkpoints are present:

```bash
python scripts/29_generate_v5_5_pocket_guided.py \
  --base . \
  --device cuda \
  --seed-count 250 \
  --max-products-per-seed 20
```

The script fails fast if any required V5.5 checkpoint is missing. To test the
script with smoke checkpoints only, pass explicit checkpoint paths. Do not use
fallback mode for final benchmark runs.

Target outputs:

```text
05_generated_candidates/v5_5_pocket_guided/generated_v5_5_pocket_guided.csv
05_generated_candidates/v5_5_pocket_guided/v5_5_pocket_guided_attempt_log.csv
05_generated_candidates/v5_5_pocket_guided/v5_5_pocket_guided_summary.json
```

## V6 Premium Dataset Roadmap

Start only after V5.5 has real logged metrics.

- BindingDB + ChEMBL JAK/JAK2 labels for stronger activity/reward learning.
- PDBbind refined/core for affinity and pose-quality supervision.
- PLINDER or BioLiP2 for residue-level contact learning.
- CrossDocked2020 later, because it is large and data-engineering-heavy.

V6 goal: replace the simple 32-dim global JAK2 pocket vector with a learned
residue-level pocket interaction encoder.

## Claim Boundary

Allowed after V5.5:

> ElectroMacroDiff V5.5 conditions anchor selection, linker choice, and validity
> gating on JAK2 pocket-electronic context during generation.

Reserved for V6:

> ElectroMacroDiff uses learned residue-level protein-ligand interaction
> embeddings for protein-conditioned macrocycle generation.
