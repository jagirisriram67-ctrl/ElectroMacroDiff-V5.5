# V5.5 Architecture: Pocket-Electronic-Conditioned Generation

## What Changed From V5.3 to V5.5

V5.3 generated molecules **blind to the protein pocket**. The pocket's electronic
properties only entered after generation, as a post-hoc scoring filter.

V5.5 introduces **three pocket-conditioned neural models** that see the JAK2
pocket's electronic profile **before and during generation**:

```
V5.3 Flow:  Seed → Anchor Model → Linker Model → Build → Dock → Score Pocket
                     (ligand only)   (ligand only)

V5.5 Flow:  Pocket Profile ──┐
                              ├→ PocketAnchorGNN ──→ Anchor Selection
                              ├→ PocketLinkerPolicy → Size + Chemotype
                              ├→ ValidityReward ───→ Pre-filter
            Seed ─────────────┘
                                        ↓
                                   Build Candidate → Dock → Score → Rank
```

## New Models

### PocketAnchorGNN (Account 1)
- **Input:** Molecular graph (64-dim node features) + pocket electronic vector (32-dim)
- **Output:** Per-atom anchor probability
- **Innovation:** The pocket vector is concatenated to every graph node before
  message passing. Anchor predictions now consider whether the attachment site
  faces a complementary region of the JAK2 pocket.

### PocketLinkerPolicy (Account 2)
- **Input:** Linker features (16-dim) + pocket electronic vector (32-dim)
- **Output:** Linker length distribution (10 classes) + chemotype distribution (11 classes)
- **Innovation:** Dual-head prediction. The pocket's hydrophobic/polar/aromatic
  composition influences which linker chemistry is proposed.

### PocketValidityRewardModel (Account 3)
- **Input:** Attempt features (19-dim) + pocket electronic vector (32-dim)
- **Output:** P(valid macrocycle)
- **Innovation:** Trained on 2,688 real attempt logs. Rejects likely-invalid
  generation proposals before RDKit construction, solving the 2.5% validity
  collapse observed in diversity-enforced generation.

## Pocket Feature Vector (32 dimensions)

| Dims | Content |
|------|---------|
| 0-5 | Normalized counts: atoms, hydrophobic, donor, acceptor, aromatic, charge |
| 6-10 | Fractional composition |
| 11 | Donor/acceptor ratio |
| 12-31 | Amino acid contact fingerprint (20 standard amino acids) |

## Honest Claims

### Allowed
- V5.5 conditions generation decisions on the JAK2 pocket electronic profile.
- Anchor selection, linker policy, and validity gating all receive pocket features.
- This is a practical pocket-electronic guidance layer, not a generative diffusion model.

### Not Allowed
- V5.5 does NOT perform true protein-conditioned diffusion.
- V5.5 does NOT use quantum electron density calculations.
- V5.5 does NOT modify the linker at the sub-atomic or bond-order level based on
  specific residue contacts.
- The pocket profile is global (same for all seeds), not seed-pose-specific.

## Files

### Source Modules
- `src/emd_v5_2_hybrid/device_helper.py`
- `src/emd_v5_2_hybrid/pocket_features.py`
- `src/emd_v5_2_hybrid/pocket_anchor_gnn.py`
- `src/emd_v5_2_hybrid/pocket_linker_policy.py`
- `src/emd_v5_2_hybrid/validity_reward.py`

### Training Scripts
- `scripts/24_build_v5_5_training_data.py`
- `scripts/25_train_pocket_anchor_gnn.py`
- `scripts/26_train_pocket_linker_policy.py`
- `scripts/27_train_validity_reward.py`
- `scripts/28_train_se3_continuation.py`

### Generation
- `scripts/29_generate_v5_5_pocket_guided.py`

### Tests
- `tests/test_v5_5_pocket_models.py` (18 tests)

### Sequential Kaggle Execution
- `docs/V5_5_STAIRCASE_KAGGLE_STEPS.md`

V5.5 is now run as a sequential staircase: anchor model first, linker policy
second, validity reward third, SE(3) continuation and final generation fourth.
This keeps the checkpoint lineage clean and avoids collision between accounts.
