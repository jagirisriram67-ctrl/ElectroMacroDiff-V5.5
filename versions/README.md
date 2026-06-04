# ElectroMacroDiff Version History Archive

This folder is a reviewer-facing history of how the project developed. The repository root remains the final V5.5 implementation. The folders here preserve earlier code, planning documents, molecules, metrics, docking outputs, model checkpoints, and benchmark evidence so the team can explain where the work started and how it matured.

## Folder Map

| Folder | Meaning | Main evidence |
|---|---|---|
| `v5_0_v5_1_planning/` | Early problem framing and upgrade planning | V5.0 technical report, V5.0 upgrade plan, V5.1 implementation plan |
| `v5_2_initial_hybrid/` | First reproducible hybrid pipeline | Original V5.2 code, notebooks, tests, SE(3) core addendum, master execution blueprint |
| `v5_3_model_guided_generation/` | Model-guided macrocycle generation and Vina-GPU evidence | V5.3 anchor/linker data, checkpoints, generated candidates, docking outputs, ADMET, ranking, benchmark reports |
| `v5_4_pocket_electronic_scoring/` | Pocket-electronic scoring layer added over generated/docked molecules | JAK2 pocket profile, pocket-electronic fit scores, ranking table, scoring script |

## Final Version Context

V5.5 is not duplicated here because the root repository is V5.5. The final implementation adds pocket-conditioned anchor selection, pocket-conditioned linker policy, validity gating, V5.5 generation outputs, V5.5 docking scores, reviewer documents, and the interactive dashboard.

## Large Artifact Handling

One V5.3 training table, `v5_3_pretrain_anchor_atom_training.csv`, is too large to keep as a normal uncompressed GitHub file. It is preserved as:

```text
versions/v5_3_model_guided_generation/03_features/v5_3_pretrain_anchor_atom_training.csv.gz
versions/v5_3_model_guided_generation/03_features/v5_3_pretrain_anchor_atom_training.manifest.json
```

The manifest records the original size and SHA-256 hash so the archived data can be verified after decompression.

## Claim Boundary

All artifacts in this repository are computational research outputs. They document code, generated molecules, docking results, model-training evidence, and ranking logic. They are not wet-lab validation data and should not be described as experimentally confirmed activity, toxicity, synthesis, or clinical evidence.
