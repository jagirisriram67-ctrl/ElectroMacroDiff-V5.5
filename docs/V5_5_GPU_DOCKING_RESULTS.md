# V5.5 GPU Docking Results

## Status

The Kaggle GPU-only Vina-GPU 2.1 output was merged and parsed successfully.

This branch is `v5_5_pocket_guided`. The frozen V5.3 baseline branch remains
unchanged.

## Docking Summary

| Metric | Value |
|---|---:|
| Clean ligands sent to Vina-GPU | 1036 |
| Raw pose files returned | 1036 |
| Parsed Vina-GPU scores | 1036 |
| Best Vina-GPU score | -12.7 kcal/mol |
| Median Vina-GPU score | -8.4 kcal/mol |
| Pocket-electronic scored subset | top 200 |
| Best pocket-electronic fit score | 0.848380 |
| Median pocket-electronic fit score | 0.783945 |

The all-ligand Vina-GPU run was a fast screen using the Kaggle GPU. Pocket
electronic scoring was run on the top-ranked 200 candidates for final decision
ranking.

## Current Top Candidate

| Field | Value |
|---|---|
| Candidate | `CAND_3688f8acdc` |
| Vina-GPU score | -12.7 kcal/mol |
| Pocket-electronic fit | 0.810776 |
| Pocket-guided final score | 0.874199 |
| Decision | primary candidate |

## Key Outputs

- `06_docking/v5_5_pocket_guided/scores/docking_scores_full_vina_gpu_2_1.csv`
- `06_docking/v5_5_pocket_guided/scores/v5_5_vina_gpu_parse_summary.json`
- `06_docking/v5_5_pocket_guided/scores/pocket_electronic_fit_scores.csv`
- `06_docking/v5_5_pocket_guided/scores/v5_5_docking_and_pocket_summary.json`
- `08_final_ranking/v5_5_pocket_guided_ranked_candidates.csv`
- `08_final_ranking/v5_5_pocket_guided_pocket_electronic_ranked_candidates.csv`
- `interface/data.js`

## Claim Boundary

These results support the V5.5 claim:

> ElectroMacroDiff V5.5 conditions anchor selection, linker choice, and validity
> gating on JAK2 pocket-electronic context during generation, then validates and
> prioritizes candidates through Vina-GPU docking and pocket-electronic rescoring.

They do not prove experimental activity, synthesis success, clinical safety, or
full protein-conditioned diffusion.
