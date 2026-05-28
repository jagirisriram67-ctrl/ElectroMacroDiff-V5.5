# EMD V5.3 Benchmark Report

This report benchmarks ElectroMacroDiff V5.3 against MED-style macrocycle generation metrics using local artifacts plus sidecar Kaggle rerun outputs. The frozen 114-candidate branch remains the project reference branch, while Kaggle reruns are treated as evidence-upgrade branches rather than replacements.

## Current Decision

ElectroMacroDiff is currently best described as a low-resource macrocycle generation plus pocket-electronic prioritization system for JAK2. It is not yet a fully protein-conditioned generator. Claims against MED should remain conservative until a logged rerun closes the raw-attempt validity gap and linker novelty is re-measured on the new branch.

## Published Reference Metrics

| Method | Validity | Uniqueness | Macrocyclization | Linker novelty |
| --- | --- | --- | --- | --- |
| MED | 93.82% | 99.94% | 99.92% | 82.81% |
| Macformer | 72.91% | 47.74% | 96.39% | 44.24% |
| MacLS | 89.67% | 95.04% | 100.00% | 0.00% |

Source: Macro-Equi-Diff PDF Table 1, extracted into `09_reports/med_comparison_assets/macro_equidiff_extracted_text.txt` and summarized by `scripts/12_build_med_vs_emd_doc.py`.

## EMD Candidate Metrics

| Branch | Scope | Candidates | Validity | Uniqueness | Macrocycle | Novel molecules | Median QED | Median SA proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selfies | filtered_survivor | 2590 | 100.00% | 100.00% | 11.35% | 97.61% | 0.4747 | 3.436 |
| rdkit | filtered_survivor | 1250 | 100.00% | 100.00% | 0.80% | 98.88% | 0.4817 | 3.403 |
| macrocycle_linker | filtered_survivor | 2935 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3898 | 4.975 |
| frozen_v5_3_baseline | filtered_survivor | 114 | 100.00% | 100.00% | 100.00% | 100.00% | 0.4139 | 4.89 |
| kaggle_logged_rerun | raw_attempt_logged | 114 | 18.12% | 100.00% | 100.00% | 100.00% | 0.414 | 4.906 |
| kaggle_diverse_rerun | raw_attempt_logged | 68 | 2.53% | 100.00% | 100.00% | 100.00% | 0.369 | 5.075 |
| kaggle_v5_5_pocket_guided | raw_attempt_logged | 1113 | 1.28% | 100.00% | 100.00% | 100.00% | 0.3913 | 4.77 |
| merged_with_v5_3 | filtered_survivor | 6888 | 100.00% | 100.00% | 48.68% | 98.91% | 0.4277 | 4.125 |

## MED Comparison Status

| Metric | kaggle_v5_5_pocket_guided | MED | Status |
| --- | --- | --- | --- |
| Validity | 1.28% | 93.82% | does_not_exceed |
| Uniqueness | 100.00% | 99.94% | exceeds |
| Macrocyclization | 100.00% | 99.92% | exceeds |
| Linker novelty | 100.00% | 82.81% | exceeds |

All MED-style metrics are comparable for this row.

## Docking Benchmark

| Branch | Scored | Coverage | Best Vina | Median Vina | Mean Vina |
| --- | --- | --- | --- | --- | --- |
| v5_2_macrocycle_campaign | 148 / 148 | 100.00% | -9.186 | -6.453 | -6.238 |
| kaggle_v5_5_pocket_guided | 1036 / 1036 | 100.00% | -12.7 | -8.4 | -8.372 |
| v5_3_model_guided_vina_gpu_2_1 | 114 / 114 | 100.00% | -11.4 | -8.3 | -8.697 |

Docking scores are AutoDock Vina proxy scores against JAK2 PDB `5AEP`; more negative is better. They are not experimental affinity.

## Pose Sanity

| Branch | Passed | Pass rate |
| --- | --- | --- |
| v5_2_macrocycle_campaign | 10 / 10 | 100.00% |
| kaggle_v5_5_pocket_guided | 20 / 20 | 100.00% |
| v5_3_model_guided_vina_gpu_2_1 | 10 / 10 | 100.00% |

## Free-Cost Path To A Defensible Benchmark Claim

1. Run the sidecar benchmark branches on Kaggle with GPU when available; hardware and quota can vary by account and week.
2. Use the raw attempt log from `scripts/17_generate_v5_3_model_guided_macrocycles.py` for each rerun branch.
3. Use `scripts/20_measure_v5_3_benchmark_gaps.py` after each rerun to measure raw-attempt validity and linker novelty.
4. Run the same fixed benchmark split three times with fixed seeds and report mean plus standard deviation.
5. Only claim MED is beaten after EMD exceeds MED on validity, uniqueness, macrocyclization, and linker novelty using the same raw-attempt style.

## Current Target Bar

- Validity target: above `93.82%`.
- Uniqueness target: above `99.94%`.
- Macrocyclization target: above `99.92%`.
- Linker novelty target: above `82.81%`.
- JAK2 docking target: improve the branch best/median Vina scores while preserving pose sanity and ADMET/synthesis filters.
