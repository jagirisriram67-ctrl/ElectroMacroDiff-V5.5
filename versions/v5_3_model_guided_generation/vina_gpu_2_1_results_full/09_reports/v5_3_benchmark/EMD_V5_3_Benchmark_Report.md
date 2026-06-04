# EMD V5.3 Benchmark Report

This report benchmarks the current ElectroMacroDiff V5.3 branch against MED-style macrocycle generation metrics using only local project artifacts and free tooling. The current V5.3 candidate CSVs are filtered survivor tables, so they are strong project metrics but are not yet paper-equivalent raw generation metrics.

## Current Decision

EMD V5.3 currently meets or exceeds MED on filtered survivor validity, uniqueness, and macrocyclization for the model-guided macrocycle branch. It must not yet claim to beat MED on the original paper benchmark because raw generation attempts and true linker novelty are not recorded in the same way.

## Published Reference Metrics

| Method | Validity | Uniqueness | Macrocyclization | Linker novelty |
| --- | --- | --- | --- | --- |
| MED | 93.82% | 99.94% | 99.92% | 82.81% |
| Macformer | 72.91% | 47.74% | 96.39% | 44.24% |
| MacLS | 89.67% | 95.04% | 100.00% | 0.00% |

Source: Macro-Equi-Diff PDF Table 1, extracted into `09_reports/med_comparison_assets/macro_equidiff_extracted_text.txt` and summarized by `scripts/12_build_med_vs_emd_doc.py`.

## EMD Candidate Metrics

| Branch | Candidates | Validity | Uniqueness | Macrocycle | Novel molecules | Median QED | Median SA proxy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| selfies | 2590 | 100.00% | 100.00% | 11.35% | 97.61% | 0.4747 | 3.436 |
| rdkit | 1250 | 100.00% | 100.00% | 0.80% | 98.88% | 0.4817 | 3.403 |
| macrocycle_linker | 2935 | 100.00% | 100.00% | 100.00% | 100.00% | 0.3898 | 4.975 |
| v5_3_model_guided_macrocycle | 114 | 100.00% | 100.00% | 100.00% | 100.00% | 0.4139 | 4.89 |
| merged_with_v5_3 | 6888 | 100.00% | 100.00% | 48.68% | 98.91% | 0.4277 | 4.125 |

## MED Comparison Status

| Metric | EMD V5.3 | MED | Status |
| --- | --- | --- | --- |
| Validity | 100.00% | 93.82% | exceeds |
| Uniqueness | 100.00% | 99.94% | exceeds |
| Macrocyclization | 100.00% | 99.92% | exceeds |
| Linker novelty | NA | 82.81% | not_comparable |

Linker novelty and raw-attempt validity are unavailable for the current filtered survivor CSVs; do not claim MED is beaten on the original paper benchmark yet.

## Docking Benchmark

| Branch | Scored | Coverage | Best Vina | Median Vina | Mean Vina |
| --- | --- | --- | --- | --- | --- |
| v5_2_macrocycle_campaign | 148 / 148 | 100.00% | -9.186 | -6.453 | -6.238 |
| v5_3_model_guided_vina_gpu_2_1 | 114 / 114 | 100.00% | -11.4 | -8.3 | -8.697 |

Docking scores are AutoDock Vina proxy scores against JAK2 PDB `5AEP`; more negative is better. They are not experimental affinity.

## Pose Sanity

| Branch | Passed | Pass rate |
| --- | --- | --- |
| v5_2_macrocycle_campaign | 10 / 10 | 100.00% |
| v5_3_model_guided | 10 / 10 | 100.00% |

## Free-Cost Path To A Defensible Benchmark Claim

1. Keep using Google Colab free T4 or local CPU/GPU for training and Vina.
2. Add raw generation attempt logs for every generated molecule, including invalid attempts.
3. Add true linker extraction for generated macrocycles so linker novelty can be compared to training linkers.
4. Run the same fixed benchmark split three times with fixed seeds and report mean plus standard deviation.
5. Only claim MED is beaten after EMD exceeds MED on validity, uniqueness, macrocyclization, and linker novelty using the same raw-attempt style.

## Current Target Bar

- Validity target: above `93.82%`.
- Uniqueness target: above `99.94%`.
- Macrocyclization target: above `99.92%`.
- Linker novelty target: above `82.81%`.
- JAK2 docking target: improve the branch best/median Vina scores while preserving pose sanity and ADMET/synthesis filters.
