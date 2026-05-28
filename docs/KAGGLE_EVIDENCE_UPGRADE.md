# Kaggle Evidence Upgrade

This phase upgrades the benchmark evidence without changing the frozen `114`-candidate V5.3 reference branch.

## What This Phase Adds

- Logged Kaggle rerun support for raw-attempt validity.
- Diverse chain-compatible linker chemotypes.
- Exact-linker novelty rejection during generation.
- Auxiliary SE(3) geometry scoring.
- Benchmark/report/dashboard support for sidecar rerun branches.

## Branch Naming

- `frozen_v5_3_baseline`
  The original V5.3 branch with `generated_v5_3_model_guided_macrocycles.csv`.
- `kaggle_logged_rerun`
  Logged sidecar rerun for raw-attempt validity measurement.
- `kaggle_diverse_rerun`
  Novelty-enforced diverse-chemotype sidecar rerun.

## Important Claim Boundary

ElectroMacroDiff is currently:

- a low-resource macrocycle generation system
- plus docking
- plus pocket-electronic rescoring
- plus auxiliary SE(3) geometry evidence

It is **not yet** a fully protein-conditioned diffusion generator. Pocket/electronic information still enters as a post-generation scoring layer rather than a direct generation-time conditioning signal.

## Kaggle Execution

Use:

- [08_kaggle_evidence_upgrade.ipynb](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/10_notebooks/08_kaggle_evidence_upgrade.ipynb)

The notebook:

1. unpacks the project zip
2. runs the logged rerun
3. runs the diverse rerun
4. scores SE(3) geometry
5. refreshes the benchmark report and dashboard data
6. writes a single output zip of the new sidecar artifacts

## Files Added In This Phase

- [17_generate_v5_3_model_guided_macrocycles.py](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/scripts/17_generate_v5_3_model_guided_macrocycles.py)
- [20_measure_v5_3_benchmark_gaps.py](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/scripts/20_measure_v5_3_benchmark_gaps.py)
- [23_score_se3_geometry.py](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/scripts/23_score_se3_geometry.py)
- [18_build_v5_3_benchmark_report.py](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/scripts/18_build_v5_3_benchmark_report.py)
- [extract_data.py](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/interface/extract_data.py)

## Future Phase, Explicitly Deferred

True pocket-guided generation requires:

1. seed docking or aligned seed poses
2. seed-level pocket contact attribution
3. residue-aware anchor bias
4. residue-aware linker template chemistry
5. only then a true pocket-biased generation mode
