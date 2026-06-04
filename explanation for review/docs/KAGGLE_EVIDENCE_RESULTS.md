# Kaggle Evidence Results

This document records the real outcomes from the Kaggle evidence-upgrade reruns that were merged back into the project on May 15, 2026.

## Branch Summary

### Frozen Reference Branch

- Branch: `frozen_v5_3_baseline`
- Candidates: `114`
- Scope: `filtered_survivor`
- Linker novelty: `24.5614%`

This remains the frozen project reference branch. It is useful for ranking, docking, and candidate discussion, but it does not provide raw-attempt validity.

### Logged Kaggle Rerun

- Branch: `kaggle_logged_rerun`
- Candidates: `114`
- Scope: `raw_attempt_logged`
- Raw attempt rows: `629`
- Valid output rows: `114`
- Raw-attempt validity: `18.1240%`
- Linker novelty: `26.3158%`
- Uniqueness: `100.0%`
- Macrocyclization: `100.0%`

Status-count breakdown:

- `valid_output`: `114`
- `no_valid_anchor_pair`: `236`
- `filtered_non_macrocycle_or_basic_filters`: `140`
- `duplicate_or_invalid_record`: `129`
- `ring_closure_failed`: `10`

Interpretation:

- The logged rerun closes the main measurement gap by providing a real raw-attempt validity number.
- The architecture remains highly selective and chemically disciplined, but its valid-output yield is far below MED.

### Diverse Kaggle Rerun

- Branch: `kaggle_diverse_rerun`
- Candidates: `68`
- Scope: `raw_attempt_logged`
- Raw attempt rows: `2688`
- Valid output rows: `68`
- Raw-attempt validity: `2.5298%`
- Linker novelty: `100.0%`
- Uniqueness: `100.0%`
- Macrocyclization: `100.0%`

Status-count breakdown:

- `valid_output`: `68`
- `rejected_known_linker`: `1720`
- `filtered_non_macrocycle_or_basic_filters`: `429`
- `no_valid_anchor_pair`: `236`
- `duplicate_or_invalid_record`: `125`
- `ring_closure_failed`: `110`

Interpretation:

- The diverse novelty-enforced branch succeeds strongly at exact-linker novelty.
- That gain comes with a major validity penalty.
- This means the novelty lever works, but it is not yet balanced well enough to beat MED on a full benchmark basis.

## MED Comparison

Active comparison branch:

- `kaggle_diverse_rerun`

Comparable metric outcomes versus MED:

- Validity: `does_not_exceed`
- Uniqueness: `exceeds`
- Macrocyclization: `exceeds`
- Linker novelty: `exceeds`

Final benchmark claim status:

- `paper_equivalent_claim_allowed = false`
- `beats_all_comparable_metrics = false`

## Scientific Takeaway

The Kaggle evidence upgrade does three useful things for the project:

1. It replaces the missing validity gap with a real raw-attempt measurement.
2. It proves the novelty-enforcement mechanism works.
3. It shows clearly that validity, not novelty, is now the dominant benchmark bottleneck.

## Honest Project Position

ElectroMacroDiff is currently best presented as:

- a low-resource macrocycle generation system,
- with docking,
- pocket-electronic rescoring,
- and auxiliary SE(3) geometry evidence,
- but not yet a fully protein-conditioned diffusion generator,
- and not yet a benchmark-beating MED replacement.

## Files Updated From Kaggle

- [emd_v5_3_benchmark_summary.json](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/09_reports/v5_3_benchmark/emd_v5_3_benchmark_summary.json)
- [EMD_V5_3_Benchmark_Report.md](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/09_reports/v5_3_benchmark/EMD_V5_3_Benchmark_Report.md)
- [v5_3_model_guided_pocket_electronic_ranked_candidates_with_se3.csv](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/08_final_ranking/v5_3_model_guided_pocket_electronic_ranked_candidates_with_se3.csv)
- [data.js](C:/Users/srira/Desktop/new_plan/EMD_V5_2_Hybrid/interface/data.js)
