# ElectroMacroDiff V5.5 Project Completion Project package

## Completion Status

The project folder is ready for team/coordinator evaluation as a computational
project package.

This project package uses real outputs currently present in the project folder only.
No experimental potency, synthesis success, clinical safety, or full
protein-conditioned diffusion claim is made.

## Final Evidence Snapshot

| Evidence item | Real project result |
|---|---:|
| Curated JAK2 ligands | 1135 |
| V5.5 generated candidates | 1113 |
| Vina-GPU docked candidates | 1036 |
| Parsed Vina-GPU scores | 1036 |
| Best Vina-GPU score | -12.7 kcal/mol |
| Median Vina-GPU score | -8.4 kcal/mol |
| Pocket-electronic scored candidates | top 200 |
| Best pocket-electronic fit | 0.848380 |
| Median pocket-electronic fit | 0.783945 |
| Raw generation attempts | 87287 |
| Raw-attempt validity | 1.2751% |
| Linker novelty | 100.0% |
| Macrocyclization | 100.0% |
| Uniqueness | 100.0% |
| Project validation | 54/54 passed |
| Unit tests | 46/46 passed |

## Current Top Ranked Candidate

| Field | Value |
|---|---|
| Candidate ID | `CAND_3688f8acdc` |
| Vina-GPU score | -12.7 kcal/mol |
| Pocket-electronic fit | 0.810776 |
| Pocket-guided final score | 0.874199 |
| Decision | primary candidate |

## Interface Dashboard

Open:

`interface/index.html`

Dashboard features:

- all 1113 ranked candidates
- search by candidate ID, parent ID, SMILES, or InChIKey
- sorting by rank, Vina score, final score, pocket fit, QED, ADMET, MW, or LogP
- filters for docked/not docked candidates
- filters for pocket-scored/not pocket-scored candidates
- macrocycle filtering
- visible-row CSV export
- real docking and pocket-fit distributions
- real attempt-status counts
- real benchmark-branch table
- selected-candidate detail panel

## Main Files For Evaluation

- `interface/index.html`
- `interface/data.js`
- `05_generated_candidates/v5_5_pocket_guided/generated_v5_5_pocket_guided.csv`
- `05_generated_candidates/v5_5_pocket_guided/v5_5_pocket_guided_attempt_log.csv`
- `06_docking/v5_5_pocket_guided/scores/docking_scores_full_vina_gpu_2_1.csv`
- `06_docking/v5_5_pocket_guided/scores/pocket_electronic_fit_scores.csv`
- `06_docking/v5_5_pocket_guided/scores/v5_5_docking_and_pocket_summary.json`
- `08_final_ranking/v5_5_pocket_guided_pocket_electronic_ranked_candidates.csv`
- `09_reports/v5_3_benchmark/EMD_V5_3_Benchmark_Report.md`
- `00_project_registry/final_project_verification.json`

## Honest Claim

ElectroMacroDiff V5.5 conditions anchor selection, linker choice, and validity
gating on JAK2 pocket-electronic context during generation, then validates and
prioritizes generated macrocycles using Vina-GPU docking, ADMET scoring,
pose sanity checks, and pocket-electronic rescoring.

## Explicit Limitations

- Candidates are computational only.
- No wet-lab validation has been performed.
- Pocket-electronic fit is a proxy from PDBQT charges and interaction features,
  not quantum electron density.
- V5.5 is pocket/electronic-conditioned generation, not full residue-level
  protein-conditioned diffusion.
- Pocket-electronic rescoring was run for the top 200 ranked docked candidates.
