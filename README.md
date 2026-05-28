# ElectroMacroDiff V5.5

Pocket-guided computational drug-discovery pipeline for JAK2-focused constrained macrocycle generation, Vina-GPU docking, ADMET proxy scoring, pocket-electronic rescoring, final candidate ranking, and interactive 3D review.

## Project Scope

ElectroMacroDiff V5.5 is a reproducible computational discovery package. It connects macrocycle generation with JAK2 pocket evidence, real docking outputs, interpretable scoring tables, and a browser-based dashboard for reviewing ranked protein-ligand poses.

This repository reports computational candidates only. It does not claim experimentally proven potency, synthesis success, clinical safety, FDA readiness, or wet-lab validation.

## V5.5 Snapshot

| Metric | Value |
|---|---:|
| Curated JAK2 ligands | 1,135 |
| Generated V5.5 candidates | 1,113 |
| Ranked V5.5 candidates | 1,113 |
| Docked candidates with parsed scores | 1,036 |
| Best Vina-GPU score | -12.7 kcal/mol |
| Median Vina-GPU score | -8.4 kcal/mol |
| Pocket-scored candidates | 200 |
| Best pocket-electronic fit score | 0.84838 |
| Top ranked candidate | CAND_3688f8acdc |
| Raw generation attempts | 87,287 |
| Raw-attempt validity | 1.2751% |
| Linker novelty | 100% |
| Uniqueness | 100% |
| Macrocyclization | 100% |
| Validation checks | 54/54 passed |
| Unit tests | 40/40 passed |

## Dashboard

The interactive dashboard is in `interface/` and is designed for static hosting. It includes:

- ranked V5.5 candidate tables
- project metrics and charts
- candidate detail views
- embedded JAK2 receptor and docked pose data
- interactive 3D protein-ligand review with 3Dmol.js

After GitHub Pages is enabled for the deployed branch, the dashboard is intended to be available at:

```text
https://jagirisriram67-ctrl.github.io/ElectroMacroDiff-V5.5/
```

## Folder Map

```text
00_project_registry/      Config, registry, progress, and artifact records
01_raw_data/              Raw target, ligand, receptor, and literature inputs
02_curated_data/          Clean ligand/activity tables and training inputs
03_features/              Descriptors, graph features, and feature tables
04_models_checkpoints/    Model checkpoints and training summaries
05_generated_candidates/  V5.5 generated molecules and attempt logs
06_docking/               Docking inputs, parsed scores, pose files, and summaries
07_admet_synthesis/       ADMET, synthesis, and safety proxy outputs
08_final_ranking/         Consensus and pocket-guided candidate ranking tables
09_reports/               Reports, candidate cards, and decision packages
10_notebooks/             Colab/Kaggle workflow notebooks and templates
11_logs/                  Daily logs and AI assistance notes
docs/                     Handover reports, methods, plans, and boundaries
interface/                Static UI dashboard
scripts/                  Command-line pipeline stages
src/                      Reusable Python implementation modules
tests/                    Unit and validation tests
```

## Key Reports

- `docs/v5_5_final_handover_report_2026-05-16.md`
- `docs/PROJECT_COMPLETION_HANDOVER_V5_5.md`
- `docs/V5_5_GENERATION_RESULTS.md`
- `docs/V5_5_GPU_DOCKING_RESULTS.md`
- `docs/SCIENTIFIC_BOUNDARIES.md`

## Local Checks

From the repository root:

```powershell
python -m unittest discover -s tests
python scripts/08_validate_project.py --base .
```

For dashboard-only review, open:

```text
interface/index.html
```

## Claim Boundary

Allowed claim:

> ElectroMacroDiff V5.5 conditions anchor selection, linker choice, and validity gating on JAK2 pocket-electronic context during generation, then validates and prioritizes candidates through Vina-GPU docking and pocket-electronic rescoring.

Not allowed claim:

> ElectroMacroDiff V5.5 proves experimental activity, clinical safety, synthesis success, or fully replaces benchmark protein-conditioned molecular generation systems.

## License

MIT License. See `LICENSE`.
