# Current Status

Last local verification: 2026-05-14.

Final project status: finalized as a computational architecture and candidate-prioritization package. Further benchmark-scale training is deferred because the available free Colab T4 runtime is not long or stable enough for reliable anchor GNN, linker, SE(3), regeneration, docking, and reranking cycles.

## Completed

- Project folder structure, registry, config, requirements, and logs are in place.
- ChEMBL data collection was expanded to a 1,500-record pull.
- `1,135` curated JAK2 ligands are saved in `02_curated_data/jak2_curated_ligands.csv`.
- The curated set includes `97` real JAK2 macrocycles with 12-20 atom rings.
- PDB `5AEP` is downloaded and the docking grid is inferred from co-crystallized `QUP`.
- RDKit descriptors and SE(3) graph tensors were rebuilt for all `1,135` curated ligands.
- Colab SE(3) training artifacts were imported: `50` epochs on `cuda`, best validation loss `9.0621`.
- V5.3 macrocycle fragment-linker datasets were built from JAK2 macrocycles and broad ChEMBL macrocycle pretraining data.
- Anchor-site and linker-size models were pretrained on broad macrocycles and fine-tuned on JAK2 macrocycles.
- V5.3 model-guided macrocycle generation produced `114` valid high-confidence candidates.
- The merged candidate pool with V5.3 contains `6,888` unique candidates.
- Dedicated V5.3 docking inputs were prepared for all `114` candidates.
- A fixed-column Vina-GPU ligand sanitizer was added and cleaned `114 / 114` PDBQT files.
- The Vina-GPU sanitizer made `552` atom-type replacements and removed the bad `CG0/CG1/NG2/OG3/SG2` failure mode.
- The Colab Vina-GPU workflow was replaced with a one-cell driver and automatic failure-debug zip.
- Vina-GPU 2.1 completed on Colab T4 for all `114` V5.3 model-guided candidates.
- Parsed V5.3 GPU docking scores: `114 / 114`.
- V5.3 GPU score distribution: best `-11.4`, median `-8.3`, mean `-8.6974` kcal/mol.
- Normalized V5.3 GPU poses: `114`.
- V5.3 top-pose sanity checks: `10 / 10` pass, no hard clashes below `1.8 A`.
- V5.3 ADMET, ranking, pose sanity, and benchmark outputs were regenerated.
- Benchmark report now includes `v5_3_model_guided_vina_gpu_2_1`.
- V5.3 GPU candidate decision package was built with top 10 CSV, markdown report, molecule images, top 10 PDBQT poses, and zip.
- V5.3 benchmark gaps are now measured explicitly: current linker novelty is `24.5614%` (`28 / 114`), below MED's `82.81%`.
- Raw-attempt validity is still unavailable for the existing `114`-candidate generation run because that run was created before attempt logging existed.
- Future V5.3 generation now writes a raw attempt log so validity can be measured from all attempted molecules, not just filtered survivors.
- Anchor-site featurization was expanded from `10` to `18` features, including charge, local ring distance, heteroatom context, terminal-chain status, and connectivity features.
- Anchor, linker-size, and SE(3) training scripts now support early stopping, gradient clipping, and checkpoint cadence controls to reduce overfitting during longer Colab training runs.
- A pure-PyTorch anchor GNN trainer was added at `scripts/14b_train_v5_3_anchor_gnn.py`, and generation can now use its `anchor_gnn_model.pt` checkpoint directly.
- Linker-size prediction now supports `16` features, including anchor-to-anchor path length and path rotatable fraction, and generation predicts linker size per anchor pair instead of once globally per molecule.
- `scripts/03_train_se3_main.py` now supports `--resume` for Colab continuation from `se3_latest_checkpoint.pt`.
- A pocket/electronic-fit layer was added to score JAK2 docked poses by PDBQT partial-charge complementarity, H-bond opportunity, hydrophobic packing, aromatic contacts, polar contacts, and clash penalties.
- Pocket/electronic fit was scored for `114 / 114` V5.3 Vina-GPU poses; best score `0.844276`, median `0.8146025`, best pocket-fit-only candidate `CAND_69ce89568d`.
- A pocket-guided V5.3 ranking table was built; `CAND_5152217faa` remains rank 1 after combining the original rank with pocket/electronic fit.
- A gap-closing Colab training plan was added for expanded macrocycle data, stronger anchor/linker retraining, SE(3) continuation, regeneration, and benchmark remeasurement.
- Benchmark comparison logic now blocks a MED-beaten claim unless raw-attempt validity and linker novelty are both measured and exceed the paper reference.
- Current validation report: `54 / 54` checks pass.
- Latest unit tests: `18 / 18` pass.
- Final project summary document added at `docs/FINAL_PROJECT_SUMMARY.md`.
- Python bytecode and test cache directories were removed; source code, models, reports, docking artifacts, and scores were preserved for reproducibility.

## Current Top V5.3 GPU Candidates

| Rank | Candidate | Vina score | Pose | Final score | Tier |
|---:|---|---:|---|---:|---|
| 1 | `CAND_5152217faa` | `-11.4` | pass | `0.860718` | final_candidate |
| 2 | `CAND_396f994be3` | `-11.4` | pass | `0.844347` | final_candidate |
| 3 | `CAND_598172bd63` | `-11.3` | pass | `0.844002` | final_candidate |
| 4 | `CAND_9c6324c16c` | `-11.3` | pass | `0.842709` | final_candidate |
| 5 | `CAND_3c64f44e76` | `-11.3` | pass | `0.836861` | final_candidate |
| 6 | `CAND_1675c88065` | `-11.0` | pass | `0.824937` | backup_candidate |
| 7 | `CAND_f134535edd` | `-11.0` | pass | `0.814857` | backup_candidate |
| 8 | `CAND_d3037c6bbb` | `-10.9` | pass | `0.812604` | backup_candidate |
| 9 | `CAND_9b2dbc98ed` | `-11.0` | pass | `0.809978` | backup_candidate |
| 10 | `CAND_dbfbb7d1cf` | `-11.2` | pass | `0.807922` | backup_candidate |

## Key Updated Files

- `docs/EMD_V5_3_Project_Architecture_Flow.md`
- `docs/FINAL_PROJECT_SUMMARY.md`
- `docs/V5_3_BENCHMARK_GAP_CLOSING_COLAB_STEPS.md`
- `docs/V5_4_POCKET_ELECTRONIC_GUIDANCE_PLAN.md`
- `docs/SCIENTIFIC_BOUNDARIES.md`
- `README.md`
- `src/emd_v5_2_hybrid/linker_size_features.py`
- `src/emd_v5_2_hybrid/anchor_gnn.py`
- `src/emd_v5_2_hybrid/pocket_electronics.py`
- `scripts/14b_train_v5_3_anchor_gnn.py`
- `scripts/17_generate_v5_3_model_guided_macrocycles.py`
- `scripts/21_score_pocket_electronics.py`
- `06_docking/v5_3_model_guided/scores/pocket_electronic_fit_scores.csv`
- `08_final_ranking/v5_3_model_guided_pocket_electronic_ranked_candidates.csv`
- `06_docking/v5_3_model_guided/scores/docking_scores_full_vina_gpu_2_1.csv`
- `06_docking/v5_3_model_guided/scores/pose_sanity_scores.csv`
- `08_final_ranking/v5_3_model_guided_ranked_candidates.csv`
- `09_reports/v5_3_benchmark/EMD_V5_3_Benchmark_Report.md`
- `09_reports/v5_3_gpu_decision_package/V5_3_GPU_Candidate_Decision_Package.md`
- `09_reports/v5_3_gpu_decision_package/V5_3_GPU_Candidate_Decision_Package.zip`

## Optional Future Work

- Do not claim experimental potency, safety, or clinical readiness; these are computational prioritization results only.
- Visually inspect the top five docked poses in PyMOL or ChimeraX.
- Re-dock or rescore the top ten with an independent seed/config or a second docking engine.
- Check protonation, tautomer, and charge states for the top five.
- Extend the new interaction-aware reranking into generation feedback, so future anchor/linker choices are biased by pocket/electronic fit rather than only filtered after docking.
- Add true quantum-electronic/ESP rescoring for the top 10 only, using xTB or DFT-lite if Colab runtime allows.
- Add parent-potency-aware seed weighting for future generation cycles.
- Rerun V5.3 generation with the new attempt logger, then remeasure raw-attempt validity and linker novelty.
- Improve linker novelty substantially; the current measured value is `24.5614%`, while MED reports `82.81%`.

These are future-resource tasks, not blockers for the current project package.

## Verified Commands

```powershell
python scripts/08_validate_project.py --base .
python scripts/18_build_v5_3_benchmark_report.py --base .
python scripts/20_measure_v5_3_benchmark_gaps.py --base .
python scripts/19_build_v5_3_gpu_decision_package.py --base . --top-n 10
python -m compileall -q src scripts tests
python -m unittest discover -s tests
```

## Next Best Step

Hand over the finalized project zip and the final decision package to the team. If the team later obtains longer GPU access, resume from:

`docs/V5_3_BENCHMARK_GAP_CLOSING_COLAB_STEPS.md`
