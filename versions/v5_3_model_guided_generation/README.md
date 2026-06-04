# V5.3 Model-Guided Generation Archive

This folder preserves the V5.3 stage, where the project moved from baseline macrocycle generation toward model-guided anchor and linker decisions.

## What V5.3 Added

- Fragment-linker training datasets for macrocycle reconstruction
- Anchor-site training data and anchor model training logs
- Linker-size pretraining and JAK2 fine-tuning logs
- V5.3 generated macrocycle candidate tables
- Vina-GPU 2.1 docking evidence for generated candidates
- ADMET, ranking, benchmark, and decision-package outputs

## Main Model Evidence

| Model stage | Folder |
|---|---|
| Anchor pretrain | `04_models_checkpoints/v5_3_anchor_pretrain/` |
| Anchor fine-tune | `04_models_checkpoints/v5_3_anchor/` |
| Linker-size pretrain | `04_models_checkpoints/v5_3_linker_size_pretrain/` |
| Linker-size fine-tune | `04_models_checkpoints/v5_3_linker_size/` |

## Main Data Evidence

| Data type | Path |
|---|---|
| JAK2 fragment-linker pairs | `02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv` |
| Broad macrocycle fragment-linker pairs | `02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv` |
| Broad macrocycle pretraining set | `02_curated_data/v5_3_pretrain_macrocycles.csv` |
| JAK2 anchor atom training table | `03_features/v5_3_anchor_atom_training.csv` |
| Broad anchor atom pretraining table | `03_features/v5_3_pretrain_anchor_atom_training.csv.gz` |

The broad anchor pretraining table is compressed because the uncompressed CSV is too large for normal GitHub storage. Verify it with `03_features/v5_3_pretrain_anchor_atom_training.manifest.json`.

## Docking and Metrics

The `vina_gpu_2_1_results_full/` folder is extracted from the V5.3 Vina-GPU 2.1 result package and preserves the molecules and metrics from that stage. The main parsed results are also available under:

```text
06_docking/v5_3_model_guided/
07_admet_synthesis/
08_final_ranking/
09_reports/v5_3_benchmark/
09_reports/v5_3_gpu_decision_package/
```

This V5.3 archive explains the base from which V5.4 pocket scoring and V5.5 pocket-conditioned generation were built.
