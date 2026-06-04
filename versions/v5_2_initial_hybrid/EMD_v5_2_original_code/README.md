# ElectroMacroDiff V5.2 Hybrid

> **Production-ready JAK2 macrocyclic inhibitor discovery pipeline**
> SE(3) Flow Matching + SELFIES + RDKit | Google Colab Optimized | 32/32 Tests Passing

---

## Quick Start (< 5 minutes)

```bash
# 1. Upload this entire folder to Google Drive
# 2. Open notebooks/quickstart_full_pipeline.ipynb in Colab
# 3. Run all cells — it works out of the box in TINY_DEBUG mode
```

**For local testing:**
```bash
pip install -r requirements_v5_2_hybrid_colab.txt
python test_pipeline.py          # 32 tests, ~14 seconds
python notebooks/quickstart_full_pipeline.py  # Full pipeline demo
```

---

## Architecture

```
DATA → FEATURES → SE(3) MODEL → GENERATION → DOCKING → ADMET → RANKING → TPP
 M1       M2          M3            M4           M5       M6       M7      M8
```

| Module | File | Purpose |
|--------|------|---------|
| M0 | `config_registry.py` | Config, checkpoints, seeds, registry |
| M1 | `data_collection.py` | ChEMBL fetch, PDB download, curation, scaffold split |
| M2 | `feature_engineering.py` | Descriptors, 3D conformers, SE(3) graph tensors |
| M3 | `se3_flow_model.py` | SE(3) equivariant flow matching: train + generate + coord→SMILES |
| M4 | `candidate_generation.py` | SELFIES mutation, RDKit atom mutation, filtering, Tanimoto diversity |
| M5 | `docking.py` | AutoDock Vina with resume, pocket extraction, pose viz |
| M6 | `admet_synthesis.py` | Lipinski/Veber/QED/PAINS/Brenk + SA Score + 12 safety alerts |
| M7 | `ranking_report.py` | Consensus ranking, molecule cards, TPP report with InChIKey |
| — | `visualization.py` | Dark-themed dashboards for datasets, training, ranking |
| — | `colab_utils.py` | GPU monitoring, batch sizing, session protection |

---

## Pipeline Features

### 3-Source Hybrid Generation
| Source | Method | Typical Yield |
|--------|--------|---------------|
| **SE(3)** | Flow matching ODE → coord reconstruction | 10-50 per run |
| **SELFIES** | Random token mutation + validity filter | 50+ per run |
| **RDKit** | Substituent swap + atom mutation + N-methylation | 30+ per run |

### Safety & Quality
- **12 structural alerts**: nitro, thiol, acyl chloride, azide, Michael acceptor, epoxide, hydrazine, aldehyde, peroxide, N-oxide, ring azo, nitrile
- **Tanimoto diversity**: Morgan FP (r=2) pairwise internal diversity per source
- **InChIKey**: Generated for all top candidates in TPP report

### Colab Optimization
- `TINY_DEBUG` mode for 2-minute validation runs
- Resume from any checkpoint after session disconnect
- Automatic GPU/CPU detection with optimal batch sizing

---

## Team Workflow (10-Day Campaign)

| Day | Notebooks | Owner |
|-----|-----------|-------|
| 1 | `00_environment`, `01_data_collection` | Student 1 |
| 2-3 | `02_ligand_features`, `03_se3_training` | Student 2 |
| 4-5 | `04_candidate_generation` | Students 3+4 |
| 6-7 | `05_docking` | Student 3 |
| 8 | `06_admet_synthesis` | Student 4 |
| 9 | `07_final_ranking_tpp` | Student 5 |
| 10 | Review, polish, submit | All |

---

## File Inventory

```
EMD_v5.2/
├── campaign_config.yaml              # All hyperparameters
├── requirements_v5_2_hybrid_colab.txt # Pip dependencies
├── test_pipeline.py                   # 32-test integration suite
├── convert_to_ipynb.py                # .py → .ipynb converter
├── setup_drive.py                     # Google Drive folder setup
├── emd_pipeline/
│   ├── __init__.py                    # 41 public exports
│   ├── config_registry.py             # M0: Config + checkpoints
│   ├── data_collection.py             # M1: ChEMBL + PDB
│   ├── feature_engineering.py         # M2: Descriptors + graphs
│   ├── se3_flow_model.py              # M3: SE(3) flow matching
│   ├── candidate_generation.py        # M4: Hybrid generation
│   ├── docking.py                     # M5: Vina docking
│   ├── admet_synthesis.py             # M6: ADMET + safety
│   ├── ranking_report.py              # M7: Ranking + TPP
│   ├── visualization.py               # Dashboard plots
│   └── colab_utils.py                 # Colab helpers
└── notebooks/
    ├── 00_environment_and_drive_check.ipynb
    ├── 01_data_collection_and_curation.ipynb
    ├── 02_ligand_features_and_splits.ipynb
    ├── 03_se3_dataloader_training.ipynb
    ├── 04_candidate_generation_hybrid.ipynb
    ├── 05_docking_and_pose_filtering.ipynb
    ├── 06_admet_synthesis_safety.ipynb
    ├── 07_final_ranking_and_tpp.ipynb
    └── quickstart_full_pipeline.ipynb
```

---

## Validation

```
32/32 tests passed in 14 seconds:
  M0: Config/dirs/progress/registry/seeds     5/5
  M1: Curation/split/PDB                      3/3
  M2: Descriptors/conformers/graphs            3/3
  M3: DataLoader/forward/backward/checkpoint   4/4
  M3+: SE(3) ODE generation + coord→SMILES     2/2
  M4: SELFIES/RDKit/filter+merge               3/3
  M6: ADMET/synthesis/safety                   3/3
  M7: Ranking/cards/TPP                        3/3
  VIZ: Dataset/ranking/grid                    3/3
  UTILS: GPU/batch/timer                       3/3
```

---

## License & Disclaimer

**This is a computational research tool.** All generated molecules are hypotheses.
No biological, pharmacological, or safety claims are made.
Experimental validation is required before any in-vitro or in-vivo studies.
