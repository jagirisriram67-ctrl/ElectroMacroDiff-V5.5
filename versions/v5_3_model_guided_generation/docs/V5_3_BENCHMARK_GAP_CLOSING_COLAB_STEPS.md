# V5.3 Benchmark Gap-Closing Colab Plan

Use this after the current V5.3 Vina-GPU baseline. The project pipeline is complete, but the MED paper benchmark is not beaten yet because raw-attempt validity is not logged for the old generation run and measured linker novelty is only `24.5614%`, below MED's `82.81%`.

This workflow is aggressive but validation-driven: it expands data, rebuilds enhanced anchor features, trains with early stopping, logs raw generation attempts, measures linker novelty, and rebuilds the benchmark. Long training is allowed, but the scripts save the best validation checkpoint instead of blindly trusting the final epoch.

## Current Baseline

| Area | Current result | Meaning |
|---|---:|---|
| V5.3 GPU docking | `114 / 114` parsed | Docking pipeline works. |
| Best Vina-GPU score | `-11.4` | Strong docking proxy, not experimental affinity. |
| Pose sanity | `10 / 10` pass | Top poses are geometrically reasonable. |
| Anchor fine-tune F1 | `0.2089` test | Weakest model. Needs better features/data. |
| Linker exact accuracy | `0.3441` test | Usable but not benchmark-level. |
| Linker within-one accuracy | `0.6238` test | Decent, should improve with more data. |
| SE(3) best val loss | `9.0621` | Under-trained and not yet used for generation. |
| Linker novelty | `24.5614%` | Major MED gap. |
| Raw-attempt validity | unavailable | Must be logged in the next generation run. |

## What This Run Changes

- Anchor training rows now have 18 features instead of 10:
  `gasteiger_charge`, `num_rotatable_neighbors`, `shortest_path_to_ring`, `is_between_rings`, `neighbor_heteroatom_count`, `is_terminal_chain_atom`, `local_connectivity_index`, and `ring_size_of_nearest_ring` were added.
- Anchor/linker training now supports patience-based early stopping, gradient clipping, weight decay controls, and safer class-imbalance handling.
- A pure-PyTorch anchor GNN is available in `scripts/14b_train_v5_3_anchor_gnn.py`. It gives the anchor model molecular graph context without requiring PyTorch Geometric installation.
- Linker-size features now include anchor-pair path features: `anchor_path_bonds`, `anchor_path_3d_distance`, and `anchor_path_rotatable_fraction`.
- V5.3 generation can use either the MLP anchor checkpoint or the new GNN anchor checkpoint, and linker-size prediction is now conditioned on each proposed anchor pair.
- SE(3) continuation training now supports patience-based early stopping and `--resume`.
- V5.3 generation now writes a raw attempt log.
- Linker novelty is measured from generated macrocycle fragmentation against training linker SMILES.
- Docked candidates can now be rescored by pocket/electronic fit after Vina-GPU docking using `scripts/21_score_pocket_electronics.py`.
- Benchmark comparison no longer treats filtered-survivor validity as paper-equivalent raw validity.

## Colab Setup

Upload the latest project zip to Drive as:

`MyDrive/EMD_V5_3_gap_training_colab_ready.zip`

Run this setup cell first:

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil, zipfile
from pathlib import Path

ZIP_PATH = Path("/content/drive/MyDrive/EMD_V5_3_gap_training_colab_ready.zip")
EXTRACT_DIR = Path("/content/EMD_V5_3_gap_training")

if not ZIP_PATH.exists():
    raise FileNotFoundError(ZIP_PATH)

shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(EXTRACT_DIR)

marker = Path("05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv")
matches = [p.parent.parent.parent for p in EXTRACT_DIR.rglob(str(marker))]
if not matches:
    raise RuntimeError("Could not find project root after extraction.")
BASE = matches[0]
print("BASE =", BASE)
```

Install dependencies:

```python
import subprocess, sys

def run(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)

run([sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy", "rdkit", "torch", "matplotlib", "requests"])
```

## Session 1 - Expand Macrocycle Pretraining Data

This uses a wider ring window. It may take 1-3 hours depending on ChEMBL/API speed.

```python
run([
    sys.executable, "scripts/16_collect_macrocycle_pretraining_data.py",
    "--base", ".",
    "--scan-limit", "250000",
    "--page-size", "1000",
    "--max-macrocycles", "8000",
    "--min-ring-size", "10",
    "--max-ring-size", "24",
], cwd=BASE)
```

## Session 2 - Rebuild Expanded Fragment, Linker, And Anchor Tables

This is the required step before retraining the enhanced 18-feature anchor model.

```python
run([
    sys.executable, "scripts/13_build_v5_3_fragment_dataset.py",
    "--base", ".",
    "--input-csv", "02_curated_data/v5_3_pretrain_macrocycles.csv",
    "--fragment-output", "02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv",
    "--anchor-output", "03_features/v5_3_pretrain_anchor_atom_training.csv",
    "--max-pairs-per-molecule", "35",
    "--min-ring-size", "10",
    "--max-ring-size", "24",
], cwd=BASE)

run([
    sys.executable, "scripts/13_build_v5_3_fragment_dataset.py",
    "--base", ".",
    "--input-csv", "02_curated_data/jak2_curated_ligands.csv",
    "--fragment-output", "02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv",
    "--anchor-output", "03_features/v5_3_anchor_atom_training.csv",
    "--max-pairs-per-molecule", "35",
    "--min-ring-size", "10",
    "--max-ring-size", "24",
], cwd=BASE)
```

## Session 3 - Retrain Anchor MLP Fallback

Target: keep a stable 18-feature MLP fallback. This is not the preferred final anchor model, but it is useful if the GNN overfits.

```python
run([
    sys.executable, "scripts/14_train_v5_3_anchor_model.py",
    "--base", ".",
    "--input-csv", "03_features/v5_3_pretrain_anchor_atom_training.csv",
    "--output-dir", "04_models_checkpoints/v5_3_anchor_pretrain",
    "--epochs", "2000",
    "--hidden-dim", "192",
    "--learning-rate", "0.0005",
    "--patience", "200",
    "--min-delta", "0.00001",
    "--checkpoint-every", "100",
    "--loss", "focal",
    "--focal-alpha", "0.65",
    "--focal-gamma", "2.0",
    "--max-pos-weight", "30",
    "--grad-clip", "1.0",
    "--device", "cuda",
], cwd=BASE)

run([
    sys.executable, "scripts/14_train_v5_3_anchor_model.py",
    "--base", ".",
    "--input-csv", "03_features/v5_3_anchor_atom_training.csv",
    "--output-dir", "04_models_checkpoints/v5_3_anchor",
    "--init-checkpoint", "04_models_checkpoints/v5_3_anchor_pretrain/anchor_site_model.pt",
    "--epochs", "1000",
    "--hidden-dim", "192",
    "--learning-rate", "0.0002",
    "--patience", "150",
    "--min-delta", "0.00001",
    "--checkpoint-every", "50",
    "--loss", "focal",
    "--focal-alpha", "0.65",
    "--focal-gamma", "2.0",
    "--max-pos-weight", "30",
    "--grad-clip", "1.0",
    "--device", "cuda",
], cwd=BASE)
```

## Session 4 - Train Anchor GNN Preferred Model

This is the higher-upside anchor model. It uses graph message passing, so it can see local bond neighborhoods and ring context that the MLP cannot.

```python
run([
    sys.executable, "scripts/14b_train_v5_3_anchor_gnn.py",
    "--base", ".",
    "--anchor-csv", "03_features/v5_3_pretrain_anchor_atom_training.csv",
    "--fragment-csv", "02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv",
    "--output-dir", "04_models_checkpoints/v5_3_anchor_gnn_pretrain",
    "--epochs", "1200",
    "--batch-size", "24",
    "--hidden-dim", "128",
    "--num-layers", "4",
    "--learning-rate", "0.0003",
    "--patience", "180",
    "--min-delta", "0.00001",
    "--checkpoint-every", "50",
    "--loss", "focal",
    "--focal-alpha", "0.65",
    "--focal-gamma", "2.0",
    "--grad-clip", "1.0",
    "--device", "cuda",
], cwd=BASE)

run([
    sys.executable, "scripts/14b_train_v5_3_anchor_gnn.py",
    "--base", ".",
    "--anchor-csv", "03_features/v5_3_anchor_atom_training.csv",
    "--fragment-csv", "02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv",
    "--output-dir", "04_models_checkpoints/v5_3_anchor_gnn",
    "--init-checkpoint", "04_models_checkpoints/v5_3_anchor_gnn_pretrain/anchor_gnn_model.pt",
    "--epochs", "700",
    "--batch-size", "16",
    "--hidden-dim", "128",
    "--num-layers", "4",
    "--learning-rate", "0.0001",
    "--patience", "120",
    "--min-delta", "0.00001",
    "--checkpoint-every", "50",
    "--loss", "focal",
    "--focal-alpha", "0.65",
    "--focal-gamma", "2.0",
    "--grad-clip", "1.0",
    "--device", "cuda",
], cwd=BASE)
```

Use the GNN checkpoint for generation if its validation/test F1 is better than the MLP checkpoint.

## Session 5 - Retrain Linker-Size Model

Target: raise exact accuracy from `0.3441` and within-one accuracy from `0.6238`. This retraining uses the new 16-feature anchor-pair-aware linker inputs, so do not initialize it from old 13-feature checkpoints.

```python
run([
    sys.executable, "scripts/15_train_v5_3_linker_size_model.py",
    "--base", ".",
    "--input-csv", "02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv",
    "--output-dir", "04_models_checkpoints/v5_3_linker_size_pretrain",
    "--epochs", "2000",
    "--hidden-dim", "256",
    "--learning-rate", "0.0005",
    "--patience", "200",
    "--min-delta", "0.00001",
    "--checkpoint-every", "100",
    "--class-weighting", "balanced",
    "--grad-clip", "1.0",
    "--device", "cuda",
], cwd=BASE)

run([
    sys.executable, "scripts/15_train_v5_3_linker_size_model.py",
    "--base", ".",
    "--input-csv", "02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv",
    "--output-dir", "04_models_checkpoints/v5_3_linker_size",
    "--init-checkpoint", "04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_model.pt",
    "--epochs", "1000",
    "--hidden-dim", "256",
    "--learning-rate", "0.0002",
    "--patience", "150",
    "--min-delta", "0.00001",
    "--checkpoint-every", "50",
    "--class-weighting", "balanced",
    "--grad-clip", "1.0",
    "--device", "cuda",
], cwd=BASE)
```

## Session 6 - Continue SE(3) Training

This improves the geometry model evidence. It still does not make the current V5.3 generation fully MED-like until SE(3) output is integrated into linker coordinate generation.

```python
run([
    sys.executable, "scripts/03_train_se3_main.py",
    "--base", ".",
    "--resume",
    "--epochs", "180",
    "--batch-size", "8",
    "--hidden-dim", "128",
    "--num-layers", "4",
    "--learning-rate", "0.0001",
    "--checkpoint-every", "10",
    "--patience", "35",
    "--min-delta", "0.001",
    "--device", "cuda",
], cwd=BASE)
```

## Session 7 - Regenerate V5.3 Candidates With Raw Attempt Logging

Target: generate a larger, measured batch. The raw attempt log is now the important benchmark artifact.

```python
run([
    sys.executable, "scripts/17_generate_v5_3_model_guided_macrocycles.py",
    "--base", ".",
    "--seed-count", "500",
    "--max-products-per-seed", "20",
    "--top-anchor-atoms", "16",
    "--min-anchor-probability", "0.03",
    "--random-seed", "43",
    "--anchor-checkpoint", "04_models_checkpoints/v5_3_anchor_gnn/anchor_gnn_model.pt",
    "--linker-checkpoint", "04_models_checkpoints/v5_3_linker_size/linker_size_model.pt",
    "--device", "cuda",
], cwd=BASE)

run([
    sys.executable, "scripts/20_measure_v5_3_benchmark_gaps.py",
    "--base", ".",
], cwd=BASE)

run([
    sys.executable, "scripts/18_build_v5_3_benchmark_report.py",
    "--base", ".",
], cwd=BASE)
```

If the GNN summary is worse than the MLP summary, rerun only the generation cell with:

```python
"--anchor-checkpoint", "04_models_checkpoints/v5_3_anchor/anchor_site_model.pt",
```

instead of the GNN checkpoint.

## Session 8 - Package Results Back To Drive

After Vina-GPU docking results are merged back into the project, run this before final packaging if the pose files are available:

```python
run([
    sys.executable, "scripts/21_score_pocket_electronics.py",
    "--base", ".",
], cwd=BASE)

run([
    sys.executable, "scripts/19_build_v5_3_gpu_decision_package.py",
    "--base", ".",
    "--top-n", "10",
], cwd=BASE)
```

```python
import os, zipfile
from pathlib import Path

OUT = Path("/content/drive/MyDrive/EMD_V5_3_gap_training_results.zip")
files = [
    "02_curated_data/v5_3_pretrain_macrocycles.csv",
    "02_curated_data/v5_3_pretrain_fragment_linker_pairs.csv",
    "02_curated_data/v5_3_macrocycle_fragment_linker_pairs.csv",
    "03_features/v5_3_pretrain_anchor_atom_training.csv",
    "03_features/v5_3_anchor_atom_training.csv",
    "04_models_checkpoints/v5_3_anchor_pretrain/anchor_site_model.pt",
    "04_models_checkpoints/v5_3_anchor_pretrain/anchor_training_summary.json",
    "04_models_checkpoints/v5_3_anchor/anchor_site_model.pt",
    "04_models_checkpoints/v5_3_anchor/anchor_training_summary.json",
    "04_models_checkpoints/v5_3_anchor_gnn_pretrain/anchor_gnn_model.pt",
    "04_models_checkpoints/v5_3_anchor_gnn_pretrain/anchor_gnn_training_summary.json",
    "04_models_checkpoints/v5_3_anchor_gnn/anchor_gnn_model.pt",
    "04_models_checkpoints/v5_3_anchor_gnn/anchor_gnn_training_summary.json",
    "04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_model.pt",
    "04_models_checkpoints/v5_3_linker_size_pretrain/linker_size_training_summary.json",
    "04_models_checkpoints/v5_3_linker_size/linker_size_model.pt",
    "04_models_checkpoints/v5_3_linker_size/linker_size_training_summary.json",
    "04_models_checkpoints/se3_flow/se3_best_checkpoint.pt",
    "04_models_checkpoints/se3_flow/se3_latest_checkpoint.pt",
    "04_models_checkpoints/se3_flow/training_summary.json",
    "05_generated_candidates/model_guided_macrocycle/generated_v5_3_model_guided_macrocycles.csv",
    "05_generated_candidates/model_guided_macrocycle/v5_3_model_guided_generation_attempt_log.csv",
    "05_generated_candidates/model_guided_macrocycle/v5_3_model_guided_linker_novelty.csv",
    "05_generated_candidates/model_guided_macrocycle/v5_3_model_guided_benchmark_gap_summary.json",
    "05_generated_candidates/merged/generated_merged_with_v5_3_model_guided.csv",
    "06_docking/v5_3_model_guided/scores/pocket_electronic_fit_scores.csv",
    "06_docking/v5_3_model_guided/scores/jak2_pocket_electronic_profile.json",
    "08_final_ranking/v5_3_model_guided_pocket_electronic_ranked_candidates.csv",
    "09_reports/v5_3_benchmark/EMD_V5_3_Benchmark_Report.md",
    "09_reports/v5_3_benchmark/emd_candidate_benchmark_metrics.csv",
    "09_reports/v5_3_benchmark/emd_v5_3_benchmark_summary.json",
]

with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for rel in files:
        p = BASE / rel
        if p.exists():
            z.write(p, rel)
        else:
            print("missing optional:", rel)
print("Saved:", OUT)
print("Size MB:", round(OUT.stat().st_size / 1024 / 1024, 2))
```

After downloading the zip, place it in:

`C:\Users\srira\Desktop\new_plan`

Then say:

`I uploaded EMD_V5_3_gap_training_results.zip. Merge it and rebuild generation, docking, ranking, benchmark.`

## Honest Success Criteria

Do not claim MED is beaten unless all of these are true:

- Raw-attempt validity is measured from the attempt log and exceeds `93.82%`.
- Linker novelty exceeds `82.81%`.
- Uniqueness remains near `99.94%+`.
- Macrocyclization remains near `99.92%+`.
- The claim uses a fixed benchmark-style split and a sufficiently larger generated set.
- Top candidates are redocked with the same engine/config used for the comparison.

Realistic near-term win:

- Anchor F1 improves meaningfully above `0.21`.
- Linker novelty rises above the current `24.56%`.
- Raw-attempt validity becomes measured.
- V5.3 still produces strong, pose-sane JAK2 candidates after independent redocking.
