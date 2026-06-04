# ElectroMacroDiff V6 "Benchmark Edition"
## Complete End-to-End Master Execution Blueprint
*Version date: 2026-05-12*

This is the upgraded execution plan for a 5-member B.Tech team. It maintains the 10-day timeline and Google Colab constraints of V5.2, but **pivots the scientific goal from a simple pipeline to establishing an industry-standard benchmark (SOTA)** for macrocycle generation using Google DeepMind-style foundational model techniques.

---

## 1. Final Project Identity
**Project name:** ElectroMacroDiff V6 Benchmark Edition

**Core claim:**
We established a lightweight, SOTA benchmark for **electronic-aware macrocycle generation** inside protein pockets. By leveraging self-supervised pre-training on 50,000+ generic macrocycles and embedding pseudo-QM electronic features into an SE(3) Flow Matching model, we outperformed classical RDKit/SELFIES baselines across Validity, Uniqueness, Novelty (VUN), and Frechet ChemNet Distance (FCD), all executed under free-tier computational constraints.

**What the project should claim:**
*   Created a pre-trained foundational model for macrocycle 3D physics.
*   Achieved conditioned Structure-Based Drug Design (SBDD) within the JAK2 pocket.
*   Set a reproducible benchmark for VUN and FCD metrics in low-resource environments.
*   Results are high-fidelity computational hypotheses prioritized by consensus docking.

**What the project must not claim:**
*   Experimentally proven JAK2 inhibitors.
*   Clinically safe drugs or FDA-ready molecules.

---

## 2. Real Constraints & DeepMind-Style Solutions
**Team:** 5 students (B.Tech level).
**Time:** 10 days.
**Compute:** Google Colab free tier (T4 GPU).

**The DeepMind Shift:**
You cannot set a benchmark by training deep learning on 700 molecules.
*   **The Fix:** Phase 1 pre-trains the model on 50k+ massive generic macrocycles (learning physics). Phase 2 fine-tunes on the 700 curated JAK2 inhibitors (learning target specificity).
*   **Storage:** Google Drive remains the mandatory persistent storage layer for the massive datasets and model checkpoints.

---

## 3. Final Deliverables
*   **Massive Pre-training Dataset:** 50k+ curated generic macrocycles.
*   **Target Dataset:** Curated JAK2 ligand dataset with ETKDGv3 conformers.
*   **Pre-trained Foundation Model:** SE(3) model checkpoint that understands generic macrocycle geometry.
*   **Fine-Tuned SBDD Model:** Model conditioned on the JAK2 5AEP pocket.
*   **Benchmark Evaluation:** FCD, VUN, and RMSD metrics comparing SE(3) against SELFIES/RDKit baselines.
*   **Consensus Docking Scores:** Evaluated via Vina and GNINA.
*   **Final Ranked Top 3-5 Molecules:** With full TPP cards and AiZynthFinder synthesizability routes.
*   **WandB Logs:** Professional cloud tracking of the training curves.

---

## 4. Team Roles (V6 Upgrade)
| Member | Role | Main Responsibility |
| :--- | :--- | :--- |
| **Student 1** | Integrator / Lead | Drive registry, `emd_pipeline` modules, GitHub syncing, final report. |
| **Student 2** | Data Lead | ChEMBL massive macrocycle mining (50k+), JAK2 curation, splits. |
| **Student 3** | Model Lead | SE(3) Pre-training (Denoising), Fine-tuning, WandB integration. |
| **Student 4** | Chemistry Lead | Electronic features (Gasteiger), ETKDGv3 conformers, baseline generation. |
| **Student 5** | Evaluation Lead | Vina/GNINA consensus docking, FCD/VUN metrics, ADMET, Synthesizability. |

---

## 5. Google Drive & Local Project Structure
Your local code is synced via GitHub, but Google Drive holds the data.

```text
EMD_V6_Benchmark/
  00_project_registry/
  01_raw_data/
    massive_macrocycles/   <-- NEW
    jak2_target/
    pdb/
  02_curated_data/
  03_features/
    etkdg_conformers/      <-- UPGRADED
    ligand_graphs/
  04_models_checkpoints/
    se3_pretrained/        <-- NEW
    se3_finetuned/         <-- NEW
  05_generated_candidates/
  06_docking/
  07_admet_synthesis/
  08_benchmark_evaluation/ <-- NEW (FCD, VUN, RMSD)
  09_reports/
  11_logs/
```

---

## 6. Dataset & Feature Engineering Plan

### 6.1 The Massive Pre-training Set (Pillar 1)
*   **Source:** ChEMBL and ZINC20.
*   **Filter:** Pull *any* molecule where `max_ring_size >= 12`.
*   **Target Size:** 50,000 to 100,000 molecules.
*   **Purpose:** Teach the model the physics of ring closure and 3D stability.

### 6.2 Target Set & Pocket (Pillar 2)
*   **Receptor:** JAK2 (PDB: 5AEP). Extract the ATP pocket coordinates to feed as context nodes to the SE(3) model.
*   **Ligands:** ~700 JAK2 specific ligands.

### 6.3 High-Fidelity Physics Features
*   **Conformers:** Must use RDKit `ETKDGv3` with `useMacrocycleTorsions=True`.
*   **Node (Atom) Features:**
    *   Atomic number embedding.
    *   **Gasteiger partial charges (Crucial for electronic-awareness).**
    *   TPSA per-atom contribution.
    *   Hybridization & Aromaticity.
*   **Edge (Bond) Features:** Conjugation flags and bond order.

---

## 7. The V6 Model Execution Strategy

### Phase 1: Foundational Pre-Training (Days 3-5)
*   **Data:** 50k+ generic macrocycles.
*   **Task:** Coordinate Denoising. The model receives noisy 3D coords and must predict the vector field to restore the clean ETKDGv3 coordinates.
*   **Goal:** 50-100 epochs until validation loss plateaus.

### Phase 2: SBDD Fine-Tuning (Days 6-7)
*   **Data:** 700 JAK2 ligands + 5AEP pocket coordinates.
*   **Task:** Pocket-conditioned generation.
*   **Goal:** The model learns to generate the macrocycle *specifically* to fit the shape and electrostatics of the JAK2 pocket.

### Phase 3: Hybrid Generation (Day 8)
*   **SE(3) Generator:** Generate 1,000+ candidates conditioned on 5AEP.
*   **Baseline Generators:** SELFIES mutation and RDKit recombination (to prove the benchmark).

---

## 8. The "Unbeatable Benchmark" Evaluation

To prove your pipeline is a SOTA benchmark, Student 5 must compute:
1.  **VUN Score:** Validity (%), Uniqueness (%), Novelty against training set (%).
2.  **Frechet ChemNet Distance (FCD):** Compare the statistical distribution of SE(3) outputs vs. SELFIES outputs vs. real JAK2 drugs. (SE(3) must have the lowest distance to real drugs).
3.  **Consensus Docking:** Run AutoDock Vina AND GNINA. Average the normalized scores. SE(3) must statistically outperform baselines.
4.  **Synthesizability:** SA_Score < 4.0 or AiZynthFinder route probability > 60%.

---

## 9. 10-Day "Future Star" Execution Schedule

*   **Day 1:** Setup GitHub, Drive, WandB. Run Tiny Debug on `emd_pipeline`.
*   **Day 2:** Student 2 mines the 50k massive macrocycle dataset. Curate JAK2 ligands.
*   **Day 3:** Student 4 computes ETKDGv3 conformers and Gasteiger charges for the massive set.
*   **Day 4:** Student 3 starts **Pre-training** the SE(3) model on the massive set (Denoising).
*   **Day 5:** Pre-training finishes. Verify checkpoint stability.
*   **Day 6:** Fine-tune SE(3) on JAK2 + 5AEP pocket (SBDD).
*   **Day 7:** Generation phase. Generate candidates via SE(3), SELFIES, and RDKit.
*   **Day 8:** Compute Benchmark Metrics (VUN, FCD). Run Consensus Docking (Vina + GNINA).
*   **Day 9:** ADMET filtering, Synthesizability scoring, and consensus ranking formula.
*   **Day 10:** Final TPP molecule cards, WandB training curves export, and Final Report generation.

---

## 10. Final Ranking Formula
To ensure the final selected molecules are scientifically flawless:

```text
final_score =
  0.40 * consensus_docking_norm (Vina + GNINA)
+ 0.20 * admet_norm (QED, Lipinski)
+ 0.20 * physics_norm (SE3 Flow Matching Confidence / 3D strain)
+ 0.15 * synthesis_norm (AiZynthFinder or SA Score)
+ 0.05 * novelty_norm
```
