# ElectroMacroDiff V5.5: A Low-Resource Pocket-Electronic Workflow for JAK2 Macrocycle Candidate Generation and Prioritization

Author 1, Author 2, Author 3, Author 4, Author 5

## Abstract

Macrocyclic inhibitors are attractive in kinase drug discovery because their constrained conformations can improve target engagement and selectivity. We present ElectroMacroDiff V5.5, a low-resource computational workflow for generating and prioritizing JAK2 macrocycle candidates. The system combines curated JAK2 ligand data, model-guided macrocycle construction, validity and novelty gating, GPU docking, ADMET and synthesis proxy scoring, and a pocket-electronic rescoring layer derived from prepared receptor and docked pose outputs. The final V5.5 branch produced 1,113 ranked macrocycle candidates from 87,287 logged generation attempts. Of these, 1,036 candidates had parsed Vina-GPU 2.1 docking scores. The best docking score was -12.7 kcal/mol, and the top pocket-guided candidate, CAND_3688f8acdc, achieved a pocket-guided final score of 0.874199 with zero Lipinski violations and a 16-atom macrocycle. Linker novelty, uniqueness, and macrocyclization were each 100% in the final candidate set, while raw-attempt validity was 1.2751%. These results support ElectroMacroDiff as a reproducible computational prioritization system rather than an experimentally validated inhibitor discovery claim. The current architecture conditions generation decisions on JAK2 pocket-electronic context and then ranks docked poses by pocket fit, but it is not yet a full residue-level protein-conditioned diffusion generator.

## 1. Introduction

Kinase inhibitor discovery often requires balancing potency, selectivity, conformational control, and drug-like properties. Macrocycles provide one route toward this balance because ring closure can reduce conformational entropy and stabilize bioactive-like conformations. At the same time, macrocycle generation is difficult: a valid molecule must satisfy chemical valence, ring topology, linker length, synthetic plausibility, and pocket compatibility.

ElectroMacroDiff was developed as a practical, low-resource pipeline for JAK2 macrocycle exploration. The project goal is not only to generate new structures, but to connect generation with a target pocket, docking evidence, and interpretable candidate review. V5.5 advances this goal by introducing pocket/electronic-conditioned generation components and by preserving raw attempt logs so benchmark gaps can be reported honestly.

## 2. Methods

### 2.1 Data Curation

The project uses a curated JAK2 ligand collection containing 1,135 ligands. Candidate records retain molecule identifiers, SMILES, molecular descriptors, ring-size annotations, and downstream scoring fields. Macrocycle-focused generation outputs are tracked separately from merged or filtered-survivor branches to avoid mixing benchmark scopes.

### 2.2 Candidate Generation

V5.5 generates macrocycle candidates through a pocket-guided workflow. The system uses ligand-derived seeds, anchor selection, linker construction, chemotype control, exact linker novelty enforcement, and a validity/reward gate. The active branch, `kaggle_v5_5_pocket_guided`, logged 87,287 attempts and produced 1,113 valid output candidates. Attempt statuses were retained to distinguish raw-attempt validity from filtered-survivor validity.

### 2.3 Pocket-Electronic Context

The JAK2 pocket profile was derived from prepared receptor and pose data. The profile includes pocket atom counts, hydrophobic atom counts, donor/acceptor proxy counts, aromatic atom counts, net charge proxy, and contact-residue statistics. This pocket profile is a practical electronic proxy and residue-contact representation. It is not a quantum electron-density calculation.

### 2.4 Docking

Candidates were prepared for docking and evaluated against the JAK2 5AEP receptor using Vina-GPU 2.1. The docking grid used center coordinates `(32.53, 13.271, -3.863)` and box size `18 x 18 x 18`. The final output contains 1,036 parsed docking scores and 1,036 raw pose files.

### 2.5 ADMET, Synthesis, and Pocket-Fit Scoring

Candidates were scored using molecular property and ADMET/synthesis proxy metrics, including molecular weight, LogP, TPSA, hydrogen-bond donor/acceptor counts, rotatable bonds, QED, synthetic accessibility proxy score, Lipinski violations, Veber pass/fail status, and novelty proxy score. Docked candidates were additionally ranked using pocket-electronic fit components, including electrostatic, hydrogen-bond, hydrophobic, and aromatic contact terms.

### 2.6 Interactive Review

The project dashboard embeds real project data and provides an interactive 3D protein-ligand viewer. It displays the real JAK2 receptor, real exported docked pose texts, candidate contact residues, optional grid box, transparent ligand and pocket surfaces, and atom or bond labels. The viewer is designed for inspection and communication; it does not create new chemical structures.

## 3. Results

### 3.1 Generation and Benchmark Metrics

The final V5.5 branch produced 1,113 candidates from 87,287 logged generation attempts, giving a raw-attempt validity of 1.2751%. The final candidate set had 100% uniqueness, 100% linker novelty, and 100% macrocyclization. These values describe the accepted output set and logged V5.5 branch; they should not be interpreted as wet-lab validation.

The largest rejection category was `rejected_by_reward_model` with 78,257 attempts, followed by `rejected_known_linker` with 5,230 attempts. The reward gate therefore had a strong filtering role in the final branch.

### 3.2 Docking Results

Of the 1,113 ranked candidates, 1,036 had parsed Vina-GPU docking scores. The best score was -12.7 kcal/mol and the median score was -8.4 kcal/mol. The best-scoring candidate was also the top-ranked candidate after pocket-guided scoring.

### 3.3 Top Candidate

The top ranked candidate was `CAND_3688f8acdc`.

```text
CN1CCN(c2cc3cc(c2)Nc2nc4c(cccn4n2)-c2ccc(F)c(c2)CSCOC3)CC1
```

Key values for this candidate:

| Property | Value |
|---|---:|
| Vina-GPU score | -12.7 kcal/mol |
| Pocket-electronic fit score | 0.810776 |
| Pocket-guided final score | 0.874199 |
| Molecular weight | 490.608 |
| LogP | 4.7517 |
| QED | 0.4132 |
| SA score | 4.306 |
| Lipinski violations | 0 |
| Ring size | 16 |

### 3.4 Pocket Profile

The JAK2 pocket profile contained 227 pocket atoms, a net charge proxy of -0.357, 45 hydrophobic atoms, 59 donor atoms, 72 acceptor atoms, and 13 aromatic atoms. The most frequent contact residues included ARG:A:980, TYR:A:931, ASN:A:981, ARG:A:938, ASN:A:859, LEU:A:932, ASP:A:939, ASP:A:994, LEU:A:855, LYS:A:882, MET:A:929, and SER:A:936.

## 4. Discussion

ElectroMacroDiff V5.5 demonstrates an integrated computational workflow that connects macrocycle generation with target-pocket evidence. The most important practical outcome is that candidate ranking is not based on generation alone. Candidates are docked, rescored by pocket-electronic contact features, filtered by ADMET and synthesis proxies, and reviewed in a protein-ligand 3D interface.

The low raw-attempt validity is an important limitation and should be stated directly. It reflects an aggressive validity and novelty gate rather than a broad unconstrained generator. The final candidate set is therefore high-confidence as a filtered computational pool, but the system still needs architectural upgrades before claiming benchmark-level raw generative performance.

The present system should be described as pocket/electronic-conditioned and pocket-scored. It should not be described as a full protein-conditioned diffusion model. A future V6 system should learn residue-level pocket-ligand embeddings from richer protein-ligand datasets and should condition generation on aligned seed poses and residue-specific interaction targets.

## 5. Limitations

The pipeline has not been experimentally validated. Docking scores are approximate and are not binding free energies. Pocket-electronic scores are proxy scores, not quantum electron-density results. The receptor profile is global for the JAK2 pocket rather than seed-pose-specific. The current SE(3) component is auxiliary evidence and is not the primary generator of final molecular coordinates in this branch.

## 6. Conclusion

ElectroMacroDiff V5.5 is a complete low-resource computational discovery workflow for JAK2 macrocycle candidate prioritization. It produces a large ranked candidate set, preserves raw attempt evidence, performs GPU docking, integrates pocket-electronic scoring, and provides a real-data 3D dashboard for protein-ligand inspection. The project is ready for team evaluation as a computational pipeline and project artifact. The next scientific step is V6: residue-level protein-conditioned generation using learned interaction encoders and richer protein-ligand datasets.

## References

Trott, O.; Olson, A. J. AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. Journal of Computational Chemistry, 2010, 31, 455-461. DOI: 10.1002/jcc.21334.

Ding, J.; Tang, S.; et al. Vina-GPU 2.0: Further accelerating AutoDock Vina and its derivatives with graphics processing units. Journal of Chemical Information and Modeling, 2023. DOI: 10.1021/acs.jcim.2c01504.

RDKit: Open-source cheminformatics. https://www.rdkit.org

Rego, N.; Koes, D. 3Dmol.js: molecular visualization with WebGL. Bioinformatics, 2015, 31, 1322-1324.
