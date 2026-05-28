# ElectroMacroDiff V5.5 Final Handover Report

Date: 2026-05-16

## Executive Summary

ElectroMacroDiff V5.5 is a JAK2-focused macrocycle generation, docking, ADMET proxy scoring, pocket-electronic rescoring, and interactive 3D review pipeline. The project is now suitable for handover as a computational discovery package with transparent metrics and reproducible local outputs.

The strongest result is not a claim of experimental activity. The strongest result is a complete evidence chain: generated candidates, real Vina-GPU docking outputs, ranked pocket-electronic scores, ADMET/synthesis proxy tables, validation checks, and an interactive 3D protein-ligand dashboard based only on project artifacts.

## Real Project Metrics

| Metric | Value |
|---|---:|
| Curated JAK2 ligands | 1,135 |
| Generated V5.5 candidates | 1,113 |
| Ranked V5.5 candidates | 1,113 |
| Docked candidates with parsed scores | 1,036 |
| Raw pose files | 1,036 |
| Pocket-scored candidates | 200 |
| Best Vina-GPU score | -12.7 kcal/mol |
| Median Vina-GPU score | -8.4 kcal/mol |
| Best pocket-electronic fit score | 0.84838 |
| Median pocket-electronic fit score | 0.783945 |
| Top ranked candidate | CAND_3688f8acdc |
| Top pocket-guided final score | 0.874199 |
| Raw generation attempts | 87,287 |
| Raw-attempt validity | 1.2751% |
| Linker novelty | 100% |
| Uniqueness | 100% |
| Macrocyclization | 100% |
| Validation checks | 54/54 passed |
| Unit tests | 40/40 passed |

## Top Ranked Candidate

Candidate `CAND_3688f8acdc` is the current top V5.5 ranked molecule.

| Field | Value |
|---|---|
| Candidate ID | CAND_3688f8acdc |
| Parent molecule | JAK2_85fd848de0 |
| Best Vina-GPU score | -12.7 kcal/mol |
| Pocket-electronic fit score | 0.810776 |
| Pocket-guided final score | 0.874199 |
| Molecular weight | 490.608 |
| LogP | 4.7517 |
| QED | 0.4132 |
| SA score | 4.306 |
| Lipinski violations | 0 |
| Max ring size | 16 |
| Macrocycle flag | true |
| Pose decision | pass |

SMILES:

```text
CN1CCN(c2cc3cc(c2)Nc2nc4c(cccn4n2)-c2ccc(F)c(c2)CSCOC3)CC1
```

## V5.5 Evidence Branch

The active benchmark branch is `kaggle_v5_5_pocket_guided`.

Attempt-status counts:

| Status | Count |
|---|---:|
| rejected_by_reward_model | 78,257 |
| rejected_known_linker | 5,230 |
| filtered_non_macrocycle_or_basic_filters | 2,108 |
| duplicate_or_invalid_record | 357 |
| ring_closure_failed | 221 |
| insufficient_anchors | 1 |
| valid_output | 1,113 |

Interpretation: the reward model aggressively rejects proposals before final output. This improves the final survivor pool but lowers raw-attempt validity. The raw validity number is therefore reported honestly and should not be presented as equal to filtered-survivor validity.

## JAK2 Pocket Evidence

The pocket profile is a proxy electronic and contact profile derived from prepared receptor and docking outputs. It is not a quantum electron-density map.

| Pocket feature | Value |
|---|---:|
| Pocket atom count | 227 |
| Pocket net charge proxy | -0.357 |
| Hydrophobic atom count | 45 |
| H-bond donor atom count | 59 |
| H-bond acceptor atom count | 72 |
| Aromatic atom count | 13 |

Top contact residues include `ARG:A:980`, `TYR:A:931`, `ASN:A:981`, `ARG:A:938`, `ASN:A:859`, `LEU:A:932`, `ASP:A:939`, `ASP:A:994`, `LEU:A:855`, `LYS:A:882`, `MET:A:929`, and `SER:A:936`.

## 3D Dashboard Upgrade

The interface now includes a stronger 3D protein-ligand review module:

- direct ranked candidate list with click-to-open behavior
- 1,036 embedded real docked pose texts
- real JAK2 receptor PDB embedded in `data.js`
- protein plus docked-pose view mode
- pose-only and protein-only modes
- ligand stick, ball-and-stick, sphere, line, and cross styles
- adjustable bond thickness
- key-atom, heavy-atom, all-atom, bond-length, contact-residue, and atom-plus-contact label modes
- configurable label density
- transparent ligand VDW shell
- transparent pocket contact surface
- optional docking grid box
- real contact-residue highlighting

The viewer does not generate or infer new molecular structures. It displays only the receptor and pose artifacts exported from the project.

## Acceptance Status

Accepted as a computational handover package:

- end-to-end generation and scoring artifacts exist
- docking and pose outputs are present
- ranked candidate tables exist
- dashboard reads from real exported data
- validation and unit tests pass
- limitations are explicitly stated

Not accepted as experimental validation:

- no wet-lab binding assay has been performed
- no molecular dynamics stability validation has been run
- no absolute binding free energy calculation has been run
- the current system is pocket/electronic-conditioned and pocket-scored, not a full residue-level protein-conditioned diffusion generator

## Recommended Next Step

The next technical phase should be V6: residue-level protein-ligand interaction learning using richer protein-ligand datasets such as PDBbind, PLINDER/BioLiP-style contact data, and JAK2 activity data from BindingDB/ChEMBL. V6 should replace the current global pocket vector with a learned pocket interaction encoder.
