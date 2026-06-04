# ElectroMacroDiff V5.2 Hybrid Final Report Draft

## Abstract

ElectroMacroDiff V5.2 Hybrid is a low-resource, checkpointed computational drug-discovery pipeline for JAK2-focused macrocycle/constrained inhibitor candidate generation and prioritization. The workflow combines curated JAK2 ligand data, RDKit descriptor/graph preparation, a custom SE(3)-aware flow matching debug-trained generator path, RDKit/SELFIES baseline generation, AutoDock Vina docking, ADMET/synthesis proxy scoring, and consensus ranking.

## Scientific Boundary

The proposed molecules are computational hypotheses for experimental follow-up. Docking scores, ADMET proxies, and synthesizability proxies are prioritization signals, not experimental proof of potency, selectivity, safety, or synthesis.

## Dataset And Preparation

- Curated JAK2 ligands: 204
- Ligand feature rows: 204
- Primary receptor: PDB 5AEP, JAK2 kinase domain
- Docking grid source: co-crystallized QUP ligand in 5AEP

## Candidate Generation

- Filtered generated candidates: 595

| source_generator | num_candidates | num_unique_inchikeys | novel_fraction | basic_filter_pass_fraction | macrocycle_fraction | constrained_ring_fraction | median_mw | median_logp | median_qed | median_sa_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macrocycle_linker | 2935 | 2935 | 1.0 | 1.0 | 1.0 | 0.0 | 516.572 | 4.2853 | 0.3898 | 4.975 |
| rdkit | 1249 | 1249 | 0.989592 | 1.0 | 0.008006 | 0.0 | 446.408 | 3.5401 | 0.4817 | 3.403 |
| selfies | 2590 | 2590 | 0.976062 | 1.0 | 0.113514 | 0.270656 | 407.4595 | 2.4584 | 0.4747 | 3.436 |
| ALL | 6774 | 6774 | 0.988928 | 1.0 | 0.478152 | 0.103484 | 463.588 | 3.4306 | 0.4282 | 4.097 |

## SE(3) Model Status

The custom SE(3)-aware flow-matching implementation passed the critical debug gate: graph loading, batch collation, forward pass, finite loss, backward pass, optimizer step, checkpoint save, and checkpoint reload. In the current sprint it should be presented as a working architecture/debug-trained generator path unless a longer training run is added.

## Docking

- Vina-scored ligands: 148 / 148
- Best Vina score: -9.186 kcal/mol
- Median Vina score: -6.452999999999999 kcal/mol
- Weakest parsed Vina score: -3.526 kcal/mol

## Final Diversity-Selected Top Candidates

### Rank 1: CAND_baa376c0dd

- Source: selfies
- SMILES: `C=Cc1ccccc1C(N)C=C1C=NC2=NN1CN1CCN(CC1)c1ccc(cc1F)N2`
- Docking score: -9.186 kcal/mol
- Final weighted score: 0.922314
- Decision: primary_candidate
- Main risk: no_major_descriptor_risk

### Rank 2: CAND_ab5598f4f0

- Source: selfies
- SMILES: `ClCCNC1=C=Nn2cccc2C2C=CC(=NN2)NC=N1`
- Docking score: -8.01 kcal/mol
- Final weighted score: 0.855059
- Decision: primary_candidate
- Main risk: no_major_descriptor_risk

### Rank 3: CAND_cc70ad3aa7

- Source: selfies
- SMILES: `NC1SN2C=CC3=NC2=C1c1cc(ncn1)NCC=CC=C(F)C=C3`
- Docking score: -7.648 kcal/mol
- Final weighted score: 0.836428
- Decision: primary_candidate
- Main risk: no_major_descriptor_risk

### Rank 4: CAND_daf82c369e

- Source: selfies
- SMILES: `CC1C=NC=C(F)CN=c2nc(Cl)ccc2=C(NC=C2N=CCCN2)N=CN1`
- Docking score: -7.584 kcal/mol
- Final weighted score: 0.822479
- Decision: primary_candidate
- Main risk: no_major_descriptor_risk

### Rank 5: CAND_3d3d2ca880

- Source: selfies
- SMILES: `CS(=O)(=O)n1cnnc(NN=CF)ncccccc2cccnc21`
- Docking score: -7.396 kcal/mol
- Final weighted score: 0.82203
- Decision: primary_candidate
- Main risk: no_major_descriptor_risk

## Validation

- Validation checks passed: 48 / 48
- Failed checks: 0

## Limitations

- Vina docking approximates pose/affinity and must be followed by pose inspection.
- ADMET and safety estimates are descriptor/proxy based.
- SE(3) model currently has debug-gate evidence; extended training would strengthen the research claim.
- Candidate synthesis and biological activity require wet-lab validation.

## References To Cite

- AutoDock Vina manual and releases: https://vina.scripps.edu/manual/
- AutoDock Vina GitHub releases: https://github.com/ccsb-scripps/AutoDock-Vina/releases
- Meeko documentation: https://meeko.readthedocs.io/
