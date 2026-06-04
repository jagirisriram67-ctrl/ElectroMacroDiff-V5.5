# ElectroMacroDiff V6 Sequential Premium-Dataset Training Plan

## Locked Direction

V6 is the premium-data upgrade after the V5.5 pocket/electronic-conditioned proof.
It must be trained one lane at a time, with the four Kaggle accounts used as clean
role-based environments rather than uncontrolled parallel jobs.

V5.5 stays frozen. V6 writes new artifacts under `v6_*` paths only.

## Current V5.5 Evidence Baseline

| Metric | Value |
|---|---:|
| Anchor GNN F1 | 0.4848 |
| Anchor GNN precision | 0.3478 |
| Linker exact accuracy | 0.5928 |
| Linker within-one accuracy | 0.8181 |
| Validity reward ROC-AUC | 0.8605 |
| SE(3) continuation val loss | 7.4328 |
| V5.5 generated candidates | 1113 |
| V5.5 raw-attempt validity | 1.2751 |
| V5.5 linker novelty | 100.0000 |

## Dataset Readiness

| Dataset | Role | Status |
|---|---|---|
| `chembl_bindingdb_jak` | Activity and reward-model supervision for JAK/JAK2 ligands. | ready |
| `pdbbind` | Protein-ligand affinity and pose-quality supervision. | not mounted / incomplete |
| `plinder` | Residue-level protein-ligand interaction embedding and split control. | not mounted / incomplete |
| `biolip2` | Biologically relevant contact and binding-site labels. | not mounted / incomplete |
| `crossdocked2020` | Large-scale pocket-conditioned generation and pose robustness. | not mounted / incomplete |

## Four Sequential Lanes

### account_1_activity_reward

- Dataset: `chembl_bindingdb_jak`
- Goal: Train JAK/JAK2 activity and affinity-prior reward models.
- Acceptance: ROC-AUC or PR-AUC improves over V5.5 validity-only reward evidence.
- Run style: smoke test first, then long training if the smoke output is valid.
- Output rule: checkpoint and metrics must use `v6_` names and must not overwrite V5.5.

Smoke command:

```bash
!python scripts/33_train_v6_activity_reward.py --base . --device auto --smoke-test --epochs 2 --max-rows 300
```

Long command:

```bash
!python scripts/33_train_v6_activity_reward.py --base . --device auto --epochs 800 --batch-size 64 --learning-rate 5e-4 --patience 120
```

### account_2_pdbbind_pose_affinity

- Dataset: `pdbbind`
- Goal: Train protein-ligand pose/affinity encoder on high-quality complexes.
- Acceptance: Validation Pearson/Spearman and pose-quality classification are reported.
- Run style: smoke test first, then long training if the smoke output is valid.
- Output rule: checkpoint and metrics must use `v6_` names and must not overwrite V5.5.

### account_3_plinder_biolip_contacts

- Dataset: `plinder, biolip2`
- Goal: Train residue-level contact and pocket interaction encoder.
- Acceptance: Residue-contact AUC/F1 beats pocket-count baseline.
- Run style: smoke test first, then long training if the smoke output is valid.
- Output rule: checkpoint and metrics must use `v6_` names and must not overwrite V5.5.

### account_4_v6_generator_distillation

- Dataset: `crossdocked2020 plus prior lanes`
- Goal: Distill learned pocket encoder into anchor/linker/generation policies.
- Acceptance: Logged generation improves raw validity, docking, and pocket-fit metrics.
- Run style: smoke test first, then long training if the smoke output is valid.
- Output rule: checkpoint and metrics must use `v6_` names and must not overwrite V5.5.

## Kaggle Notebook Contract

Each notebook should contain only these sections:

1. Setup and mounted dataset check.
2. `python scripts/31_prepare_v6_premium_manifest.py --base . ...`.
3. Smoke run for that lane.
4. Long run for that lane.
5. Zip only the lane outputs and metrics.

Do not put giant dataset downloads in the notebook. Mount/upload the dataset as a Kaggle
dataset first, then pass the mounted path to the manifest script.

## V6 Claim Boundary

Allowed after successful V6 training:

- V6 uses learned protein-ligand and residue-contact evidence to condition generation.
- V6 improves the V5.5 pocket/electronic-conditioned policy with premium structural data.

Not allowed until a true generative model is trained and benchmarked:

- Full protein-conditioned diffusion has beaten MED.
- Quantum electron density was used during generation.
- Experimental potency or biological activity is proven.
