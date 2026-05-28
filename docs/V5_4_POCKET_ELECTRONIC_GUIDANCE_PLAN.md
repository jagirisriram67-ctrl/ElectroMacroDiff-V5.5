# V5.4 Pocket And Electronic Guidance Plan

The project's main aim is not only to generate valid macrocycles. The real aim is to generate JAK2-pocket-fitted molecules: candidates whose shape, anchor placement, electronic pattern, hydrogen-bond opportunities, hydrophobic packing, and final docked pose are compatible with the `5AEP` binding pocket.

## What Was Missing

V5.3 had strong engineering pieces:

- JAK2 ligand curation.
- `5AEP` receptor and `QUP`-centered docking grid.
- Anchor/linker macrocycle generation.
- SE(3) geometry model evidence.
- Vina-GPU docking for all `114` model-guided candidates.
- Pose sanity and final ranking.

But the generation/ranking loop was still mostly molecule-internal:

- Anchor models saw ligand atom features, not the receptor pocket.
- Linker prediction saw macrocycle/core features, not the protein pocket.
- Ranking used docking score and crude pose sanity, but not residue interactions or electrostatic complementarity.
- Electron density was not represented. Only partial-charge/electronic proxies existed.

## New Practical Layer Added

The new pocket/electronic scoring layer uses the available receptor and pose artifacts:

- Receptor PDBQT partial charges from `06_docking/receptor/jak2_prepared.pdbqt`.
- Docked ligand PDBQT partial charges from `06_docking/v5_3_model_guided/poses/`.
- Distance-based contact geometry.
- Hydrogen-bond donor/acceptor opportunity.
- Hydrophobic packing.
- Aromatic contact opportunities.
- Favorable versus unfavorable electrostatic contacts.
- Hard clash penalty.

This produces:

- `06_docking/v5_3_model_guided/scores/pocket_electronic_fit_scores.csv`
- `06_docking/v5_3_model_guided/scores/jak2_pocket_electronic_profile.json`
- `08_final_ranking/v5_3_model_guided_pocket_electronic_ranked_candidates.csv`

Important boundary:

This is an electrostatic and interaction proxy, not a quantum electron-density map. True electron-density or electrostatic-potential surfaces would require a quantum workflow such as xTB/DFT/ESP charge generation for the top candidates.

## Current Pocket/Electronic Result

After scoring all `114` V5.3 Vina-GPU poses:

- Scored candidates: `114 / 114`.
- Best pocket/electronic-fit score: `0.844276`.
- Median pocket/electronic-fit score: `0.8146025`.
- Best candidate by pocket/electronic fit alone: `CAND_69ce89568d`.
- Original rank 1, `CAND_5152217faa`, remains pocket-guided rank 1 because it combines the best docking/ranking score with acceptable pocket/electronic fit.

Pocket profile from the current top pose:

- Pocket atoms within local interaction shell: `235`.
- Net pocket charge proxy: approximately neutral, `-0.011`.
- Hydrophobic atom count: `45`.
- H-bond donor atom count: `60`.
- H-bond acceptor atom count: `75`.
- Aromatic atom count: `15`.
- Major contact residues include `ARG:A:980`, `ARG:A:938`, `TYR:A:931`, `ASN:A:981`, `ASP:A:939`, `ASP:A:994`, `LEU:A:855`, and `LEU:A:932`.

## How This Should Guide The Next Generation Cycle

The next generation cycle should optimize for a composite target, not only validity or Vina score:

```text
generation_acceptance_score =
  anchor_model_score
+ linker_model_score
+ macrocycle_validity
+ predicted_property_score
+ docking_score_after_pose
+ pocket_electronic_fit_score
+ interaction_recovery_score
- clash_penalty
- overfit_similarity_penalty
```

For the current codebase, the practical route is:

1. Generate with the GNN anchor model and 16-feature linker model.
2. Dock generated candidates with Vina-GPU.
3. Run `scripts/21_score_pocket_electronics.py`.
4. Select candidates using `pocket_guided_final_score`, not raw Vina score alone.
5. Feed the best pocket-fit anchor/linker patterns back into generation settings.

## True Electron-Density Upgrade Path

The current proxy should become a two-tier electronic workflow:

Tier 1, fast for all candidates:

- PDBQT/Gasteiger partial charges.
- Donor/acceptor/hydrophobic/aromatic interaction fingerprints.
- Pocket-electronic-fit score.

Tier 2, slow for top 10 only:

- Generate protonation/tautomer states.
- Optimize docked ligand pose locally.
- Run xTB or DFT-lite partial charge/ESP calculation.
- Compare ligand electrostatic potential against pocket charge field.
- Add an explicit `xtb_esp_fit_score` column.

This keeps the project Colab-realistic while still respecting the original ElectroMacroDiff theme.

## Commands

Run after Vina-GPU docking and ranking:

```powershell
python scripts/21_score_pocket_electronics.py --base .
python scripts/19_build_v5_3_gpu_decision_package.py --base . --top-n 10
python scripts/08_validate_project.py --base .
python -m unittest discover -s tests
```

## Honest Claim

Use:

> ElectroMacroDiff now includes a pocket/electronic-fit feedback layer that scores docked candidates by JAK2 pocket contact geometry, partial-charge complementarity, hydrogen-bond opportunity, hydrophobic packing, aromatic contacts, and clash penalties.

Do not use:

> ElectroMacroDiff computes true electron density for all generated molecules.
