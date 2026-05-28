# Execution Checklist

## Day 1 - Setup And Tiny Pipeline

- Create Drive folder structure.
- Validate `campaign_config.yaml`.
- Install requirements in Colab.
- Run environment check.
- Confirm Tiny Debug mode writes progress JSON.
- Confirm project registry receives a run row.

## Day 2 - Data Collection And Curation

- Query ChEMBL for human JAK2 activity records.
- Save raw data before cleaning.
- Download PDB `5AEP`.
- Canonicalize and deduplicate SMILES.
- Convert activity units to nM.
- Write curated ligand table.

## Day 3 - Features And SE(3) Debug

- Compute RDKit descriptors.
- Generate conformers.
- Build graph tensors.
- Load one graph batch.
- Run forward pass, loss, backward pass, optimizer step.
- Save and reload checkpoint.

## Day 4 - SE(3) Training And Baselines

- Train SE(3) if Day 3 gate passed.
- Save latest and best checkpoints.
- Run RDKit and SELFIES baseline generation.
- Write generation metrics.

## Day 5 - Inference And Merge

- Run SE(3) inference or architecture-demo output.
- Merge SE(3), SELFIES, and RDKit candidates.
- Remove duplicates and training-set overlaps.
- Filter by validity, MW, logP, TPSA, QED, ring size.

## Day 6-7 - Docking

- Prepare receptor.
- Dock known controls.
- Dock 50-150 candidates.
- Save scores and top pose images.

## Day 8 - ADMET And Synthesis

- Compute ADMET/drug-likeness scores.
- Run PAINS/Brenk filters where available.
- Compute SA score or fallback synthesis proxy.
- Write safety proxy notes honestly.

## Day 9 - Ranking

- Merge docking, ADMET, synthesis, novelty, and diversity scores.
- Rank top 10.
- Select final 3-5 plus backup 5.

## Day 10 - Report

- Build molecule cards.
- Write TPP-style report.
- Produce final slides.
- Package audit folder.
