# V5.3 Vina-GPU 2.1 Candidate Decision Package

## Screening Summary

- GPU docking coverage: `114 / 114` parsed Vina-GPU 2.1 scores.
- Best Vina score: `-11.4` kcal/mol.
- Median Vina score: `-8.3` kcal/mol.
- Pose sanity checks: `10 / 10` passed for the inspected top poses.
- Scores are docking proxies, not experimental affinities.

## Recommended Selection

- Primary candidates: CAND_5152217faa, CAND_396f994be3, CAND_598172bd63, CAND_9c6324c16c, CAND_3c64f44e76.
- Backup candidates: CAND_d3037c6bbb, CAND_dbfbb7d1cf, CAND_f134535edd, CAND_9b2dbc98ed, CAND_1675c88065.

## Top Candidate Table

| Rank | Candidate | Vina | Pose | Final | PocketFit | PocketFinal | PocketDecision | ADMET | Synth | QED | SA | Risk | Tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CAND_5152217faa | -11.4 | pass | 0.861 | 0.803 | 0.848 | pass | 0.608 | 0.594 | 0.349 | 4.653 | 1_lipinski_violations | final_candidate |
| 2 | CAND_396f994be3 | -11.4 | pass | 0.844 | 0.822 | 0.839 | pass | 0.581 | 0.521 | 0.29 | 5.315 | 1_lipinski_violations | final_candidate |
| 3 | CAND_598172bd63 | -11.3 | pass | 0.844 | 0.811 | 0.837 | pass | 0.607 | 0.539 | 0.346 | 5.153 | 1_lipinski_violations | final_candidate |
| 4 | CAND_9c6324c16c | -11.3 | pass | 0.843 | 0.809 | 0.835 | pass | 0.592 | 0.55 | 0.314 | 5.054 | 1_lipinski_violations | final_candidate |
| 5 | CAND_3c64f44e76 | -11.3 | pass | 0.837 | 0.814 | 0.832 | pass | 0.586 | 0.519 | 0.299 | 5.326 | 1_lipinski_violations | final_candidate |
| 6 | CAND_d3037c6bbb | -10.9 | pass | 0.813 | 0.804 | 0.811 | pass | 0.613 | 0.538 | 0.36 | 5.159 | 1_lipinski_violations | backup_candidate |
| 7 | CAND_dbfbb7d1cf | -11.2 | pass | 0.808 | 0.803 | 0.807 | pass | 0.479 | 0.522 | 0.283 | 5.298 | 2_lipinski_violations | backup_candidate |
| 8 | CAND_f134535edd | -11 | pass | 0.815 | 0.768 | 0.805 | review | 0.597 | 0.52 | 0.325 | 5.321 | 1_lipinski_violations | backup_candidate |
| 9 | CAND_9b2dbc98ed | -11 | pass | 0.81 | 0.767 | 0.801 | review | 0.581 | 0.509 | 0.288 | 5.415 | 1_lipinski_violations | backup_candidate |
| 10 | CAND_1675c88065 | -11 | pass | 0.825 | 0.707 | 0.799 | review | 0.613 | 0.566 | 0.36 | 4.909 | 1_lipinski_violations | backup_candidate |

## Interpretation

The top candidates combine Vina-GPU score, pose sanity, ADMET/synthesis proxies, diversity-aware ranking, and the new pocket/electronic-fit proxy. The pocket-fit score uses PDBQT partial charges and distance-based interaction checks; it is not a quantum electron-density calculation.

## Suggested Next Validation

1. Visually inspect the top five docked poses in PyMOL or ChimeraX.
2. Re-dock the top ten with an independent seed/config or CPU Vina for reproducibility.
3. Check protonation/tautomer states for the top five before making any potency claim.
4. Run a higher-fidelity rescoring method only on the top five if compute is available.

## Top Candidate SMILES
- `1` `CAND_5152217faa`: `Fc1cc2cc(F)c1CN1CCOCC1CCCC1(CCNCC1)n1cc(cn1)-c1cnc3cccc-2c3n1`
- `2` `CAND_396f994be3`: `CN1CCN2c3ccc(cn3)-c3cnc4cccc(c4n3)-c3cc(F)c(c(F)c3)CN3CCS(=O)(=O)CC3CCCC2C1`
- `3` `CAND_598172bd63`: `Fc1cc2cc(F)c1CN1CCOCC1CCCC1CNCCC1n1cc(cn1)-c1cnc3cccc-2c3n1`
- `4` `CAND_9c6324c16c`: `CN1CCN2c3ccc(cc3)-c3cnc4cccc(c4n3)-c3cc(F)c(c(F)c3)CN3CCOCC3COCC2C1`
- `5` `CAND_3c64f44e76`: `CN1CCN2c3ccc(cn3)-c3cnc4cccc(c4n3)-c3cc(F)c(c(F)c3)CN3CCS(=O)(=O)CC3COCC2C1`
- `6` `CAND_d3037c6bbb`: `Fc1cc2cc(F)c1CN1CCOCC1CNCC1CNCCC1n1cc(cn1)-c1cnc3cccc-2c3n1`
- `7` `CAND_dbfbb7d1cf`: `CN1CCN2c3ccc(cc3)-c3cnc4cccc(c4n3)-c3cc(F)c(c(F)c3)CN3CCOCC3CSCC2C1`
- `8` `CAND_f134535edd`: `CN1CCN2c3ccc(cn3)-c3cnc4cccc(c4n3)-c3cc(F)c(c(F)c3)CN3CCS(=O)(=O)CC3CNCC2C1`
- `9` `CAND_9b2dbc98ed`: `CN1CCN2c3ccc(cn3)-c3cnc4cccc(c4n3)-c3cc(F)c(c(F)c3)CN3CCS(=O)(=O)CC3CSCC2C1`
- `10` `CAND_1675c88065`: `Fc1cc2cc(F)c1CN1CCOC(CNCC3(CCNCC3)n3cc(cn3)-c3cnc4cccc-2c4n3)C1`
