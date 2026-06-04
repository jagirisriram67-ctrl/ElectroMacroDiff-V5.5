# V5.4 Pocket-Electronic Scoring Archive

This folder preserves the V5.4 scoring layer. V5.4 did not replace the generator. It added JAK2 pocket-electronic analysis and rescoring over the generated/docked V5.3 candidate branch.

What V5.4 contributed:

- JAK2 pocket electronic profile from the prepared receptor
- pocket-electronic fit scores for generated candidates
- pocket-aware ranking table
- scoring script and pocket-electronics module

Key files:

```text
docs/V5_4_POCKET_ELECTRONIC_GUIDANCE_PLAN.md
scripts/21_score_pocket_electronics.py
src/emd_v5_2_hybrid/pocket_electronics.py
06_docking/v5_3_model_guided/scores/jak2_pocket_electronic_profile.json
06_docking/v5_3_model_guided/scores/pocket_electronic_fit_scores.csv
08_final_ranking/v5_3_model_guided_pocket_electronic_ranked_candidates.csv
```

This stage is important because it marks the transition from docking-only ranking toward pocket-electronic prioritization. The later V5.5 root implementation then moved pocket context into active generation decisions.
