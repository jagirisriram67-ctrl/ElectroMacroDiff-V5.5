# V5.5 Pocket-Guided Generation Results

V5.5 used the merged pocket-conditioned checkpoints:

- `PocketAnchorGNN`
- `PocketLinkerPolicy`
- `PocketValidityRewardModel`
- SE(3) continuation as auxiliary geometry evidence only

## Generation Summary

| Metric | V5.5 result |
|---|---:|
| Generated candidates | `1,113` |
| Raw attempt rows | `87,287` |
| Valid output rows | `1,113` |
| Raw-attempt validity | `1.2751%` |
| Uniqueness | `100.0%` |
| Macrocyclization | `100.0%` |
| Linker novelty | `100.0%` |
| Novel molecules | `100.0%` |
| Median QED | `0.3913` |
| Median SA proxy | `4.77` |

## Attempt Status Counts

| Status | Count |
|---|---:|
| `rejected_by_reward_model` | `78,257` |
| `rejected_known_linker` | `5,230` |
| `filtered_non_macrocycle_or_basic_filters` | `2,108` |
| `valid_output` | `1,113` |
| `duplicate_or_invalid_record` | `357` |
| `ring_closure_failed` | `221` |
| `insufficient_anchors` | `1` |

## Interpretation

V5.5 succeeded at the project-facing objective of using pocket/electronic context
during generation decisions and greatly expanded the number of valid novel
macrocycles from the V5.3 sidecar branches.

The conservative benchmark interpretation is:

- V5.5 exceeds MED-style uniqueness, macrocyclization, and linker novelty.
- V5.5 does **not** exceed MED raw-attempt validity.
- V5.5 should not be claimed as a fully benchmark-beating MED replacement.

The low raw validity is mostly caused by counting reward-model rejections as raw
attempts. That is honest and conservative, but for the next branch we should run
a threshold sweep and report both:

- proposal-level validity including reward rejections
- construction-level validity after reward gating

## Current Claim

Allowed:

> ElectroMacroDiff V5.5 conditions anchor selection, linker choice, and validity
> gating on JAK2 pocket-electronic context during generation, producing `1,113`
> unique, novel, macrocyclic candidates in the logged Kaggle branch.

Not allowed:

> ElectroMacroDiff V5.5 fully beats MED across all comparable metrics.

## Next Step

Dock the `1,113` V5.5 candidates, then run pocket-electronic scoring, SE(3)
auxiliary scoring, and final ranking for the V5.5 branch.
