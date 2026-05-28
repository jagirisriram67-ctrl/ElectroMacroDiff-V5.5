"""V5.5 Training data builder for pocket-conditioned models.

Builds training datasets for all three V5.5 models from existing project artifacts:
  1. PocketAnchorGNN training data — anchor labels + pocket features
  2. PocketLinkerPolicy training data — linker size + chemotype labels + pocket features
  3. PocketValidityRewardModel training data — attempt logs + pocket features

Run this ONCE before the sequential Kaggle staircase. By default it writes a
small manifest and the reward table only; use ``--write-expanded-csvs`` if you
explicitly want pocket-feature-duplicated anchor/linker CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5.5 pocket-conditioned training datasets.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--write-expanded-csvs", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_rows = args.max_rows or 2000

    import pandas as pd
    from emd_v5_2_hybrid.pocket_features import pocket_feature_vector
    from emd_v5_2_hybrid.validity_reward import (
        attempt_label,
        attempt_to_feature_vector,
    )

    base = Path(args.base).resolve()
    output_dir = base / "03_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load pocket profile ----
    pocket_json = base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
    if not pocket_json.exists():
        print(f"ERROR: Pocket profile not found at {pocket_json}")
        sys.exit(1)
    pocket_profile = json.loads(pocket_json.read_text(encoding="utf-8"))
    pocket_vector = pocket_feature_vector(pocket_profile)
    print(f"Pocket feature vector: {len(pocket_vector)} dims")

    # ---- 1. Anchor training data with pocket features ----
    print("\n=== Building V5.5 Anchor Training Data ===")
    anchor_csv = base / "03_features" / "v5_3_anchor_atom_training.csv"
    if anchor_csv.exists():
        anchor_df = pd.read_csv(anchor_csv, nrows=args.max_rows) if args.max_rows else pd.read_csv(anchor_csv, nrows=5)
        print(f"  Source exists: {anchor_csv} ({len(anchor_df)} preview rows loaded)")
        if args.write_expanded_csvs:
            anchor_df = pd.read_csv(anchor_csv, nrows=args.max_rows) if args.max_rows else pd.read_csv(anchor_csv)
            for i, val in enumerate(pocket_vector):
                anchor_df[f"pocket_feat_{i}"] = val
            out_anchor = output_dir / "v5_5_pocket_anchor_training.csv"
            anchor_df.to_csv(out_anchor, index=False)
            print(f"  Wrote {len(anchor_df)} rows to {out_anchor}")
        else:
            print("  Skipped expanded anchor CSV; trainers load pocket vector directly.")
    else:
        print(f"  WARNING: {anchor_csv} not found, skipping anchor data")

    # Also build from pretrain data if available
    pretrain_anchor_csv = base / "03_features" / "v5_3_pretrain_anchor_atom_training.csv"
    if pretrain_anchor_csv.exists():
        preview = pd.read_csv(pretrain_anchor_csv, nrows=args.max_rows or 5)
        print(f"  Source exists: {pretrain_anchor_csv} ({len(preview)} preview rows loaded)")
        if args.write_expanded_csvs:
            pretrain_df = pd.read_csv(pretrain_anchor_csv, nrows=args.max_rows) if args.max_rows else pd.read_csv(pretrain_anchor_csv)
            for i, val in enumerate(pocket_vector):
                pretrain_df[f"pocket_feat_{i}"] = val
            out_pretrain = output_dir / "v5_5_pocket_anchor_pretrain.csv"
            pretrain_df.to_csv(out_pretrain, index=False)
            print(f"  Wrote {len(pretrain_df)} pretrain rows to {out_pretrain}")
        else:
            print("  Skipped expanded pretrain anchor CSV; this avoids duplicating a huge table.")
    else:
        print(f"  INFO: Pretrain anchor data not found (optional)")

    # ---- 2. Linker policy training data ----
    print("\n=== Building V5.5 Linker Policy Training Data ===")
    fragment_csv = base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv"
    if fragment_csv.exists():
        frag_df = pd.read_csv(fragment_csv, nrows=args.max_rows) if args.max_rows else pd.read_csv(fragment_csv)
        # Add chemotype labels based on linker SMILES content
        frag_df["chemotype_label"] = frag_df["linker_smiles"].apply(_infer_chemotype_from_linker)
        out_linker = output_dir / "v5_5_pocket_linker_training.csv"
        if args.write_expanded_csvs:
            for i, val in enumerate(pocket_vector):
                frag_df[f"pocket_feat_{i}"] = val
            frag_df.to_csv(out_linker, index=False)
            print(f"  Wrote {len(frag_df)} rows to {out_linker}")
        else:
            chemotype_manifest = output_dir / "v5_5_linker_chemotype_manifest.csv"
            frag_df[["fragment_id", "mol_id", "linker_smiles", "chemotype_label"]].to_csv(chemotype_manifest, index=False)
            print(f"  Wrote compact chemotype manifest to {chemotype_manifest}")
        print(f"  Chemotype distribution:")
        print(frag_df["chemotype_label"].value_counts().to_string(header=False))
    else:
        print(f"  WARNING: {fragment_csv} not found, skipping linker data")

    # ---- 3. Validity reward training data ----
    print("\n=== Building V5.5 Validity Reward Training Data ===")
    attempt_logs = [
        base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles_kaggle_logged_attempt_log.csv",
        base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles_kaggle_diverse_attempt_log.csv",
    ]

    all_reward_rows = []
    for log_path in attempt_logs:
        if not log_path.exists():
            print(f"  INFO: {log_path.name} not found, skipping")
            continue
        log_df = pd.read_csv(log_path, nrows=args.max_rows) if args.max_rows else pd.read_csv(log_path)
        print(f"  Processing {log_path.name}: {len(log_df)} attempts")

        # Filter to rows that have anchor scores (skip seed-level failures)
        valid_rows = log_df.dropna(subset=["anchor_score_a", "anchor_score_b"])
        for _, row in valid_rows.iterrows():
            attempt_dict = row.to_dict()
            features = attempt_to_feature_vector(attempt_dict, pocket_vector)
            label = attempt_label(str(row.get("status", "")))
            reward_row = {f"feat_{i}": v for i, v in enumerate(features)}
            reward_row["label_valid"] = label
            reward_row["status"] = str(row.get("status", ""))
            reward_row["source_log"] = log_path.name
            reward_row["parent_mol_id"] = str(row.get("parent_mol_id", ""))
            all_reward_rows.append(reward_row)

    if all_reward_rows:
        reward_df = pd.DataFrame(all_reward_rows)
        out_reward = output_dir / "v5_5_validity_reward_training.csv"
        reward_df.to_csv(out_reward, index=False)
        valid_count = int(reward_df["label_valid"].sum())
        total = len(reward_df)
        print(f"  Wrote {total} rows ({valid_count} valid, {total - valid_count} invalid)")
        print(f"  Positive rate: {valid_count / max(total, 1) * 100:.2f}%")
    else:
        print("  WARNING: No attempt logs found for reward model training")

    # ---- Summary ----
    print("\n=== V5.5 Training Data Build Complete ===")
    summary = {
        "pocket_profile": str(pocket_json),
        "pocket_feature_dim": len(pocket_vector),
        "smoke_test": bool(args.smoke_test),
        "max_rows": args.max_rows,
        "write_expanded_csvs": bool(args.write_expanded_csvs),
        "outputs": {
            "anchor": str(output_dir / "v5_5_pocket_anchor_training.csv"),
            "anchor_pretrain": str(output_dir / "v5_5_pocket_anchor_pretrain.csv"),
            "linker": str(output_dir / "v5_5_pocket_linker_training.csv"),
            "reward": str(output_dir / "v5_5_validity_reward_training.csv"),
        },
    }
    summary_path = output_dir / "v5_5_training_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


def _infer_chemotype_from_linker(smiles: str) -> str:
    """Infer the most likely chemotype from a linker SMILES string."""
    s = str(smiles) if smiles else ""
    # Count heteroatoms in the linker
    n_count = s.count("N") + s.count("n")
    o_count = s.count("O") + s.count("o")
    s_count = s.count("S") + s.count("s")

    if o_count >= 2 and n_count == 0 and s_count == 0:
        return "dioxa"
    if n_count >= 2 and o_count == 0 and s_count == 0:
        return "diaza"
    if s_count >= 2 and o_count == 0 and n_count == 0:
        return "dithio"
    if n_count >= 1 and o_count >= 1 and s_count == 0:
        if s.index("N") < s.index("O") if "N" in s and "O" in s else True:
            return "aza_oxa"
        return "oxa_aza"
    if o_count >= 1 and s_count >= 1:
        return "oxa_thio"
    if n_count >= 1 and s_count >= 1:
        return "aza_thio"
    if o_count == 1:
        return "oxa"
    if n_count == 1:
        return "aza"
    if s_count == 1:
        return "thioether"
    return "alkyl"


if __name__ == "__main__":
    main()
