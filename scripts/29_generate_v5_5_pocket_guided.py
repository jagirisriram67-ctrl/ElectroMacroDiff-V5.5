"""Generate V5.5 pocket-guided macrocycles with policy gating.

This branch keeps V5.3 frozen. It writes only to the V5.5 sidecar folder and
logs every attempt so raw validity can be measured honestly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="V5.5 pocket-guided macrocycle generation.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed-count", type=int, default=250)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--max-products-per-seed", type=int, default=20)
    parser.add_argument("--top-anchor-atoms", type=int, default=12)
    parser.add_argument("--min-anchor-probability", type=float, default=0.05)
    parser.add_argument("--reward-threshold", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--attempt-log-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--anchor-checkpoint", default=None)
    parser.add_argument("--linker-checkpoint", default=None)
    parser.add_argument("--reward-checkpoint", default=None)
    parser.add_argument("--allow-fallback-models", action="store_true")
    parser.add_argument("--disable-linker-novelty", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.seed_count = min(args.seed_count, 5)
        args.max_products_per_seed = min(args.max_products_per_seed, 3)

    import pandas as pd
    import torch
    from rdkit import Chem

    from emd_v5_2_hybrid.anchor_gnn import graph_from_smiles_for_anchor_gnn
    from emd_v5_2_hybrid.baseline_generation import (
        _linker_atom_numbers,
        _macrocycle_anchor_indices,
        _try_bridge_atoms,
        candidate_record,
        write_candidates_csv,
    )
    from emd_v5_2_hybrid.device_helper import device_summary, resolve_device
    from emd_v5_2_hybrid.generation_metrics import write_generation_metrics
    from emd_v5_2_hybrid.linker_size_features import LINKER_FEATURE_COLUMNS, feature_vector, molecule_feature_map
    from emd_v5_2_hybrid.macrocycle_fragmentation import fragment_macrocycle_record
    from emd_v5_2_hybrid.pocket_anchor_gnn import load_pocket_anchor_gnn
    from emd_v5_2_hybrid.pocket_features import load_pocket_feature_vector
    from emd_v5_2_hybrid.pocket_linker_policy import CHEMOTYPE_LABELS, LINKER_SIZE_LABELS, load_pocket_linker_policy
    from emd_v5_2_hybrid.validity_reward import attempt_to_feature_vector, load_validity_reward

    base = Path(args.base).resolve()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(json.dumps(device_summary(device), indent=2))

    folder_name = "v5_5_pocket_guided_smoke" if args.smoke_test else "v5_5_pocket_guided"
    output_dir = base / "05_generated_candidates" / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv).resolve() if args.output_csv else output_dir / "generated_v5_5_pocket_guided.csv"
    attempt_log_csv = Path(args.attempt_log_csv).resolve() if args.attempt_log_csv else output_dir / "v5_5_pocket_guided_attempt_log.csv"
    summary_json = Path(args.summary_json).resolve() if args.summary_json else output_dir / "v5_5_pocket_guided_summary.json"
    metrics_csv = output_csv.with_name(f"{output_csv.stem}_metrics.csv")

    pocket_json = base / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
    pocket_vector = load_pocket_feature_vector(pocket_json)
    pocket_tensor = torch.tensor(pocket_vector, dtype=torch.float32, device=device)

    anchor_path = Path(args.anchor_checkpoint).resolve() if args.anchor_checkpoint else base / "04_models_checkpoints" / "v5_5_pocket_anchor_gnn" / "pocket_anchor_gnn_best.pt"
    linker_path = Path(args.linker_checkpoint).resolve() if args.linker_checkpoint else base / "04_models_checkpoints" / "v5_5_pocket_linker_policy" / "pocket_linker_policy_best.pt"
    reward_path = Path(args.reward_checkpoint).resolve() if args.reward_checkpoint else base / "04_models_checkpoints" / "v5_5_validity_reward" / "validity_reward_best.pt"
    anchor_model, anchor_checkpoint = load_required_or_fallback(load_pocket_anchor_gnn, anchor_path, device, "PocketAnchorGNN", args.allow_fallback_models)
    linker_model, size_labels, chemotype_labels, linker_checkpoint = load_linker_required_or_fallback(linker_path, device, args.allow_fallback_models)
    reward_model, reward_checkpoint = load_required_or_fallback(load_validity_reward, reward_path, device, "ValidityReward", args.allow_fallback_models)

    curated_csv = base / "02_curated_data" / "jak2_curated_ligands.csv"
    if not curated_csv.exists():
        raise FileNotFoundError(curated_csv)
    seeds = pd.read_csv(curated_csv)
    if "pActivity" in seeds.columns:
        seeds = seeds.sort_values("pActivity", ascending=False)
    seeds = seeds.iloc[args.seed_offset : args.seed_offset + args.seed_count]
    training_inchikeys = set(seeds.get("inchikey", pd.Series(dtype=str)).dropna().astype(str))
    training_linkers = set() if args.disable_linker_novelty else load_training_linkers(base)
    print(f"Seeds: {len(seeds)} | training linkers: {len(training_linkers)}")

    rng = random.Random(args.random_seed)
    records: list[dict] = []
    attempts: list[dict] = []
    seen_inchikeys: set[str] = set()
    start_time = time.time()

    for seed_index, seed_row in enumerate(seeds.to_dict(orient="records"), start=1):
        seed_smiles = str(seed_row.get("canonical_smiles", ""))
        parent_id = str(seed_row.get("mol_id", f"SEED_{seed_index}"))
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is None:
            attempts.append({"parent_mol_id": parent_id, "seed_smiles": seed_smiles, "status": "invalid_seed_smiles"})
            continue

        graph = graph_from_smiles_for_anchor_gnn(seed_smiles, atom_feature_dim=64)
        if graph is None:
            attempts.append({"parent_mol_id": parent_id, "seed_smiles": seed_smiles, "status": "graph_build_failed"})
            continue

        if anchor_model is None:
            anchor_probs = torch.ones(graph["num_nodes"], dtype=torch.float32) * 0.5
        else:
            with torch.no_grad():
                anchor_probs = torch.sigmoid(
                    anchor_model(
                        graph["node_features"].to(device),
                        graph["edge_index"].to(device),
                        graph["edge_features"].to(device),
                        pocket_tensor,
                    )
                ).cpu()

        anchor_candidates = _macrocycle_anchor_indices(mol)
        scored_anchors = [(idx, float(anchor_probs[idx])) for idx in anchor_candidates if idx < len(anchor_probs)]
        scored_anchors.sort(key=lambda item: item[1], reverse=True)
        top_anchors = [(idx, score) for idx, score in scored_anchors[: args.top_anchor_atoms] if score >= args.min_anchor_probability]
        if len(top_anchors) < 2:
            attempts.append({"parent_mol_id": parent_id, "seed_smiles": seed_smiles, "status": "insufficient_anchors"})
            continue

        produced = 0
        for (atom_a, score_a), (atom_b, score_b) in combinations(top_anchors, 2):
            if produced >= args.max_products_per_seed:
                break
            try:
                path_bonds = len(Chem.GetShortestPath(mol, int(atom_a), int(atom_b))) - 1
            except Exception:
                path_bonds = 0
            if path_bonds < 3 or path_bonds > 20:
                continue

            feature_map = molecule_feature_map(mol, desired_ring_size=path_bonds + 5, atom_a=atom_a, atom_b=atom_b)
            linker_features = feature_vector(feature_map, LINKER_FEATURE_COLUMNS)
            feature_tensor = torch.tensor([linker_features], dtype=torch.float32, device=device)

            if linker_model is None:
                size_choices = rng.sample(range(len(LINKER_SIZE_LABELS)), min(3, len(LINKER_SIZE_LABELS)))
                chemotype_choices = rng.sample(range(len(CHEMOTYPE_LABELS)), min(3, len(CHEMOTYPE_LABELS)))
                size_probs = [0.5 for _ in LINKER_SIZE_LABELS]
                chemotype_probs = [0.5 for _ in CHEMOTYPE_LABELS]
                size_labels = LINKER_SIZE_LABELS
                chemotype_labels = CHEMOTYPE_LABELS
            else:
                with torch.no_grad():
                    size_logits, chemotype_logits = linker_model(feature_tensor, pocket_tensor)
                    size_distribution = torch.softmax(size_logits[0], dim=0).cpu()
                    chemotype_distribution = torch.softmax(chemotype_logits[0], dim=0).cpu()
                size_choices = torch.topk(size_distribution, min(3, len(size_distribution))).indices.tolist()
                chemotype_choices = torch.topk(chemotype_distribution, min(3, len(chemotype_distribution))).indices.tolist()
                size_probs = size_distribution.tolist()
                chemotype_probs = chemotype_distribution.tolist()

            for size_index in size_choices:
                for chemotype_index in chemotype_choices:
                    if produced >= args.max_products_per_seed:
                        break
                    linker_length = int(size_labels[size_index])
                    chemotype = str(chemotype_labels[chemotype_index])
                    target_ring = path_bonds + linker_length
                    if target_ring < 12 or target_ring > 24:
                        continue
                    size_prob = float(size_probs[size_index])
                    chemotype_prob = float(chemotype_probs[chemotype_index])
                    policy_probability = size_prob * chemotype_prob
                    attempt = {
                        "parent_mol_id": parent_id,
                        "seed_smiles": seed_smiles,
                        "atom_a": int(atom_a),
                        "atom_b": int(atom_b),
                        "anchor_score_a": round(float(score_a), 6),
                        "anchor_score_b": round(float(score_b), 6),
                        "path_bonds": int(path_bonds),
                        "linker_length": linker_length,
                        "target_ring": int(target_ring),
                        "chemotype": chemotype,
                        "size_probability": round(size_prob, 6),
                        "chemotype_probability": round(chemotype_prob, 6),
                        "linker_probability": round(policy_probability, 6),
                        "policy_probability": round(policy_probability, 6),
                    }

                    if reward_model is not None:
                        reward_features = attempt_to_feature_vector(attempt, pocket_vector)
                        with torch.no_grad():
                            reward_prob = float(
                                torch.sigmoid(
                                    reward_model(torch.tensor([reward_features], dtype=torch.float32, device=device))
                                ).item()
                            )
                        attempt["reward_probability"] = round(reward_prob, 6)
                        if reward_prob < args.reward_threshold:
                            attempt["status"] = "rejected_by_reward_model"
                            attempts.append(attempt)
                            continue
                    else:
                        attempt["reward_probability"] = 1.0

                    product = _try_bridge_atoms(mol, int(atom_a), int(atom_b), _linker_atom_numbers(linker_length, chemotype))
                    if product is None:
                        attempt["status"] = "ring_closure_failed"
                        attempts.append(attempt)
                        continue

                    smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
                    attempt["generated_smiles"] = smiles
                    record = candidate_record(
                        smiles,
                        source_generator="v5_5_pocket_guided",
                        parent_mol_id=parent_id,
                        training_inchikeys=training_inchikeys,
                    )
                    if not record or record["inchikey"] in seen_inchikeys:
                        attempt["status"] = "duplicate_or_invalid_record"
                        attempts.append(attempt)
                        continue
                    if not record["has_macrocycle_12_20"] or not record["passes_basic_filters"]:
                        attempt["status"] = "filtered_non_macrocycle_or_basic_filters"
                        attempts.append(attempt)
                        continue
                    if training_linkers:
                        extracted = {
                            str(fragment.get("linker_smiles"))
                            for fragment in fragment_macrocycle_record(
                                {"mol_id": record["candidate_id"], "canonical_smiles": smiles},
                                max_pairs_per_molecule=50,
                                min_ring_size=12,
                                max_ring_size_allowed=24,
                            )
                            if fragment.get("linker_smiles")
                        }
                        known = extracted & training_linkers
                        attempt["extracted_linker_smiles"] = ";".join(sorted(extracted))
                        if known:
                            attempt["known_linker_smiles"] = ";".join(sorted(known))
                            attempt["status"] = "rejected_known_linker"
                            attempts.append(attempt)
                            continue

                    seen_inchikeys.add(record["inchikey"])
                    records.append(record)
                    attempt["candidate_id"] = record["candidate_id"]
                    attempt["inchikey"] = record["inchikey"]
                    attempt["status"] = "valid_output"
                    attempts.append(attempt)
                    produced += 1

        if seed_index == 1 or seed_index % 25 == 0:
            print(f"Seed {seed_index}/{len(seeds)} | candidates={len(records)} | attempts={len(attempts)}")

    write_candidates_csv(records, output_csv)
    pd.DataFrame(attempts).to_csv(attempt_log_csv, index=False)
    write_generation_metrics(output_csv, metrics_csv)
    status_counts = dict(sorted(Counter(str(row.get("status", "")) for row in attempts).items()))
    valid_rows = status_counts.get("valid_output", 0)
    summary = {
        "status": "completed",
        "generation_type": "v5_5_pocket_guided",
        "smoke_test": bool(args.smoke_test),
        "device": device,
        "total_attempts": len(attempts),
        "valid_candidates": len(records),
        "raw_attempt_validity_percent": round(valid_rows / max(len(attempts), 1) * 100.0, 4),
        "attempt_status_counts": status_counts,
        "reward_threshold": args.reward_threshold,
        "linker_novelty_enforced": not args.disable_linker_novelty,
        "allow_fallback_models": bool(args.allow_fallback_models),
        "models": {
            "anchor_checkpoint": None if anchor_checkpoint is None else str(anchor_path),
            "linker_checkpoint": None if linker_checkpoint is None else str(linker_path),
            "reward_checkpoint": None if reward_checkpoint is None else str(reward_path),
        },
        "output_csv": str(output_csv),
        "attempt_log_csv": str(attempt_log_csv),
        "metrics_csv": str(metrics_csv),
        "total_time_seconds": round(time.time() - start_time, 1),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def load_required_or_fallback(loader, path: Path, device: str, name: str, allow_fallback: bool):
    if path.exists():
        print(f"Loaded {name}: {path}")
        return loader(path, device=device)
    if allow_fallback:
        print(f"WARNING: {name} missing; using explicit fallback because --allow-fallback-models was set.")
        return None, None
    raise FileNotFoundError(f"Missing required {name} checkpoint: {path}")


def load_linker_required_or_fallback(path: Path, device: str, allow_fallback: bool):
    from emd_v5_2_hybrid.pocket_linker_policy import CHEMOTYPE_LABELS, LINKER_SIZE_LABELS, load_pocket_linker_policy

    if path.exists():
        print(f"Loaded PocketLinkerPolicy: {path}")
        return load_pocket_linker_policy(path, device=device)
    if allow_fallback:
        print("WARNING: PocketLinkerPolicy missing; using explicit uniform fallback.")
        return None, LINKER_SIZE_LABELS, CHEMOTYPE_LABELS, None
    raise FileNotFoundError(f"Missing required PocketLinkerPolicy checkpoint: {path}")


def load_training_linkers(base: Path) -> set[str]:
    import pandas as pd

    linkers: set[str] = set()
    for name in ["v5_3_pretrain_fragment_linker_pairs.csv", "v5_3_macrocycle_fragment_linker_pairs.csv"]:
        path = base / "02_curated_data" / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=lambda column: column == "linker_smiles")
        if "linker_smiles" in frame.columns:
            linkers.update(frame["linker_smiles"].dropna().astype(str))
    return linkers


if __name__ == "__main__":
    main()
