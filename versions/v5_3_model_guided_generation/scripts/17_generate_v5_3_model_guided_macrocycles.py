from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.baseline_generation import (
    _linker_atom_numbers,
    _macrocycle_anchor_indices,
    _try_bridge_atoms,
    candidate_record,
    linker_chemotypes,
    write_candidates_csv,
)
from emd_v5_2_hybrid.anchor_gnn import AnchorMessageGNN, graph_from_mol_for_anchor_gnn
from emd_v5_2_hybrid.generation_metrics import write_generation_metrics
from emd_v5_2_hybrid.macrocycle_fragmentation import (
    _distance_to_largest_ring,
    anchor_atom_feature_row,
    extract_linker_smiles_from_smiles,
    largest_ring_atoms,
)
from emd_v5_2_hybrid.linker_size_features import (
    LINKER_FEATURE_COLUMNS,
    feature_vector,
    molecule_feature_map,
)
from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress


ANCHOR_FEATURE_COLUMNS = [
    "atomic_num",
    "degree",
    "formal_charge",
    "is_aromatic",
    "is_ring",
    "total_h",
    "hybridization_sp",
    "hybridization_sp2",
    "hybridization_sp3",
    "mass",
    "gasteiger_charge",
    "num_rotatable_neighbors",
    "shortest_path_to_ring",
    "is_between_rings",
    "neighbor_heteroatom_count",
    "is_terminal_chain_atom",
    "local_connectivity_index",
    "ring_size_of_nearest_ring",
]

def _rdkit():
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Install rdkit for V5.3 model-guided macrocycle generation") from exc
    return Chem


def load_training_linkers(paths: list[Path]) -> set[str]:
    import pandas as pd

    linkers: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "linker_smiles" not in frame.columns:
            continue
        linkers.update(frame["linker_smiles"].dropna().astype(str))
    return linkers


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Install torch for V5.3 model-guided macrocycle generation") from exc
    return torch, nn


def make_anchor_model(input_dim: int, hidden_dim: int):
    _torch_mod, nn = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, 1),
    )


def make_linker_model(input_dim: int, hidden_dim: int, output_dim: int):
    _torch_mod, nn = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, output_dim),
    )


def load_anchor_model(path: Path, device: str):
    torch, _nn = _torch()
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model_type") == "anchor_message_gnn":
        model = AnchorMessageGNN(
            node_dim=int(checkpoint.get("node_dim", 64)),
            edge_dim=int(checkpoint.get("edge_dim", 6)),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint.get("num_layers", 4)),
            dropout=float(checkpoint.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint, ["__anchor_message_gnn__"]
    feature_columns = checkpoint.get("feature_columns", ANCHOR_FEATURE_COLUMNS)
    model = make_anchor_model(len(feature_columns), int(checkpoint["hidden_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, feature_columns


def load_linker_model(path: Path, device: str):
    torch, _nn = _torch()
    checkpoint = torch.load(path, map_location=device)
    feature_columns = checkpoint.get("feature_columns", LINKER_FEATURE_COLUMNS)
    labels = [int(v) for v in checkpoint["labels"]]
    model = make_linker_model(len(feature_columns), int(checkpoint["hidden_dim"]), len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels, checkpoint, feature_columns


def atom_feature_map(
    mol,
    atom,
    ring_atoms: set[int],
    rings: list[tuple[int, ...]],
    ring_memberships: dict[int, set[int]],
    distance_to_ring: dict[int, int],
) -> dict[str, float]:
    return anchor_atom_feature_row(mol, atom, ring_atoms, rings, ring_memberships, distance_to_ring)


def score_anchor_atoms(mol, model, device: str, top_n: int, feature_columns: list[str]) -> list[tuple[int, float]]:
    torch, _nn = _torch()
    Chem = _rdkit()
    candidate_indices = _macrocycle_anchor_indices(mol)
    if not candidate_indices:
        return []
    if getattr(model, "is_anchor_gnn", False):
        graph = graph_from_mol_for_anchor_gnn(mol, atom_feature_dim=getattr(model, "node_dim", 64))
        with torch.no_grad():
            logits = model(
                graph["node_features"].to(device),
                graph["edge_index"].to(device),
                graph["edge_features"].to(device),
            )
            probabilities_by_atom = torch.sigmoid(logits).detach().cpu().tolist()
        ranked = sorted(
            ((atom_idx, float(probabilities_by_atom[atom_idx])) for atom_idx in candidate_indices),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_n]
    try:
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass
    ring_size, ring_atoms = largest_ring_atoms(mol)
    rings = [tuple(int(idx) for idx in ring) for ring in mol.GetRingInfo().AtomRings()]
    ring_memberships: dict[int, set[int]] = {}
    for ring_id, ring in enumerate(rings):
        for atom_idx in ring:
            ring_memberships.setdefault(int(atom_idx), set()).add(ring_id)
    distance_to_ring = _distance_to_largest_ring(mol, ring_atoms)
    features = []
    feature_maps = {
        atom_idx: atom_feature_map(
            mol,
            mol.GetAtomWithIdx(atom_idx),
            ring_atoms,
            rings,
            ring_memberships,
            distance_to_ring,
        )
        for atom_idx in candidate_indices
    }
    for atom_idx in candidate_indices:
        values = feature_maps[atom_idx]
        features.append([float(values.get(column, 0.0)) for column in feature_columns])
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device)
        probabilities = torch.sigmoid(model(x)).view(-1).detach().cpu().tolist()
    ranked = sorted(zip(candidate_indices, probabilities), key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def linker_features(mol, desired_ring_size: int, atom_a: int | None = None, atom_b: int | None = None) -> dict[str, float]:
    return molecule_feature_map(mol, desired_ring_size=desired_ring_size, atom_a=atom_a, atom_b=atom_b)


def predict_linker_lengths_for_pair(
    mol,
    model,
    labels: list[int],
    feature_columns: list[str],
    device: str,
    atom_a: int,
    atom_b: int,
) -> dict[int, tuple[int, float]]:
    torch, _nn = _torch()
    desired_sizes = list(range(12, 21))
    features = [
        feature_vector(linker_features(mol, desired_ring_size=size, atom_a=atom_a, atom_b=atom_b), feature_columns)
        for size in desired_sizes
    ]
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=device)
        probabilities = torch.softmax(model(x), dim=1).detach().cpu()
    predictions: dict[int, tuple[int, float]] = {}
    for row_index, desired_ring_size in enumerate(desired_sizes):
        best_index = int(probabilities[row_index].argmax().item())
        predictions[desired_ring_size] = (int(labels[best_index]), float(probabilities[row_index, best_index].item()))
    return predictions


def ranked_pair_options(
    mol,
    anchor_scores: list[tuple[int, float]],
    linker_model,
    linker_labels: list[int],
    linker_feature_columns: list[str],
    device: str,
    min_anchor_probability: float,
):
    Chem = _rdkit()
    score_by_atom = dict(anchor_scores)
    options = []
    eligible_atoms = [atom for atom, score in anchor_scores if score >= min_anchor_probability]
    for atom_a, atom_b in combinations(eligible_atoms, 2):
        try:
            path = Chem.GetShortestPath(mol, atom_a, atom_b)
        except Exception:
            continue
        path_bonds = max(len(path) - 1, 0)
        if path_bonds < 4:
            continue
        linker_predictions = predict_linker_lengths_for_pair(
            mol,
            linker_model,
            linker_labels,
            linker_feature_columns,
            device,
            atom_a=atom_a,
            atom_b=atom_b,
        )
        for desired_ring_size, (predicted_linker_length, linker_probability) in linker_predictions.items():
            actual_ring_size = path_bonds + predicted_linker_length
            if actual_ring_size != desired_ring_size:
                continue
            if actual_ring_size < 12 or actual_ring_size > 20:
                continue
            pair_score = score_by_atom[atom_a] * score_by_atom[atom_b] * linker_probability
            options.append(
                {
                    "atom_a": atom_a,
                    "atom_b": atom_b,
                    "path_bonds": path_bonds,
                    "linker_length": predicted_linker_length,
                    "target_ring": actual_ring_size,
                    "anchor_score_a": score_by_atom[atom_a],
                    "anchor_score_b": score_by_atom[atom_b],
                    "linker_probability": linker_probability,
                    "pair_score": pair_score,
                }
            )
    return sorted(options, key=lambda row: row["pair_score"], reverse=True)


def generate_model_guided_macrocycles(
    seed_rows: list[dict],
    anchor_model,
    linker_model,
    linker_labels: list[int],
    device: str,
    training_inchikeys: set[str],
    top_anchor_atoms: int,
    min_anchor_probability: float,
    max_products_per_seed: int,
    random_seed: int,
    anchor_feature_columns: list[str],
    linker_feature_columns: list[str],
    chemotype_set: str,
    enforce_linker_novelty: bool,
    novelty_mode: str,
    training_linkers: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    Chem = _rdkit()
    rng = random.Random(random_seed)
    records: list[dict] = []
    attempt_rows: list[dict] = []
    seen: set[str] = set()
    training_linkers = training_linkers or set()
    for row in seed_rows:
        seed_smiles = str(row.get("canonical_smiles", ""))
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is None:
            attempt_rows.append(
                {
                    "parent_mol_id": str(row.get("mol_id", "")),
                    "seed_smiles": seed_smiles,
                    "status": "seed_invalid_smiles",
                }
            )
            continue
        parent_id = str(row.get("mol_id", ""))
        anchor_scores = score_anchor_atoms(mol, anchor_model, device, top_n=top_anchor_atoms, feature_columns=anchor_feature_columns)
        pair_options = ranked_pair_options(
            mol,
            anchor_scores,
            linker_model,
            linker_labels,
            linker_feature_columns,
            device,
            min_anchor_probability,
        )
        if not anchor_scores:
            attempt_rows.append({"parent_mol_id": parent_id, "seed_smiles": seed_smiles, "status": "no_anchor_candidates"})
        if anchor_scores and not pair_options:
            attempt_rows.append({"parent_mol_id": parent_id, "seed_smiles": seed_smiles, "status": "no_valid_anchor_pair"})
        produced = 0
        for option in pair_options:
            if produced >= max_products_per_seed:
                break
            chemotypes = linker_chemotypes(chemotype_set)
            rng.shuffle(chemotypes)
            for chemotype in chemotypes:
                if produced >= max_products_per_seed:
                    break
                attempt = {
                    "parent_mol_id": parent_id,
                    "seed_smiles": seed_smiles,
                    "atom_a": int(option["atom_a"]),
                    "atom_b": int(option["atom_b"]),
                    "path_bonds": int(option["path_bonds"]),
                    "linker_length": int(option["linker_length"]),
                    "target_ring": int(option["target_ring"]),
                    "anchor_score_a": round(float(option["anchor_score_a"]), 6),
                    "anchor_score_b": round(float(option["anchor_score_b"]), 6),
                    "linker_probability": round(float(option["linker_probability"]), 6),
                    "chemotype": chemotype,
                }
                product = _try_bridge_atoms(
                    mol,
                    int(option["atom_a"]),
                    int(option["atom_b"]),
                    _linker_atom_numbers(int(option["linker_length"]), chemotype),
                )
                if product is None:
                    attempt["status"] = "ring_closure_failed"
                    attempt_rows.append(attempt)
                    continue
                smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
                attempt["generated_smiles"] = smiles
                note = (
                    f"v5_3_model_guided_anchor_pair={option['atom_a']}-{option['atom_b']};"
                    f"anchor_scores={option['anchor_score_a']:.3f},{option['anchor_score_b']:.3f};"
                    f"linker_length={option['linker_length']};"
                    f"target_ring={option['target_ring']};"
                    f"linker_prob={option['linker_probability']:.3f};"
                    f"chemotype={chemotype}"
                )
                record = candidate_record(
                    smiles,
                    source_generator="v5_3_model_guided_macrocycle",
                    parent_mol_id=parent_id,
                    training_inchikeys=training_inchikeys,
                    note=note,
                )
                if not record or record["inchikey"] in seen:
                    attempt["status"] = "duplicate_or_invalid_record"
                    attempt_rows.append(attempt)
                    continue
                if not record["has_macrocycle_12_20"] or not record["passes_basic_filters"]:
                    attempt["status"] = "filtered_non_macrocycle_or_basic_filters"
                    attempt_rows.append(attempt)
                    continue
                if enforce_linker_novelty:
                    if novelty_mode != "exact":
                        raise ValueError(f"Unsupported novelty mode: {novelty_mode}")
                    linker_smiles = extract_linker_smiles_from_smiles(
                        str(record["canonical_smiles"]),
                        mol_id=str(record["candidate_id"]),
                        max_pairs_per_molecule=50,
                        min_ring_size=12,
                        max_ring_size_allowed=24,
                    )
                    attempt["extracted_linker_smiles"] = ";".join(linker_smiles)
                    if not linker_smiles:
                        attempt["status"] = "rejected_no_extractable_linker"
                        attempt_rows.append(attempt)
                        continue
                    known_linkers = sorted(smiles for smiles in linker_smiles if smiles in training_linkers)
                    if known_linkers:
                        attempt["known_linker_smiles"] = ";".join(known_linkers)
                        attempt["status"] = "rejected_known_linker"
                        attempt_rows.append(attempt)
                        continue
                seen.add(record["inchikey"])
                records.append(record)
                attempt["candidate_id"] = record["candidate_id"]
                attempt["inchikey"] = record["inchikey"]
                attempt["status"] = "valid_output"
                attempt_rows.append(attempt)
                produced += 1
    return records, attempt_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate V5.3 model-guided macrocycle candidates.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--seed-count", type=int, default=250)
    parser.add_argument("--seed-offset", type=int, default=0, help="Skip this many potency-sorted JAK2 seeds before generation.")
    parser.add_argument("--max-products-per-seed", type=int, default=12)
    parser.add_argument("--top-anchor-atoms", type=int, default=10)
    parser.add_argument("--min-anchor-probability", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--anchor-checkpoint", default=None)
    parser.add_argument("--linker-checkpoint", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--metrics-csv", default=None)
    parser.add_argument("--attempt-log-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--skip-merged-output", action="store_true", help="Use for independent Colab chunks that will be merged later.")
    parser.add_argument("--chemotype-set", default="legacy", choices=["legacy", "diverse"])
    parser.add_argument("--enforce-linker-novelty", action="store_true")
    parser.add_argument("--novelty-mode", default="exact", choices=["exact"])
    parser.add_argument("--training-fragment-csv", action="append", default=None)
    args = parser.parse_args()

    import pandas as pd
    torch, _nn = _torch()

    base = Path(args.base).resolve()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    curated_csv = base / "02_curated_data" / "jak2_curated_ligands.csv"
    anchor_checkpoint = (
        Path(args.anchor_checkpoint).resolve()
        if args.anchor_checkpoint
        else base / "04_models_checkpoints" / "v5_3_anchor" / "anchor_site_model.pt"
    )
    linker_checkpoint = (
        Path(args.linker_checkpoint).resolve()
        if args.linker_checkpoint
        else base / "04_models_checkpoints" / "v5_3_linker_size" / "linker_size_model.pt"
    )
    default_output_csv = base / "05_generated_candidates" / "model_guided_macrocycle" / "generated_v5_3_model_guided_macrocycles.csv"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else default_output_csv
    metrics_csv = (
        Path(args.metrics_csv).resolve()
        if args.metrics_csv
        else (
            base / "05_generated_candidates" / "model_guided_macrocycle" / "v5_3_model_guided_generation_metrics.csv"
            if output_csv == default_output_csv
            else output_csv.with_name(f"{output_csv.stem}_metrics.csv")
        )
    )
    attempt_log_csv = (
        Path(args.attempt_log_csv).resolve()
        if args.attempt_log_csv
        else (
            output_csv.with_name("v5_3_model_guided_generation_attempt_log.csv")
            if output_csv == default_output_csv
            else output_csv.with_name(f"{output_csv.stem}_attempt_log.csv")
        )
    )
    merged_csv = base / "05_generated_candidates" / "merged" / "generated_merged_with_v5_3_model_guided.csv"
    merged_metrics_csv = base / "05_generated_candidates" / "merged" / "generation_comparison_metrics_with_v5_3_model_guided.csv"
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else (
            base / "00_project_registry" / "progress_v5_3_model_guided_generation.json"
            if output_csv == default_output_csv
            else output_csv.with_name(f"{output_csv.stem}_summary.json")
        )
    )
    training_fragment_paths = (
        [Path(path).resolve() for path in args.training_fragment_csv]
        if args.training_fragment_csv
        else [
            base / "02_curated_data" / "v5_3_pretrain_fragment_linker_pairs.csv",
            base / "02_curated_data" / "v5_3_macrocycle_fragment_linker_pairs.csv",
        ]
    )

    curated = pd.read_csv(curated_csv)
    sorted_seeds = curated.sort_values("p_activity", ascending=False).reset_index(drop=True)
    seed_offset = max(0, int(args.seed_offset))
    seed_end = seed_offset + max(0, int(args.seed_count))
    seeds = sorted_seeds.iloc[seed_offset:seed_end].to_dict(orient="records")
    training_inchikeys = set(curated["inchikey"].dropna().astype(str))
    training_linkers = load_training_linkers(training_fragment_paths) if args.enforce_linker_novelty else set()
    anchor_model, anchor_payload, anchor_feature_columns = load_anchor_model(anchor_checkpoint, device)
    linker_model, linker_labels, linker_payload, linker_feature_columns = load_linker_model(linker_checkpoint, device)

    records, attempt_rows = generate_model_guided_macrocycles(
        seeds,
        anchor_model=anchor_model,
        linker_model=linker_model,
        linker_labels=linker_labels,
        device=device,
        training_inchikeys=training_inchikeys,
        top_anchor_atoms=args.top_anchor_atoms,
        min_anchor_probability=args.min_anchor_probability,
        max_products_per_seed=args.max_products_per_seed,
        random_seed=args.random_seed,
        anchor_feature_columns=anchor_feature_columns,
        linker_feature_columns=linker_feature_columns,
        chemotype_set=args.chemotype_set,
        enforce_linker_novelty=bool(args.enforce_linker_novelty),
        novelty_mode=args.novelty_mode,
        training_linkers=training_linkers,
    )
    write_candidates_csv(records, output_csv)
    write_generation_metrics(output_csv, metrics_csv)
    attempt_log_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(attempt_rows).to_csv(attempt_log_csv, index=False)

    existing_merged = base / "05_generated_candidates" / "merged" / "generated_merged_filtered.csv"
    if args.skip_merged_output:
        merged_csv = output_csv
        merged_metrics_csv = metrics_csv
    elif existing_merged.exists():
        existing_frame = pd.read_csv(existing_merged)
        guided_frame = pd.read_csv(output_csv)
        merged_frame = pd.concat([existing_frame, guided_frame], ignore_index=True)
        if not merged_frame.empty and "inchikey" in merged_frame.columns:
            merged_frame = merged_frame.drop_duplicates("inchikey")
        merged_frame.to_csv(merged_csv, index=False)
        write_generation_metrics(merged_csv, merged_metrics_csv)
    else:
        merged_csv = output_csv
        merged_metrics_csv = metrics_csv

    guided_count = int(len(pd.read_csv(output_csv)))
    merged_count = int(len(pd.read_csv(merged_csv)))
    macrocycle_count = int(pd.read_csv(output_csv)["has_macrocycle_12_20"].astype(bool).sum()) if guided_count else 0
    attempt_status_counts = dict(sorted(Counter(str(row.get("status", "")) for row in attempt_rows).items()))
    summary = {
        "stage": "V5_3_model_guided_macrocycle_generation",
        "status": "completed",
        "device": device,
        "seed_offset": seed_offset,
        "seed_end_exclusive": seed_end,
        "seed_count": len(seeds),
        "top_anchor_atoms": args.top_anchor_atoms,
        "min_anchor_probability": args.min_anchor_probability,
        "max_products_per_seed": args.max_products_per_seed,
        "chemotype_set": args.chemotype_set,
        "enforce_linker_novelty": bool(args.enforce_linker_novelty),
        "novelty_mode": args.novelty_mode,
        "training_fragment_paths": [str(path) for path in training_fragment_paths if path.exists()],
        "training_linker_count": int(len(training_linkers)),
        "anchor_checkpoint": str(anchor_checkpoint),
        "anchor_hidden_dim": int(anchor_payload["hidden_dim"]),
        "anchor_feature_columns": anchor_feature_columns,
        "linker_checkpoint": str(linker_checkpoint),
        "linker_hidden_dim": int(linker_payload["hidden_dim"]),
        "linker_labels": linker_labels,
        "linker_feature_columns": linker_feature_columns,
        "output_csv": str(output_csv),
        "attempt_log_csv": str(attempt_log_csv),
        "metrics_csv": str(metrics_csv),
        "merged_csv": str(merged_csv),
        "merged_metrics_csv": str(merged_metrics_csv),
        "generated_candidates": guided_count,
        "generated_macrocycles_12_20": macrocycle_count,
        "raw_attempt_rows": int(len(attempt_rows)),
        "valid_attempt_rows": int(sum(1 for row in attempt_rows if row.get("status") == "valid_output")),
        "raw_attempt_validity_percent": round(
            float(sum(1 for row in attempt_rows if row.get("status") == "valid_output") / max(len(attempt_rows), 1) * 100.0),
            4,
        ),
        "attempt_status_counts": attempt_status_counts,
    }
    save_progress(summary_json, summary)
    register_artifact(base, "V5_3_model_guided_generation", output_csv, "model_guided_macrocycle_candidates", owner="Student 4")
    register_artifact(base, "V5_3_model_guided_generation", attempt_log_csv, "model_guided_generation_attempt_log", owner="Student 4")
    register_artifact(base, "V5_3_model_guided_generation", metrics_csv, "model_guided_generation_metrics", owner="Student 1")
    register_artifact(base, "V5_3_model_guided_generation", merged_csv, "merged_candidate_csv_with_model_guided", owner="Student 1")
    register_run(
        base,
        stage="V5_3_model_guided_generation",
        status="completed",
        input_path=f"{curated_csv}; {anchor_checkpoint}; {linker_checkpoint}",
        output_path=str(output_csv),
        molecules_in=len(seeds),
        molecules_out=guided_count,
        random_seed=args.random_seed,
        notes=f"model_guided_macrocycles={guided_count}; merged_total={merged_count}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
