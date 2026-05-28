from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run, save_progress
from emd_v5_2_hybrid.se3_dataset import graph_from_smiles
from emd_v5_2_hybrid.se3_flow import SE3FlowConfig, SE3FlowMatching, flow_matching_loss
from emd_v5_2_hybrid.train_se3 import collate_graphs, move_batch


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Install torch for SE(3) geometry scoring") from exc
    return torch


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Install pandas for SE(3) geometry scoring") from exc
    return pd


def safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def score_candidate_geometry(
    model,
    smiles: str,
    candidate_id: str,
    device: str,
    atom_feature_dim: int,
    num_repeats: int,
    random_seed: int,
    row_index: int,
) -> dict[str, object]:
    torch = _torch()
    graph = graph_from_smiles(
        smiles,
        mol_id=candidate_id,
        atom_feature_dim=atom_feature_dim,
        random_seed=random_seed,
    )
    if graph is None:
        return {
            "candidate_id": candidate_id,
            "canonical_smiles": smiles,
            "se3_geometry_loss": None,
            "se3_geometry_confidence": None,
            "se3_geometry_status": "embed_failed",
        }

    batch = move_batch(collate_graphs([graph]), device)
    losses: list[float] = []
    with torch.no_grad():
        for repeat_index in range(num_repeats):
            seed_value = int(random_seed + row_index * 1000 + repeat_index)
            torch.manual_seed(seed_value)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
            loss = flow_matching_loss(model, batch)
            losses.append(float(loss.item()))

    mean_loss = sum(losses) / max(len(losses), 1)
    confidence = 1.0 / (1.0 + max(mean_loss, 0.0))
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "se3_geometry_loss": round(mean_loss, 6),
        "se3_geometry_confidence": round(confidence, 6),
        "se3_geometry_status": "scored",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score auxiliary SE(3) geometry confidence for generated candidates.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--ranking-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-ranking-csv", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means score all candidates in the ranking CSV.")
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--stage", default="V5_3_se3_geometry_scoring")
    args = parser.parse_args()

    pd = _pandas()
    torch = _torch()

    base = Path(args.base).resolve()
    ranking_path = (
        Path(args.ranking_csv).resolve()
        if args.ranking_csv
        else (
            base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates.csv"
            if (base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates.csv").exists()
            else base / "08_final_ranking" / "v5_3_model_guided_ranked_candidates.csv"
        )
    )
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else base / "04_models_checkpoints" / "se3_flow" / "se3_best_checkpoint.pt"
    )
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else base / "06_docking" / "v5_3_model_guided" / "scores" / "se3_geometry_scores.csv"
    )
    output_ranking_csv = (
        Path(args.output_ranking_csv).resolve()
        if args.output_ranking_csv
        else base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates_with_se3.csv"
    )
    progress_path = base / "00_project_registry" / "progress_v5_3_se3_geometry_scoring.json"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ranking = pd.read_csv(ranking_path)
    if args.max_candidates > 0:
        ranking = ranking.head(args.max_candidates).copy()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    atom_feature_dim = int(config.get("atom_feature_dim", 64))
    hidden_dim = int(config.get("hidden_dim", 128))
    num_layers = int(config.get("num_layers", 4))
    model = SE3FlowMatching(
        SE3FlowConfig(
            atom_feature_dim=atom_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    score_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(ranking.to_dict(orient="records")):
        score_rows.append(
            score_candidate_geometry(
                model=model,
                smiles=str(row.get("canonical_smiles", row.get("smiles", ""))),
                candidate_id=str(row.get("candidate_id", "")),
                device=device,
                atom_feature_dim=atom_feature_dim,
                num_repeats=max(int(args.num_repeats), 1),
                random_seed=int(args.random_seed),
                row_index=row_index,
            )
        )

    scores = pd.DataFrame(score_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_csv, index=False)

    merged = pd.read_csv(ranking_path)
    merged = merged.drop(columns=[col for col in ["se3_geometry_loss", "se3_geometry_confidence", "se3_geometry_status"] if col in merged.columns])
    merged = merged.merge(
        scores[["candidate_id", "se3_geometry_loss", "se3_geometry_confidence", "se3_geometry_status"]],
        on="candidate_id",
        how="left",
    )
    output_ranking_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_ranking_csv, index=False)

    scored = scores[scores["se3_geometry_status"].astype(str) == "scored"].copy()
    summary = {
        "stage": args.stage,
        "status": "completed",
        "ranking_csv": str(ranking_path),
        "checkpoint": str(checkpoint_path),
        "output_csv": str(output_csv),
        "output_ranking_csv": str(output_ranking_csv),
        "device": device,
        "max_candidates": int(args.max_candidates),
        "num_repeats": max(int(args.num_repeats), 1),
        "scored_rows": int(len(scored)),
        "total_rows": int(len(scores)),
        "median_se3_geometry_loss": None if scored.empty else round(float(scored["se3_geometry_loss"].median()), 6),
        "median_se3_geometry_confidence": None if scored.empty else round(float(scored["se3_geometry_confidence"].median()), 6),
        "best_se3_geometry_confidence": None if scored.empty else round(float(scored["se3_geometry_confidence"].max()), 6),
        "note": "SE(3) geometry confidence is auxiliary evidence only and does not modify the weighted project ranking.",
    }
    save_progress(progress_path, summary)

    for path, artifact_type in [
        (output_csv, "se3_geometry_scores"),
        (output_ranking_csv, "pocket_guided_ranking_with_se3_geometry"),
        (progress_path, "se3_geometry_scoring_progress"),
    ]:
        register_artifact(base, args.stage, path, artifact_type, owner="Student 1")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=f"{ranking_path}; {checkpoint_path}",
        output_path=f"{output_csv}; {output_ranking_csv}",
        molecules_in=len(ranking),
        molecules_out=len(scored),
        notes=f"device={device}; median_confidence={summary['median_se3_geometry_confidence']}",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
