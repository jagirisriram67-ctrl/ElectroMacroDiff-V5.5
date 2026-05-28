"""Consensus ranking utilities for final candidate prioritization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


DEFAULT_WEIGHTS = {
    "docking_score": 0.35,
    "pose_sanity": 0.15,
    "admet": 0.20,
    "synthesis": 0.15,
    "novelty_diversity": 0.10,
    "safety_proxy": 0.05,
}


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    docking_score: float | None = None
    pose_score: float = 0.5
    admet_score: float = 0.5
    synthesis_score: float = 0.5
    novelty_score: float = 0.5
    safety_proxy_score: float = 0.5


def minmax(values: Iterable[float], reverse: bool = False) -> list[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi == lo:
        return [0.5 for _ in vals]
    scaled = [(value - lo) / (hi - lo) for value in vals]
    if reverse:
        scaled = [1.0 - value for value in scaled]
    return scaled


def docking_to_score(docking_scores: Iterable[float | None]) -> list[float]:
    """Convert Vina-like docking energies into 0-1 scores.

    More negative docking energies are better, so scaling is reversed.
    Missing scores receive a conservative 0.0.
    """

    scores = list(docking_scores)
    raw = []
    for score in scores:
        if score is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            raw.append(value)
    if not raw:
        return [0.0 for _ in scores]
    lo = min(float(score) for score in raw)
    hi = max(float(score) for score in raw)
    output: list[float] = []
    for score in scores:
        try:
            value = float(score) if score is not None else float("nan")
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value):
            output.append(0.0)
        elif hi == lo:
            output.append(0.5)
        else:
            output.append((hi - value) / (hi - lo))
    return output


def weighted_score(candidate: CandidateScore, weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    docking_component = candidate.docking_score if candidate.docking_score is not None else 0.0
    total = (
        weights["docking_score"] * docking_component
        + weights["pose_sanity"] * candidate.pose_score
        + weights["admet"] * candidate.admet_score
        + weights["synthesis"] * candidate.synthesis_score
        + weights["novelty_diversity"] * candidate.novelty_score
        + weights["safety_proxy"] * candidate.safety_proxy_score
    )
    return round(float(total), 6)


def decision_label(score: float) -> str:
    if score >= 0.78:
        return "primary_candidate"
    if score >= 0.62:
        return "backup_candidate"
    if score >= 0.45:
        return "hold_for_review"
    return "reject_or_low_priority"


def rank_dataframe(frame, weights: dict[str, float] | None = None):
    """Rank a pandas DataFrame with standard candidate score columns."""

    required = [
        "candidate_id",
        "docking_score_scaled",
        "pose_score",
        "admet_score",
        "synthesis_score",
        "novelty_score",
        "safety_proxy_score",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing ranking columns: {missing}")

    ranked = frame.copy()
    scores = []
    for row in ranked.to_dict(orient="records"):
        candidate = CandidateScore(
            candidate_id=str(row["candidate_id"]),
            docking_score=float(row["docking_score_scaled"]),
            pose_score=float(row["pose_score"]),
            admet_score=float(row["admet_score"]),
            synthesis_score=float(row["synthesis_score"]),
            novelty_score=float(row["novelty_score"]),
            safety_proxy_score=float(row["safety_proxy_score"]),
        )
        scores.append(weighted_score(candidate, weights))
    ranked["final_weighted_score"] = scores
    ranked = ranked.sort_values("final_weighted_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["decision"] = ranked["final_weighted_score"].map(decision_label)
    return ranked
