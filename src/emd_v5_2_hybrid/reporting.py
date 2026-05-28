"""Small report artifact helpers."""

from __future__ import annotations

from pathlib import Path


def display_value(value, fallback: str = "not_available") -> str:
    text = str(value)
    if text.lower() in {"nan", "none", ""}:
        return fallback
    return text


def write_markdown_candidate_card(candidate: dict, output_dir: str | Path) -> Path:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate_id = str(candidate.get("candidate_id", "candidate"))
    path = output_root / f"{candidate_id}_card.md"
    lines = [
        f"# Candidate {candidate_id}",
        "",
        f"- Source generator: {candidate.get('source_generator', '')}",
        f"- SMILES: `{display_value(candidate.get('smiles', candidate.get('canonical_smiles', '')))}`",
        f"- InChIKey: `{display_value(candidate.get('inchikey', ''))}`",
        f"- Docking score: {display_value(candidate.get('docking_score', 'not_run'), 'not_run')}",
        f"- QED: {display_value(candidate.get('qed', ''))}",
        f"- SA score/proxy: {display_value(candidate.get('sa_score', ''))}",
        f"- ADMET score: {display_value(candidate.get('admet_score', ''))}",
        f"- Safety proxy score: {display_value(candidate.get('safety_proxy_score', ''))}",
        f"- Final weighted score: {display_value(candidate.get('final_weighted_score', ''))}",
        f"- Decision: {display_value(candidate.get('decision', ''))}",
        f"- Main risk: {display_value(candidate.get('main_risk', ''))}",
        "",
        "## Scientific Note",
        "",
        "This is a computational prioritization hypothesis and requires experimental validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_limitations_section(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = """# Limitations

- Docking scores are approximate ranking signals, not measured binding affinities.
- ADMET and safety filters are descriptor/proxy screens, not biological safety proof.
- The SE(3) flow model is constrained by small data, Colab runtime limits, and short training.
- Candidate synthesizability is preliminary unless full retrosynthesis is run and manually reviewed.
- Final molecules must be validated by synthesis, biochemical assay, selectivity profiling, and cell-based testing.
"""
    output.write_text(text, encoding="utf-8")
    return output
