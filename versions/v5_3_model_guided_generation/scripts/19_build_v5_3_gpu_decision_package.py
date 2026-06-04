from __future__ import annotations

import argparse
import math
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run


def fmt(value: object, digits: int = 3) -> str:
    if value in ("", None):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def maybe_render_images(top, image_dir: Path) -> dict[str, Path]:
    rendered: dict[str, Path] = {}
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception as exc:
        print(f"RDKit image rendering skipped: {exc}")
        return rendered

    image_dir.mkdir(parents=True, exist_ok=True)
    for row in top.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        smiles = str(row.get("canonical_smiles", row.get("smiles", "")))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        output = image_dir / f"{int(row['rank']):02d}_{candidate_id}.png"
        Draw.MolToFile(mol, str(output), size=(1100, 760), legend=f"Rank {int(row['rank'])}: {candidate_id}")
        rendered[candidate_id] = output
    return rendered


def build_report(top, full_ranking, docking, pose, output_md: Path) -> Path:
    scored = int(docking["best_score"].notna().sum())
    score_count = len(docking)
    best_score = docking["best_score"].min()
    median_score = docking["best_score"].median()
    pose_pass = int((pose.get("pose_decision", "") == "pass").sum()) if len(pose) else 0

    rows = []
    has_pocket = "pocket_electronic_fit_score" in top.columns
    for row in top.to_dict(orient="records"):
        rank_value = row.get("pocket_guided_rank", row.get("rank", ""))
        item = [
            int(rank_value) if str(rank_value).strip() else "",
            row["candidate_id"],
            fmt(row.get("best_score"), 1),
            row.get("pose_decision", ""),
            fmt(row.get("final_weighted_score"), 3),
        ]
        if has_pocket:
            item.extend(
                [
                    fmt(row.get("pocket_electronic_fit_score"), 3),
                    fmt(row.get("pocket_guided_final_score"), 3),
                    row.get("pocket_fit_decision", ""),
                ]
            )
        item.extend(
            [
                fmt(row.get("admet_score"), 3),
                fmt(row.get("synthesis_score"), 3),
                fmt(row.get("qed"), 3),
                fmt(row.get("sa_score"), 3),
                row.get("main_risk", ""),
                row.get("selection_tier", ""),
            ]
        )
        rows.append(item)

    primary = top[top["selection_tier"].astype(str).eq("final_candidate")]
    backups = top[top["selection_tier"].astype(str).ne("final_candidate")]
    primary_ids = ", ".join(primary["candidate_id"].astype(str).tolist()) or "None"
    backup_ids = ", ".join(backups["candidate_id"].astype(str).head(5).tolist()) or "None"

    text = "\n".join(
        [
            "# V5.3 Vina-GPU 2.1 Candidate Decision Package",
            "",
            "## Screening Summary",
            "",
            f"- GPU docking coverage: `{scored} / {score_count}` parsed Vina-GPU 2.1 scores.",
            f"- Best Vina score: `{fmt(best_score, 1)}` kcal/mol.",
            f"- Median Vina score: `{fmt(median_score, 1)}` kcal/mol.",
            f"- Pose sanity checks: `{pose_pass} / {len(pose)}` passed for the inspected top poses.",
            "- Scores are docking proxies, not experimental affinities.",
            "",
            "## Recommended Selection",
            "",
            f"- Primary candidates: {primary_ids}.",
            f"- Backup candidates: {backup_ids}.",
            "",
            "## Top Candidate Table",
            "",
            markdown_table(
                (
                    [
                        "Rank",
                        "Candidate",
                        "Vina",
                        "Pose",
                        "Final",
                        "PocketFit",
                        "PocketFinal",
                        "PocketDecision",
                        "ADMET",
                        "Synth",
                        "QED",
                        "SA",
                        "Risk",
                        "Tier",
                    ]
                    if has_pocket
                    else [
                    "Rank",
                    "Candidate",
                    "Vina",
                    "Pose",
                    "Final",
                    "ADMET",
                    "Synth",
                    "QED",
                    "SA",
                    "Risk",
                    "Tier",
                    ]
                ),
                rows,
            ),
            "",
            "## Interpretation",
            "",
            (
                "The top candidates combine Vina-GPU score, pose sanity, ADMET/synthesis proxies, diversity-aware ranking, and the new pocket/electronic-fit proxy. The pocket-fit score uses PDBQT partial charges and distance-based interaction checks; it is not a quantum electron-density calculation."
                if has_pocket
                else "The top five are the current final candidates because they combine strong Vina-GPU scores, passing pose sanity, acceptable ADMET proxy scores, and diversity-aware ranking. The next five are useful backups for visual inspection, alternate protonation/tautomer checks, and repeat docking validation."
            ),
            "",
            "## Suggested Next Validation",
            "",
            "1. Visually inspect the top five docked poses in PyMOL or ChimeraX.",
            "2. Re-dock the top ten with an independent seed/config or CPU Vina for reproducibility.",
            "3. Check protonation/tautomer states for the top five before making any potency claim.",
            "4. Run a higher-fidelity rescoring method only on the top five if compute is available.",
            "",
            "## Top Candidate SMILES",
            "",
        ]
    )

    smiles_lines = []
    for row in top.to_dict(orient="records"):
        smiles = row.get("canonical_smiles", row.get("smiles", ""))
        rank_value = row.get("pocket_guided_rank", row.get("rank", ""))
        smiles_lines.append(f"- `{int(rank_value) if str(rank_value).strip() else ''}` `{row['candidate_id']}`: `{smiles}`")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(text + "\n".join(smiles_lines) + "\n", encoding="utf-8")
    return output_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a V5.3 Vina-GPU candidate decision package.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--stage", default="V5_3_vina_gpu_decision_package")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    ranking_candidates = [
        base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates_with_se3.csv",
        base / "08_final_ranking" / "v5_3_model_guided_pocket_electronic_ranked_candidates.csv",
        base / "08_final_ranking" / "v5_3_model_guided_ranked_candidates.csv",
    ]
    ranking_path = next((path for path in ranking_candidates if path.exists()), ranking_candidates[-1])
    docking_path = base / "06_docking" / "v5_3_model_guided" / "scores" / "docking_scores_full_vina_gpu_2_1.csv"
    pose_path = base / "06_docking" / "v5_3_model_guided" / "scores" / "pose_sanity_scores.csv"
    output_dir = base / "09_reports" / "v5_3_gpu_decision_package"
    image_dir = output_dir / "molecule_images"
    output_csv = output_dir / "v5_3_gpu_top_candidates.csv"
    output_md = output_dir / "V5_3_GPU_Candidate_Decision_Package.md"
    output_zip = output_dir / "V5_3_GPU_Candidate_Decision_Package.zip"

    ranking = pd.read_csv(ranking_path)
    rank_column = "pocket_guided_rank" if "pocket_guided_rank" in ranking.columns else "rank"
    ranking = ranking.sort_values(rank_column)
    docking = pd.read_csv(docking_path)
    docking["best_score"] = pd.to_numeric(docking["best_score"], errors="coerce")
    pose = pd.read_csv(pose_path) if pose_path.exists() else pd.DataFrame(columns=["candidate_id"])

    top = ranking.head(args.top_n).copy()
    rendered = maybe_render_images(top, image_dir)
    top["molecule_image"] = [str(rendered.get(str(cid), "")) for cid in top["candidate_id"]]

    keep_cols = [
        "rank",
        "pocket_guided_rank",
        "candidate_id",
        "selection_tier",
        "decision",
        "pocket_guided_decision",
        "best_score",
        "final_weighted_score",
        "pocket_guided_final_score",
        "pose_decision",
        "pose_score",
        "pose_score_with_pocket_electronics",
        "pocket_electronic_fit_score",
        "electrostatic_score",
        "hbond_score",
        "hydrophobic_score",
        "aromatic_score",
        "hbond_opportunity_pairs",
        "hydrophobic_contacts",
        "aromatic_contacts",
        "pocket_fit_decision",
        "contacts_within_4A",
        "hard_clashes_lt_1_8A",
        "se3_geometry_loss",
        "se3_geometry_confidence",
        "se3_geometry_status",
        "admet_score",
        "synthesis_score",
        "safety_proxy_score",
        "qed",
        "sa_score",
        "mw",
        "logp",
        "tpsa",
        "lipinski_violations",
        "veber_pass",
        "max_ring_size",
        "main_risk",
        "canonical_smiles",
        "molecule_image",
    ]
    keep_cols = [column for column in keep_cols if column in top.columns]
    output_dir.mkdir(parents=True, exist_ok=True)
    top[keep_cols].to_csv(output_csv, index=False)

    build_report(top, ranking, docking, pose, output_md)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in [output_csv, output_md, ranking_path, docking_path, pose_path]:
            if path.exists():
                z.write(path, path.relative_to(base))
        for image in image_dir.glob("*.png"):
            z.write(image, image.relative_to(base))
        pose_dir = base / "06_docking" / "v5_3_model_guided" / "poses"
        for candidate_id in top["candidate_id"].astype(str):
            pose_file = pose_dir / f"{candidate_id}_vina_out.pdbqt"
            if pose_file.exists():
                z.write(pose_file, pose_file.relative_to(base))

    for path, artifact_type in [
        (output_csv, "v5_3_gpu_top_candidates_csv"),
        (output_md, "v5_3_gpu_decision_report"),
        (output_zip, "v5_3_gpu_decision_package_zip"),
    ]:
        register_artifact(base, args.stage, path, artifact_type, owner="Student 1")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=f"{ranking_path}; {docking_path}; {pose_path}",
        output_path=str(output_dir),
        molecules_in=len(ranking),
        molecules_out=len(top),
        notes="Built Vina-GPU top-candidate decision package",
    )

    print(f"Top candidate CSV: {output_csv}")
    print(f"Decision report: {output_md}")
    print(f"Package zip: {output_zip}")
    print(f"Rendered molecule images: {len(rendered)} / {len(top)}")


if __name__ == "__main__":
    main()
