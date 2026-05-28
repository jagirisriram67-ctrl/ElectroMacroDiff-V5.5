from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.pose_analysis import analyze_pose, render_pose_png
from emd_v5_2_hybrid.registry import register_artifact, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze top docked poses for simple geometric sanity checks.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--ranking-csv", default=None)
    parser.add_argument("--receptor-pdb", default=None)
    parser.add_argument("--grid-json", default=None)
    parser.add_argument("--pose-dir", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--stage", default="M5_pose_sanity")
    args = parser.parse_args()

    import pandas as pd

    base = Path(args.base).resolve()
    ranking_path = Path(args.ranking_csv).resolve() if args.ranking_csv else base / "08_final_ranking" / "final_ranked_candidates.csv"
    receptor_pdb = Path(args.receptor_pdb).resolve() if args.receptor_pdb else base / "06_docking" / "receptor" / "5AEP_receptor_clean.pdb"
    grid_json = Path(args.grid_json).resolve() if args.grid_json else base / "06_docking" / "receptor" / "docking_grid_5AEP_QUP.json"
    pose_dir = Path(args.pose_dir).resolve() if args.pose_dir else base / "06_docking" / "poses"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else base / "06_docking" / "scores" / "pose_sanity_scores.csv"
    image_dir = Path(args.image_dir).resolve() if args.image_dir else base / "06_docking" / "images" / "pose_sanity"

    ranking = pd.read_csv(ranking_path).sort_values("rank")
    selected = ranking[ranking.get("selection_tier", "") == "final_candidate"].copy()
    if len(selected) < args.top_n:
        inspect = pd.concat([selected, ranking]).drop_duplicates("candidate_id").head(args.top_n)
    else:
        inspect = selected.head(args.top_n)

    rows = []
    rendered = 0
    for row in inspect.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        pose_path = pose_dir / f"{candidate_id}_vina_out.pdbqt"
        if not pose_path.exists():
            continue
        result = analyze_pose(
            candidate_id=candidate_id,
            pose_pdbqt=pose_path,
            receptor_pdb=receptor_pdb,
            grid_json=grid_json,
            docking_score=row.get("docking_score"),
        )
        if args.render:
            png = render_pose_png(candidate_id, pose_path, receptor_pdb, image_dir / f"{candidate_id}_pose.png")
            result["pose_image"] = str(png)
            register_artifact(base, args.stage, png, "pose_sanity_png", owner="Student 5")
            rendered += 1
        rows.append(result)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    register_artifact(base, args.stage, output_csv, "pose_sanity_scores", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=str(ranking_path),
        output_path=str(output_csv),
        molecules_in=len(inspect),
        molecules_out=len(rows),
        notes=f"Rendered {rendered} pose images",
    )
    print(f"Pose sanity rows: {len(rows)}")
    print(f"Pose sanity CSV: {output_csv}")


if __name__ == "__main__":
    main()
