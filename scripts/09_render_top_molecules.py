from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 2D PNG structures for top ranked candidates.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw

    base = Path(args.base).resolve()
    ranking_path = base / "08_final_ranking" / "final_ranked_candidates.csv"
    output_dir = base / "09_reports" / "molecule_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking = pd.read_csv(ranking_path).sort_values("rank").head(args.top_n)
    rows = []
    rendered = 0
    for row in ranking.to_dict(orient="records"):
        candidate_id = str(row["candidate_id"])
        smiles = str(row.get("smiles", row.get("canonical_smiles", "")))
        mol = Chem.MolFromSmiles(smiles)
        output_path = output_dir / f"{candidate_id}.png"
        status = "failed"
        if mol is not None:
            Draw.MolToFile(mol, str(output_path), size=(900, 650), legend=candidate_id)
            status = "rendered"
            rendered += 1
            register_artifact(base, "M8_reporting", output_path, "candidate_2d_png", owner="Student 1")
        rows.append(
            {
                "rank": row.get("rank", ""),
                "candidate_id": candidate_id,
                "smiles": smiles,
                "image_path": str(output_path) if output_path.exists() else "",
                "status": status,
            }
        )

    manifest_path = output_dir / "molecule_image_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    register_artifact(base, "M8_reporting", manifest_path, "molecule_image_manifest", owner="Student 1")
    register_run(
        base,
        stage="M8_render_molecules",
        status="completed",
        input_path=str(ranking_path),
        output_path=str(output_dir),
        molecules_in=len(ranking),
        molecules_out=rendered,
        notes="2D molecule PNGs rendered from ranked SMILES",
    )
    print(f"Rendered {rendered} / {len(ranking)} molecule images in {output_dir}")


if __name__ == "__main__":
    main()
