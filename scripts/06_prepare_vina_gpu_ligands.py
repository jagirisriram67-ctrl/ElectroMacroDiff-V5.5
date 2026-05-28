from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run


VALID_AD_TYPES = {
    "C",
    "A",
    "N",
    "O",
    "P",
    "S",
    "H",
    "F",
    "I",
    "NA",
    "OA",
    "SA",
    "HD",
    "Mg",
    "Mn",
    "Zn",
    "Ca",
    "Fe",
    "Cl",
    "Br",
}

TYPE_ALIASES = {
    "SE": "S",
    "SI": "C",
    "LP": "C",
}

ATOM_RECORD_PREFIXES = ("ATOM", "HETATM")


def _element_from_atom_name(atom_name: str) -> str:
    letters = "".join(ch for ch in atom_name.strip() if ch.isalpha())
    if not letters:
        return ""
    upper = letters.upper()
    if upper.startswith("CL"):
        return "Cl"
    if upper.startswith("BR"):
        return "Br"
    if upper.startswith("MG"):
        return "Mg"
    if upper.startswith("MN"):
        return "Mn"
    if upper.startswith("ZN"):
        return "Zn"
    if upper.startswith("CA"):
        return "Ca"
    if upper.startswith("FE"):
        return "Fe"
    return upper[0]


def infer_ad_type(atom_name: str, raw_type: str) -> str:
    raw = raw_type.strip()
    if raw in VALID_AD_TYPES:
        return raw
    if raw.upper() in TYPE_ALIASES:
        return TYPE_ALIASES[raw.upper()]

    raw_upper = raw.upper()
    if raw_upper.startswith("CL"):
        return "Cl"
    if raw_upper.startswith("BR"):
        return "Br"
    if raw_upper.startswith("MG"):
        return "Mg"
    if raw_upper.startswith("MN"):
        return "Mn"
    if raw_upper.startswith("ZN"):
        return "Zn"
    if raw_upper.startswith("CA"):
        return "Ca"
    if raw_upper.startswith("FE"):
        return "Fe"
    if raw_upper.startswith("HD"):
        return "HD"
    if raw_upper.startswith("OA"):
        return "OA"
    if raw_upper.startswith("NA"):
        return "NA"
    if raw_upper.startswith("SA"):
        return "SA"

    element = _element_from_atom_name(atom_name) or raw_upper[:1]
    if element == "H":
        return "HD"
    if element == "C":
        return "C"
    if element == "N":
        return "N"
    if element == "O":
        return "OA"
    if element == "S":
        return "S"
    if element == "P":
        return "P"
    if element == "F":
        return "F"
    if element == "I":
        return "I"
    if element in {"Cl", "Br", "Mg", "Mn", "Zn", "Ca", "Fe"}:
        return element
    return "C"


def atom_type_from_line(line: str) -> str:
    padded = line.rstrip("\n").ljust(79)
    return "".join(padded[77:79].split())


def rewrite_atom_type(line: str) -> tuple[str, str, str]:
    raw_line = line.rstrip("\n")
    padded = raw_line.ljust(79)
    atom_name = padded[12:16].strip()
    raw_type = atom_type_from_line(padded)
    if not raw_type:
        parts = raw_line.split()
        raw_type = parts[-1] if parts else ""
    new_type = infer_ad_type(atom_name, raw_type)
    if new_type not in VALID_AD_TYPES:
        raise ValueError(f"Could not infer valid AutoDock type from {raw_type!r} in line: {raw_line}")
    # Vina parses AutoDock atom type from columns 78-79, so rewrite those
    # fixed columns instead of regex-replacing arbitrary text.
    rewritten = padded[:77] + f"{new_type:<2}"
    return rewritten.rstrip() + "\n", raw_type, new_type


def first_model_or_all(lines: list[str]) -> list[str]:
    has_model = any(line.startswith("MODEL") for line in lines)
    if not has_model:
        return [line for line in lines if not line.startswith(("MODEL", "ENDMDL"))]

    kept: list[str] = []
    in_first_model = False
    seen_first_model = False
    for line in lines:
        if line.startswith("MODEL"):
            if seen_first_model:
                break
            seen_first_model = True
            in_first_model = True
            continue
        if line.startswith("ENDMDL"):
            if in_first_model:
                break
            continue
        if in_first_model:
            kept.append(line)
    return kept


def sanitize_pdbqt(input_path: Path, output_path: Path) -> dict:
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    lines = first_model_or_all(lines)
    output_lines: list[str] = []
    replacements: Counter[str] = Counter()
    atoms = 0
    branches = 0
    invalid_after: list[tuple[int, str]] = []

    for line_no, line in enumerate(lines, start=1):
        if line.startswith(ATOM_RECORD_PREFIXES):
            atoms += 1
            rewritten, raw_type, new_type = rewrite_atom_type(line)
            if raw_type != new_type:
                replacements[f"{raw_type}->{new_type}"] += 1
            final_type = atom_type_from_line(rewritten)
            if final_type not in VALID_AD_TYPES:
                invalid_after.append((line_no, final_type))
            output_lines.append(rewritten)
        elif line.startswith("BRANCH"):
            branches += 1
            output_lines.append(line)
        elif line.startswith(("MODEL", "ENDMDL")):
            continue
        else:
            output_lines.append(line)

    if atoms == 0:
        raise ValueError(f"No ATOM/HETATM records found in {input_path}")
    if invalid_after:
        sample = ", ".join(f"line {line_no}: {atom_type}" for line_no, atom_type in invalid_after[:5])
        raise ValueError(f"Invalid AutoDock types remain in {input_path}: {sample}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(output_lines), encoding="utf-8")
    return {
        "ligand_id": input_path.stem,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_atoms": atoms,
        "num_branches": branches,
        "num_replacements": sum(replacements.values()),
        "replacements": ";".join(f"{key}:{value}" for key, value in sorted(replacements.items())),
        "status": "cleaned",
    }


def clean_directory(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Vina-GPU-safe ligand PDBQT files.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--expected-count", type=int, default=114)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for PDBQT sanitization.")
    parser.add_argument("--stage", default="V5_3_vina_gpu_ligand_sanitize")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else base / "06_docking" / "v5_3_model_guided" / "ligands_pdbqt"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else base / "06_docking" / "v5_3_model_guided" / "ligands_pdbqt_vina_gpu_clean"
    )
    summary_csv = (
        Path(args.summary_csv).resolve()
        if args.summary_csv
        else base / "06_docking" / "v5_3_model_guided" / "scores" / "vina_gpu_ligand_sanitization_summary.csv"
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing ligand input directory: {input_dir}")
    ligands = sorted(input_dir.glob("CAND_*.pdbqt"))
    if args.expected_count and len(ligands) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} ligand PDBQT files, found {len(ligands)} in {input_dir}")
    if args.clean:
        clean_directory(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    worker_count = max(int(args.workers), 1)
    if worker_count == 1 or len(ligands) <= 1:
        rows = [sanitize_pdbqt(ligand, output_dir / ligand.name) for ligand in ligands]
    else:
        indexed_rows: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(sanitize_pdbqt, ligand, output_dir / ligand.name): index
                for index, ligand in enumerate(ligands)
            }
            for future in as_completed(futures):
                indexed_rows[futures[future]] = future.result()
        rows = [indexed_rows[index] for index in range(len(ligands))]

    clean_ligands = sorted(output_dir.glob("CAND_*.pdbqt"))
    if args.expected_count and len(clean_ligands) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} cleaned PDBQT files, found {len(clean_ligands)} in {output_dir}")

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ligand_id",
                "input_path",
                "output_path",
                "num_atoms",
                "num_branches",
                "num_replacements",
                "replacements",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total_replacements = sum(int(row["num_replacements"]) for row in rows)
    max_atoms = max(int(row["num_atoms"]) for row in rows) if rows else 0
    max_branches = max(int(row["num_branches"]) for row in rows) if rows else 0
    register_artifact(base, args.stage, output_dir, "vina_gpu_clean_ligands", owner="Student 5")
    register_artifact(base, args.stage, summary_csv, "vina_gpu_ligand_sanitization_summary", owner="Student 5")
    register_run(
        base,
        stage=args.stage,
        status="completed",
        input_path=str(input_dir),
        output_path=str(output_dir),
        molecules_in=len(ligands),
        molecules_out=len(clean_ligands),
        notes=f"total_replacements={total_replacements}; max_atoms={max_atoms}; max_branches={max_branches}; workers={worker_count}",
    )
    print(f"Input ligands: {len(ligands)}")
    print(f"Cleaned ligands: {len(clean_ligands)}")
    print(f"Total atom-type replacements: {total_replacements}")
    print(f"Max atoms: {max_atoms}; max branches: {max_branches}")
    print(f"Workers: {worker_count}")
    print(f"Clean ligand dir: {output_dir}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
