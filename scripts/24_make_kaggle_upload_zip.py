from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def should_skip(path: Path) -> bool:
    skipped_names = {"__pycache__", ".pytest_cache"}
    if any(part in skipped_names for part in path.parts):
        return True
    if any(str(part).endswith("_smoke") for part in path.parts):
        return True
    if path.suffix.lower() == ".zip":
        return True
    return False


def build_zip(repo_root: Path, output_zip: Path, include_root_dir: bool = True) -> dict[str, object]:
    files: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip(path.relative_to(repo_root)):
            continue
        files.append(path)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            rel = path.relative_to(repo_root)
            arcname = (Path(repo_root.name) / rel).as_posix() if include_root_dir else rel.as_posix()
            archive.write(path, arcname=arcname)

    with ZipFile(output_zip, "r") as archive:
        names = archive.namelist()
    invalid = [name for name in names if "\\" in name]
    if invalid:
        raise ValueError(f"Zip still contains Windows separators: {invalid[:5]}")

    return {
        "repo_root": str(repo_root),
        "output_zip": str(output_zip),
        "file_count": len(files),
        "sha256": sha256_file(output_zip),
        "root_dir_included": include_root_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Kaggle-safe upload zip with POSIX path separators.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--output-zip", default=None)
    parser.add_argument("--no-root-dir", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.base).resolve()
    output_zip = (
        Path(args.output_zip).resolve()
        if args.output_zip
        else repo_root.parent / f"{repo_root.name}_Kaggle_Upload_POSIX.zip"
    )
    summary = build_zip(repo_root, output_zip, include_root_dir=not args.no_root_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
