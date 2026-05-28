from __future__ import annotations

import argparse
import json
import platform
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emd_v5_2_hybrid.registry import register_artifact, register_run


GITHUB_API = "https://api.github.com/repos/ccsb-scripps/AutoDock-Vina/releases/latest"


def desired_asset_name() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return "vina_1.2.7_win.exe"
    if system == "linux":
        if "aarch64" in machine or "arm64" in machine:
            return "vina_1.2.7_linux_aarch64"
        return "vina_1.2.7_linux_x86_64"
    if system == "darwin":
        if "arm64" in machine or "aarch64" in machine:
            return "vina_1.2.7_mac_aarch64"
        return "vina_1.2.7_mac_x86_64"
    raise RuntimeError(f"Unsupported platform for Vina binary bootstrap: {platform.platform()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official AutoDock Vina release binary into tools/vina.")
    parser.add_argument("--base", default=str(ROOT))
    parser.add_argument("--asset-name", default=None)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    asset_name = args.asset_name or desired_asset_name()
    with urllib.request.urlopen(GITHUB_API, timeout=60) as response:
        release = json.loads(response.read().decode("utf-8"))
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    if asset_name not in assets:
        raise RuntimeError(f"Could not find asset {asset_name}. Available: {sorted(assets)}")

    output_dir = base / "tools" / "vina"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "vina.exe" if asset_name.endswith(".exe") else "vina"
    output_path = output_dir / output_name
    urllib.request.urlretrieve(assets[asset_name]["browser_download_url"], output_path)
    if not output_path.name.endswith(".exe"):
        output_path.chmod(0o755)

    register_artifact(base, "M5_vina_binary", output_path, "vina_executable", owner="Student 5")
    register_run(
        base,
        stage="M5_vina_binary",
        status="completed",
        output_path=str(output_path),
        notes=f"Downloaded {asset_name} from official GitHub release {release.get('tag_name')}",
    )
    print(f"Downloaded {asset_name} to {output_path}")


if __name__ == "__main__":
    main()
