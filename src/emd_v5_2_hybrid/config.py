"""Configuration loading and path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MissingDependencyError(RuntimeError):
    """Raised when an optional package is required for the requested action."""


@dataclass(frozen=True)
class ProjectConfig:
    """Thin convenience wrapper around the YAML campaign configuration."""

    data: dict[str, Any]
    path: Path | None = None

    @property
    def project_name(self) -> str:
        return str(self.data.get("project", {}).get("name", "EMD_V5_2_Hybrid"))

    @property
    def random_seed(self) -> int:
        return int(self.data.get("seeds", {}).get("random_seed", 42))

    @property
    def drive_base(self) -> str:
        return str(self.data.get("paths", {}).get("drive_base", "/content/drive/MyDrive/EMD_V5_2_Hybrid"))

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML.

    PyYAML is intentionally lazy-imported so initialization tests can run before
    Colab dependencies are installed.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError("Install pyyaml to load campaign_config.yaml") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    return ProjectConfig(data=load_yaml(config_path), path=config_path)


def default_config_path(base: str | Path) -> Path:
    return Path(base) / "00_project_registry" / "campaign_config.yaml"
