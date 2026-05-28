"""Shared snippets for Colab notebooks."""

from __future__ import annotations

from pathlib import Path

from .registry import ensure_project_tree, register_run, save_progress


def setup_colab_project(base: str = "/content/drive/MyDrive/EMD_V5_2_Hybrid") -> Path:
    root = Path(base)
    ensure_project_tree(root)
    register_run(root, stage="notebook_setup", status="completed", notes="project tree validated")
    return root


def tiny_debug_slice(frame, tiny_debug: bool = True, size: int = 25):
    if tiny_debug:
        return frame.head(size).copy()
    return frame


def mark_notebook_progress(base: str | Path, notebook_name: str, step: str, index: int = 0) -> None:
    save_progress(
        Path(base) / "00_project_registry" / f"progress_{notebook_name}.json",
        {"notebook": notebook_name, "step": step, "last_completed_index": index},
    )
