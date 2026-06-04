"""
EMD V5.2 Hybrid — Convert .py Notebook Scripts to .ipynb
========================================================
Converts the annotated Python scripts in notebooks/ into
proper Jupyter .ipynb files that work directly in Google Colab.

Usage:
    python convert_to_ipynb.py
    python convert_to_ipynb.py --output-dir 10_notebooks
"""

import os
import sys
import json
import re
import argparse


def py_to_ipynb(py_path: str, ipynb_path: str):
    """Convert a structured .py notebook script to .ipynb format.
    
    Splits on '# ── CELL' markers. Lines starting with '# !' become
    shell commands. Comment-only blocks that look like installs or
    Drive mounts are placed in their own code cells.
    """
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    
    # Split into cells by the CELL markers
    cells = []
    current_cell_lines = []
    current_cell_title = ""
    header_lines = []
    in_header = True
    
    for line in lines:
        # Detect cell boundary
        cell_match = re.match(r"^# ── CELL \d+:?\s*(.*?)\s*─*$", line)
        
        if cell_match:
            in_header = False
            # Save previous cell
            if current_cell_lines:
                cells.append({
                    "title": current_cell_title,
                    "lines": current_cell_lines,
                })
            elif header_lines:
                cells.append({
                    "title": "Header",
                    "lines": header_lines,
                    "is_markdown": True,
                })
                header_lines = []
            
            current_cell_title = cell_match.group(1).strip()
            current_cell_lines = []
        elif in_header:
            header_lines.append(line)
        else:
            current_cell_lines.append(line)
    
    # Don't forget the last cell
    if current_cell_lines:
        cells.append({
            "title": current_cell_title,
            "lines": current_cell_lines,
        })
    
    # Build notebook JSON
    nb_cells = []
    
    for cell in cells:
        if cell.get("is_markdown"):
            # Convert header comments to markdown
            md_lines = []
            for line in cell["lines"]:
                if line.startswith("# ="):
                    continue
                elif line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.strip() == "#":
                    md_lines.append("")
                elif line.strip():
                    md_lines.append(line)
            
            if md_lines:
                nb_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [l + "\n" for l in md_lines]
                })
            continue
        
        # Add markdown title cell
        if cell["title"]:
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {cell['title']}\n"]
            })
        
        # Process code lines
        code_lines = []
        for line in cell["lines"]:
            # Convert commented-out pip/shell commands
            if re.match(r"^# !(pip|apt|wget|git|cd)", line):
                code_lines.append(line[2:])  # Uncomment the command
            else:
                code_lines.append(line)
        
        # Remove leading/trailing blank lines
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        
        if code_lines:
            nb_cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": [l + "\n" for l in code_lines],
                "execution_count": None,
                "outputs": []
            })
    
    # Assemble notebook
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "accelerator": "GPU"
        },
        "cells": nb_cells
    }
    
    os.makedirs(os.path.dirname(ipynb_path), exist_ok=True)
    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"  [OK] {os.path.basename(py_path)} -> {os.path.basename(ipynb_path)} ({len(nb_cells)} cells)")


def convert_all(notebooks_dir: str, output_dir: str = None):
    """Convert all .py notebook scripts to .ipynb format."""
    if output_dir is None:
        output_dir = notebooks_dir
    
    py_files = sorted([
        f for f in os.listdir(notebooks_dir)
        if f.endswith(".py") and not f.startswith("__")
    ])
    
    print(f"Converting {len(py_files)} notebook scripts to .ipynb format...\n")
    
    for py_file in py_files:
        py_path = os.path.join(notebooks_dir, py_file)
        ipynb_name = py_file.replace(".py", ".ipynb")
        ipynb_path = os.path.join(output_dir, ipynb_name)
        
        try:
            py_to_ipynb(py_path, ipynb_path)
        except Exception as e:
            print(f"  [FAIL] {py_file}: {e}")
    
    print(f"\nDone! {len(py_files)} notebooks converted.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .py notebooks to .ipynb")
    parser.add_argument(
        "--notebooks-dir",
        default=os.path.join(os.path.dirname(__file__), "notebooks"),
        help="Directory containing .py notebook scripts"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for .ipynb files (default: same as input)"
    )
    
    args = parser.parse_args()
    convert_all(args.notebooks_dir, args.output_dir)
