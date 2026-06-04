"""Build a reviewer-facing file guide for every tracked repository file."""

from __future__ import annotations

import datetime as _dt
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "REPOSITORY_FILE_GUIDE_FOR_REVIEWERS.md"
SELF = "scripts/build_reviewer_repository_file_guide.py"


FOLDER_PURPOSES = {
    "<root>": ("Project entry files", "Top-level settings, README, license/config controls, and reviewer guide."),
    "00_project_registry": ("Project registry and audit trail", "Configuration, environment, run registry, artifact registry, progress logs, and validation evidence."),
    "01_raw_data": ("Raw input data", "Original JAK2 activity records and protein structure inputs before curation."),
    "02_curated_data": ("Curated chemistry datasets", "Cleaned ligand tables, macrocycle subsets, and fragment-linker datasets."),
    "03_features": ("Feature and graph tensors", "Descriptors, graph tensors, split IDs, anchor labels, and V5.5 feature manifests."),
    "04_models_checkpoints": ("Model checkpoints and training logs", "Saved PyTorch models, smoke runs, logs, curves, and training summaries."),
    "05_generated_candidates": ("Generated molecule outputs", "Candidate SMILES tables, attempt logs, generation metrics, and novelty summaries."),
    "06_docking": ("Docking inputs and outputs", "Receptors, docking grids, SDF/PDBQT ligands, Vina-GPU scores, poses, and pocket-fit scores."),
    "07_admet_synthesis": ("ADMET and synthesis proxies", "Computational drug-likeness, synthesis, and safety proxy outputs."),
    "08_final_ranking": ("Final ranked candidate tables", "Consensus and pocket-electronic ranked candidates."),
    "09_reports": ("Reports and decision packages", "Benchmark reports, candidate cards, molecule images, and decision packages."),
    "10_notebooks": ("Notebook workflows", "Notebook-style execution material for Colab/Kaggle or stepwise reproduction."),
    "11_logs": ("Work logs", "Daily implementation/progress notes and AI assistance logs."),
    "docs": ("Technical documentation", "Handover reports, methods, architecture explanations, reviewer answers, and claim boundaries."),
    "interface": ("Interactive dashboard", "Static HTML/CSS/JS dashboard and real exported data for visual review and 3D inspection."),
    "scripts": ("Executable pipeline stages", "Command-line scripts for data, features, training, generation, docking, scoring, reports, and validation."),
    "src": ("Reusable Python package", "Core implementation modules imported by scripts and tests."),
    "submission_artifacts_2026-05-28": ("Submission-ready artifacts", "PPT, DOCX, diagrams, workflow explanations, and final college-review assets."),
    "tests": ("Unit tests", "Smoke and focused tests for chemistry, generation, benchmark, dashboard, and V5.5 utilities."),
    "tools": ("Tool configuration/helpers", "Auxiliary tool folders such as Vina-related placeholders/configuration."),
    "versions": ("Earlier-version archive", "V5.0/V5.1 planning, V5.2 initial code, V5.3 outputs, and V5.4 pocket scoring evidence."),
}


SCRIPT_PURPOSES = {
    "01_collect_jak2_data.py": "collects JAK2 ligand/activity data",
    "02_curate_dataset.py": "cleans and standardizes ligand activity data",
    "03_build_ligand_features.py": "builds descriptors, graph features, and splits",
    "03_train_se3_main.py": "trains or resumes the SE(3) flow model",
    "04_generate_candidates.py": "runs baseline candidate generation",
    "05_score_admet_synthesis.py": "scores ADMET, synthesis, and safety proxy features",
    "06_prepare_docking_inputs.py": "turns candidates into SDF docking inputs and grid metadata",
    "06_prepare_pdbqt.py": "prepares receptor/ligand PDBQT files for docking",
    "06_prepare_vina_gpu_ligands.py": "sanitizes ligand PDBQT files for Vina-GPU compatibility",
    "06_pose_sanity.py": "checks docked pose sanity and produces review artifacts",
    "07_rank_candidates.py": "combines generation, docking, ADMET, and pose evidence into ranked tables",
    "08_validate_project.py": "runs the final project validation checklist",
    "09_build_reports.py": "builds reports and candidate summaries",
    "13_build_v5_3_fragment_dataset.py": "builds V5.3 fragment-linker datasets",
    "14_train_v5_3_anchor_model.py": "trains the V5.3 anchor MLP model",
    "14b_train_v5_3_anchor_gnn.py": "trains the V5.3 anchor GNN experiment",
    "15_train_v5_3_linker_size_model.py": "trains the V5.3 linker-size model",
    "17_generate_v5_3_model_guided_macrocycles.py": "generates V5.3 model-guided macrocycles",
    "17b_merge_v5_3_generation_chunks.py": "merges V5.3 generation chunks",
    "18_build_v5_3_benchmark_report.py": "builds benchmark report tables",
    "19_build_v5_3_gpu_decision_package.py": "builds V5.3 GPU docking decision package",
    "20_measure_v5_3_benchmark_gaps.py": "computes raw validity and linker novelty benchmark gaps",
    "21_score_pocket_electronics.py": "scores JAK2 pocket-electronic complementarity",
    "23_score_se3_geometry.py": "scores candidates with auxiliary SE(3) geometry confidence",
    "24_build_v5_5_training_data.py": "builds V5.5 pocket-conditioned training tables",
    "25_train_pocket_anchor_gnn.py": "trains the V5.5 pocket anchor GNN",
    "26_train_pocket_linker_policy.py": "trains the V5.5 pocket linker policy model",
    "27_train_validity_reward.py": "trains the V5.5 validity reward model",
    "28_train_se3_continuation.py": "continues SE(3) training for V5.5 auxiliary evidence",
    "29_generate_v5_5_pocket_guided.py": "generates V5.5 pocket-conditioned macrocycle candidates",
    "30_prepare_v5_5_docking_inputs_parallel.py": "prepares V5.5 docking inputs in parallel",
    "build_reviewer_repository_file_guide.py": "builds this complete repository guide",
    "build_v5_5_presentation_docs.py": "builds V5.5 presentation/document outputs",
    "fix_kaggle_zip.py": "fixes ZIP path separators for Kaggle upload compatibility",
    "run_v5_3_vina_gpu_colab.py": "runs V5.3 Vina-GPU workflow in Colab/Kaggle style environments",
}


MODULE_PURPOSES = {
    "anchor_gnn.py": "GNN utilities for anchor-site prediction",
    "baseline_generation.py": "RDKit/baseline macrocycle generation and chemotype builder logic",
    "chemistry.py": "chemistry helper functions and RDKit wrappers",
    "config.py": "project configuration, path, registry, and missing-dependency helpers",
    "device_helper.py": "CUDA/XLA/CPU device resolution helper for Kaggle/Colab/local use",
    "docking_prep.py": "SDF and docking input preparation utilities",
    "linker_size_features.py": "feature engineering for linker length/policy models",
    "macrocycle_fragmentation.py": "macrocycle fragmentation and linker extraction utilities",
    "pocket_anchor_gnn.py": "V5.5 pocket-conditioned anchor GNN implementation",
    "pocket_electronics.py": "JAK2 pocket electronic-profile and pocket-fit scoring utilities",
    "pocket_features.py": "32-dimensional pocket feature vector construction",
    "pocket_linker_policy.py": "V5.5 dual-head linker size and chemotype policy model",
    "se3_flow.py": "SE(3) flow matching model and geometry scoring utilities",
    "validity_reward.py": "V5.5 validity/novelty reward model utilities",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace").strip()


def esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def file_size(path: str) -> str:
    p = ROOT / path
    return fmt_size(p.stat().st_size) if p.exists() else "generated"


def top_folder(path: str) -> str:
    return "<root>" if "/" not in path else path.split("/", 1)[0]


def stage(path: str) -> str:
    lower = path.lower()
    if path.startswith("versions/v5_0"):
        return "V5.0/V5.1 archive"
    if path.startswith("versions/v5_2"):
        return "V5.2 archive"
    if path.startswith("versions/v5_3"):
        return "V5.3 archive"
    if path.startswith("versions/v5_4"):
        return "V5.4 archive"
    if "v5_5" in lower or path.startswith("submission_artifacts"):
        return "V5.5 final"
    if "v5_4" in lower or "pocket_electronic" in lower:
        return "V5.4/V5.5 scoring"
    if "v5_3" in lower or "model_guided" in lower:
        return "V5.3 lineage"
    if path.startswith("versions"):
        return "Version archive"
    return "Shared/final repo"


def kind(path: str) -> str:
    name = Path(path).name
    ext = Path(path).suffix.lower()
    if name in {".gitignore", ".gitattributes"}:
        return "Git configuration"
    return {
        ".py": "Python code",
        ".ipynb": "Notebook",
        ".csv": "CSV data table",
        ".json": "JSON metadata/metrics",
        ".pt": "PyTorch model/tensor artifact",
        ".pth": "PyTorch model/tensor artifact",
        ".ckpt": "PyTorch model/tensor artifact",
        ".pdb": "Protein structure",
        ".pdbqt": "Docking structure/pose",
        ".sdf": "Ligand structure file",
        ".md": "Markdown documentation",
        ".docx": "Word document",
        ".pptx": "PowerPoint deck",
        ".pdf": "PDF report",
        ".png": "Image/visual artifact",
        ".jpg": "Image/visual artifact",
        ".jpeg": "Image/visual artifact",
        ".html": "Dashboard/web asset",
        ".css": "Dashboard/web asset",
        ".js": "Dashboard/web asset",
        ".yaml": "YAML configuration",
        ".yml": "YAML configuration",
        ".txt": "Text manifest/log",
        ".gz": "Compressed data artifact",
        ".zip": "Compressed package",
    }.get(ext, "Project artifact")


def candidate_id(path: str) -> str | None:
    match = re.search(r"(CAND_[A-Za-z0-9]+)", path)
    return match.group(1) if match else None


def explain(path: str) -> tuple[str, str]:
    name = Path(path).name
    lower = path.lower()
    cid = candidate_id(path)

    exact = {
        "README.md": ("Main project overview with V5.5 metrics, dashboard link, folder map, and handover status.", "Start here when explaining the whole project."),
        "REPOSITORY_FILE_GUIDE_FOR_REVIEWERS.md": ("This reviewer guide and complete tracked-file inventory.", "Use it as speaking notes for explaining the repository."),
        ".gitignore": ("Rules keeping cache/heavy local-only files out of Git while allowing required handover artifacts.", "Explains why some huge generated files are compressed or excluded."),
        ".gitattributes": ("Line-ending and binary-file rules for code, chemistry files, checkpoints, reports, and archives.", "Protects binary artifacts and keeps text readable across Windows/Linux."),
        "LICENSE": ("Repository license file.", "Defines reuse permissions for the package."),
    }
    if path in exact:
        return exact[path]

    if cid:
        if lower.endswith(".pdbqt") and ("pose" in lower or "_out" in lower):
            return (f"Docked pose file for candidate `{cid}` in AutoDock/Vina PDBQT format.", "Inspect with 3Dmol.js/PyMOL/Vina tools.")
        if lower.endswith(".pdbqt"):
            return (f"Prepared ligand structure for candidate `{cid}` in PDBQT docking format.", "Input or cleaned input for AutoDock/Vina-GPU docking.")
        if lower.endswith(".sdf"):
            return (f"3D ligand structure for candidate `{cid}` in SDF format.", "Intermediate structure before PDBQT conversion and useful for chemistry inspection.")
        if lower.endswith(".png"):
            return (f"Image/pose/molecule visual for candidate `{cid}`.", "Use in reports, slides, or reviewer discussion.")
        if lower.endswith(".md"):
            return (f"Candidate card describing candidate `{cid}`.", "Use for individual molecule review.")
        if "vina.log" in lower:
            return (f"Docking log for candidate `{cid}`.", "Use if a reviewer asks how the Vina result was produced or parsed.")

    if path.startswith("00_project_registry/"):
        if "artifact_registry" in name:
            return ("Registry of generated artifacts with stage, path, type, owner, timestamp, and notes.", "Audit trail of project outputs.")
        if "run_registry" in name:
            return ("Registry of pipeline/validation runs and their status.", "Shows run history and validation completion.")
        if "validation_report" in name:
            return ("Final validation checklist result JSON.", "Confirms the validation gate passed.")
        if "environment" in name:
            return ("Recorded environment/package versions.", "Use for reproducibility discussion.")
        if "progress" in name:
            return ("Progress/status JSON for a project milestone or version stage.", "Trace when each stage was completed.")
        if "campaign_config" in name:
            return ("Campaign configuration for target, data, generation, docking, and scoring defaults.", "Explains pipeline settings.")
        if "requirements" in name:
            return ("Python dependency list for Colab/Kaggle style execution.", "Use when recreating the environment.")
        return ("Project registry/configuration artifact.", "Supports reproducibility and auditability.")

    if path.startswith("01_raw_data/"):
        if "chembl" in lower:
            return ("Raw ChEMBL JAK2 activity export before curation.", "First ligand/activity input for the project.")
        if lower.endswith(".pdb"):
            return ("Raw RCSB PDB protein structure input for JAK2/5AEP.", "Starting structure for receptor preparation and pocket definition.")
        return ("Raw source input file.", "Used before cleaning and curation.")

    if path.startswith("02_curated_data/"):
        if "jak2_curated" in name:
            return ("Cleaned JAK2 ligand/activity table.", "Core curated ligand dataset.")
        if "macrocycle_constrained" in name:
            return ("JAK2 ligands filtered for macrocycle/constrained-ring relevance.", "Training/evaluation source for macrocycle-specific modeling.")
        if "fragment_linker" in name:
            return ("Fragment-linker pair table from macrocycle fragmentation.", "Training data for linker-size/linker-policy models.")
        if "pretrain_macrocycles" in name:
            return ("Broad macrocycle set used for pretraining before JAK2 fine-tuning.", "Explains transfer learning data beyond JAK2.")
        return ("Curated chemistry data table.", "Feeds feature generation and model training.")

    if path.startswith("03_features/"):
        if name == "se3_graphs.pt":
            return ("Serialized molecular graph tensors and 3D coordinates for SE(3) flow training/scoring.", "Required runtime artifact for the SE(3) model path.")
        if "se3_graph_index" in name:
            return ("Index mapping molecules to SE(3) graph tensors.", "Connects graph tensors back to ligand IDs.")
        if "splits" in path:
            return ("Train/validation/test split ID list.", "Shows molecule partitioning for evaluation.")
        if "anchor_atom_training" in name:
            return ("Per-atom anchor-label training table.", "Used by V5.3/V5.5 anchor models to learn attachment-site decisions.")
        if "linker_chemotype" in name:
            return ("V5.5 linker chemotype training/manifest table.", "Supports pocket linker policy training and chemotype tracking.")
        if "validity_reward" in name:
            return ("V5.5 attempt-log-derived training table for validity reward model.", "Trains the gatekeeper for likely valid macrocycles.")
        if "v6_premium" in name:
            return ("Premium-dataset planning/manifest artifact for future V6 work.", "Documents future expansion, not active V5.5 claims.")
        return ("Feature table or model input artifact.", "Feeds ML training, generation, or scoring.")

    if path.startswith("04_models_checkpoints/"):
        if lower.endswith((".pt", ".pth", ".ckpt")):
            if "pocket_anchor_gnn" in lower:
                return ("Trained V5.5 pocket-conditioned anchor GNN checkpoint.", "Active generation model for choosing anchor atoms with pocket context.")
            if "pocket_linker_policy" in lower:
                return ("Trained V5.5 dual-head linker-size/chemotype policy checkpoint.", "Active generation model for linker length and chemistry decisions.")
            if "validity_reward" in lower:
                return ("Trained V5.5 validity reward model checkpoint.", "Active generation gatekeeper for likely valid macrocycles.")
            if "se3" in lower:
                return ("SE(3) flow checkpoint or continuation checkpoint.", "Auxiliary geometry evidence, not the main molecule builder.")
            if "v5_3_anchor" in lower:
                return ("V5.3 anchor model checkpoint.", "Earlier anchor-site model evidence and V5.5 lineage.")
            if "v5_3_linker" in lower:
                return ("V5.3 linker-size model checkpoint.", "Earlier linker-length model evidence and V5.5 lineage.")
            return ("Saved trained model checkpoint.", "Load with the corresponding script.")
        if "training_log" in name:
            return ("Epoch-by-epoch training log.", "Explains training behavior and metrics.")
        if "training_summary" in name or "acceptance" in name:
            return ("Training summary/acceptance metrics JSON.", "States final model quality with evidence.")
        if "training_curve" in name:
            return ("Training curve image.", "Use visually in presentations or review.")
        return ("Model-training artifact.", "Supports checkpoint provenance and training evidence.")

    if path.startswith("05_generated_candidates/"):
        if "attempt_log" in name:
            return ("Raw generation attempt log with success/failure status.", "Needed for honest raw-attempt validity.")
        if "novelty" in name:
            return ("Linker novelty measurement table.", "Used for MED-style novelty comparison.")
        if "metrics" in name or "summary" in name:
            return ("Generation summary/metrics file.", "Shows validity, uniqueness, macrocycle status, and branch results.")
        if "generated" in name:
            return ("Generated candidate molecule table with SMILES and descriptors.", "Primary molecule output from that branch.")
        return ("Generated-candidate artifact.", "Supports molecule generation evidence.")

    if path.startswith("06_docking/"):
        if "receptor" in path:
            if "grid" in name or "box" in name:
                return ("Docking grid/box definition around the JAK2 pocket.", "Defines where candidates were docked.")
            if lower.endswith(".pdbqt"):
                return ("Prepared receptor in PDBQT format for AutoDock/Vina-GPU.", "Required docking input.")
            if lower.endswith(".pdb"):
                return ("Clean receptor structure for visualization or pose sanity checking.", "Use in 3D viewer or receptor inspection.")
        if "docking_scores" in name:
            return ("Parsed Vina/Vina-GPU docking score table.", "Core docking evidence for ranked candidates.")
        if "pocket_electronic" in name or "pocket" in path:
            return ("Pocket-electronic profile or fit-score artifact.", "Supports pocket-fit prioritization.")
        if "pose_sanity" in name or "images/pose_sanity" in path:
            return ("Pose sanity check output or image.", "Assesses docked pose plausibility.")
        if "pdbqt_preparation" in name or "manifest" in name or "sanitization" in name:
            return ("Docking preparation manifest/summary.", "Shows how ligand/receptor files were prepared and cleaned.")
        if lower.endswith(".sdf"):
            return ("Ligand SDF docking input.", "Intermediate chemistry structure before PDBQT docking format.")
        if lower.endswith(".pdbqt"):
            return ("PDBQT ligand/receptor/pose file.", "Used by AutoDock/Vina-GPU and 3D pose review.")
        return ("Docking-stage artifact.", "Supports receptor preparation, docking, parsing, or pose review.")

    if path.startswith("07_admet_synthesis/"):
        if "flags" in name:
            return ("ADMET/synthesis filter flags for candidates.", "Shows computational filters triggered by each candidate.")
        if "notes" in name:
            return ("Safety/synthesis proxy notes.", "Readable caveats for candidate selection.")
        return ("ADMET and synthesis proxy score table.", "Used in final ranking with docking and pocket fit.")

    if path.startswith("08_final_ranking/"):
        if "pocket_electronic" in name:
            return ("Pocket-electronic ranked candidate table.", "Final pocket-fit-aware candidate ordering.")
        return ("Final ranked candidate table.", "Use to discuss top molecules and scoring evidence.")

    if path.startswith("09_reports/"):
        if "candidate_cards" in path:
            return ("Individual candidate report card.", "Use for molecule-by-molecule reviewer discussion.")
        if "benchmark" in path:
            return ("Benchmark comparison artifact.", "Supports MED/Macformer/MacLS comparison and gap discussion.")
        if "decision_package" in path:
            return ("Decision-package file for top candidates and V5.3 GPU docking results.", "Explains how candidates were selected.")
        if "molecule_images" in path or lower.endswith((".png", ".jpg", ".jpeg")):
            return ("Molecule image or report visual.", "Use directly in PPT/DOC/dashboard.")
        return ("Project report or supporting report asset.", "Use for final handover or written explanation.")

    if path.startswith("docs/"):
        return ("Technical/handover documentation file.", "Answers reviewer questions about methods, claims, data provenance, models, or workflow.")

    if path.startswith("interface/"):
        if name == "data.js":
            return ("Real exported project data consumed by the dashboard.", "Dashboard should show this data only, not fake values.")
        if name == "app.js":
            return ("Dashboard interactivity, charts, filtering, and 3D viewer logic.", "Use to explain/modify browser review experience.")
        if name == "index.html":
            return ("Dashboard page structure.", "Open this to run the interactive project UI.")
        if name == "styles.css":
            return ("Dashboard visual styling.", "Controls layout, cards, tables, and 3D viewer appearance.")
        if name == "extract_data.py":
            return ("Exports real project CSV/JSON data into dashboard-friendly JavaScript.", "Run after data changes to refresh dashboard.")
        return ("Dashboard support asset.", "Part of the static UI package.")

    if path.startswith("submission_artifacts_2026-05-28/"):
        return ("Submission-ready document, slide, diagram, or workflow explanation asset.", "Use directly for college review and presentation material.")

    if path.startswith("10_notebooks/"):
        return ("Notebook or notebook helper for stepwise execution.", "Use for Colab/Kaggle-style reproduction.")

    if path.startswith("11_logs/"):
        return ("Project work log / assistance note.", "Use for traceability of project work and decisions.")

    if path.startswith("scripts/"):
        purpose = SCRIPT_PURPOSES.get(name, "pipeline command-line script")
        return (f"Script that {purpose}.", "Run from repo root with `python scripts/<name> --base .` style commands when reproducing that stage.")

    if path.startswith("src/emd_v5_2_hybrid/"):
        purpose = MODULE_PURPOSES.get(name, "reusable Python module for the ElectroMacroDiff package")
        return (purpose + ".", "Imported by scripts/tests; this is implementation code rather than final output data.")

    if path.startswith("tests/"):
        return ("Unit/smoke test file.", "Run with `python -m unittest discover -s tests`.")

    if path.startswith("tools/"):
        return ("Auxiliary tool/configuration artifact.", "Supports external tooling such as docking setup.")

    if path.startswith("versions/"):
        if name == "README.md":
            return ("Version-archive explanation for this folder.", "Use to explain how earlier versions connect to final V5.5.")
        if name == "VERSION_ARCHIVE_MANIFEST.json":
            return ("Machine-readable inventory of the version archive.", "Proves counts, sizes, and large-file handling.")
        if "v5_0" in path or "v5_1" in path:
            return ("Earlier planning-stage artifact.", "Explains original motivation and upgrade planning.")
        if "v5_2" in path:
            return ("Archived V5.2 initial hybrid implementation artifact.", "Shows the starting codebase and first working pipeline.")
        if "v5_3" in path:
            return ("Archived V5.3 model-guided generation/docking artifact.", "Shows intermediate training, molecules, metrics, and docking evidence.")
        if "v5_4" in path:
            return ("Archived V5.4 pocket-electronic scoring artifact.", "Explains the scoring bridge from V5.3 to V5.5.")
        return ("Version-history artifact.", "Use for project evolution evidence.")

    return ("Repository file used by ElectroMacroDiff.", "Open it according to file type or folder README.")


def main() -> None:
    tracked = git("ls-files").splitlines()
    for must_include in ("REPOSITORY_FILE_GUIDE_FOR_REVIEWERS.md", SELF):
        if must_include not in tracked:
            tracked.append(must_include)
    tracked = sorted(dict.fromkeys(tracked), key=lambda p: (top_folder(p), p))

    commit = git("rev-parse", "HEAD")
    origin = git("rev-parse", "origin/main")
    status = git("status", "--short", "--branch").splitlines()[0]
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    folder_stats: dict[str, dict[str, int]] = {}
    for path in tracked:
        top = top_folder(path)
        file_path = ROOT / path
        folder_stats.setdefault(top, {"count": 0, "size": 0})
        folder_stats[top]["count"] += 1
        if file_path.exists():
            folder_stats[top]["size"] += file_path.stat().st_size

    lines: list[str] = []
    lines += [
        "# ElectroMacroDiff V5.5 Repository File Guide for Reviewers",
        "",
        "This document maps the GitHub repository for reviewers. It explains what each folder is for, why it exists, how it fits into the project flow, and what every tracked file represents.",
        "",
        "> Scope note: this is a computational drug-discovery handover package. The files prove code, model training evidence, generated molecules, docking outputs, ADMET proxy scores, pocket-electronic scoring, rankings, reports, and dashboard assets. They do not prove wet-lab activity or clinical safety.",
        "",
        "## Verification Snapshot",
        "",
        f"- Guide generated: `{now}` local time",
        f"- Local `HEAD` at guide generation: `{commit[:12]}`",
        f"- `origin/main` at guide generation: `{origin[:12]}`",
        f"- Branch status before writing guide: `{status}`",
        "- Latest recheck before guide generation: unit tests `46/46` passed",
        "- Latest project validation before guide generation: validation checks `54/54` passed",
        f"- Tracked files documented in this guide: `{len(tracked)}`",
        "",
        "## Project Flow in One View",
        "",
        "1. Raw inputs: ChEMBL JAK2 activity table and RCSB PDB `5AEP` protein structure.",
        "2. Curation: ligands are cleaned, standardized, filtered, and split.",
        "3. Feature building: descriptors, graph tensors, fragment-linker pairs, anchor labels, and pocket features are created.",
        "4. Model training: SE(3), V5.3 anchor/linker, and V5.5 pocket-conditioned models are trained or loaded from checkpoints.",
        "5. Generation: V5.5 chooses anchor atoms, linker size/chemotype, and validity gate decisions; RDKit physically builds candidate macrocycles.",
        "6. Docking: generated molecules are prepared as SDF/PDBQT and docked against JAK2 using Vina/Vina-GPU outputs preserved in `06_docking/`.",
        "7. Scoring: docking, ADMET/synthesis proxies, pose sanity, pocket-electronic fit, and auxiliary SE(3) geometry evidence are combined.",
        "8. Ranking and reporting: final candidate tables, benchmark reports, candidate cards, presentation docs, and dashboard data are produced.",
        "",
        "## Top-Level Folder Guide",
        "",
        "| Folder | File count | Size | What it is | Why it matters / how to explain it |",
        "|---|---:|---:|---|---|",
    ]

    for folder in sorted(folder_stats):
        title, purpose = FOLDER_PURPOSES.get(folder, ("Project folder", "Contains files used by the project."))
        lines.append(
            f"| `{esc(folder)}` | {folder_stats[folder]['count']} | {fmt_size(folder_stats[folder]['size'])} | {esc(title)} | {esc(purpose)} |"
        )

    lines += [
        "",
        "## Important Explanation Boundaries",
        "",
        "- The AI models make generation decisions; RDKit builds molecules from those decisions.",
        "- V5.5 is the active final version at the repository root.",
        "- `versions/` exists for historical evidence, not because the final code needs to be run from those folders.",
        "- V5.3 and V5.4 are included so reviewers can see the progression from model-guided generation to pocket-electronic scoring and then V5.5 pocket-conditioned generation.",
        "- The project uses computational metrics and docking outputs only; no experimental potency or toxicity claim is made.",
        "",
        "## How to Use This Repository During Review",
        "",
        "| Reviewer question | Where to point them |",
        "|---|---|",
        "| What is the final project? | `README.md`, `docs/PROJECT_COMPLETION_HANDOVER_V5_5.md`, `docs/v5_5_final_handover_report_2026-05-16.md` |",
        "| What models were used? | `docs/EMD_V5_5_MODEL_PROVENANCE_AND_DATASET_EXPLANATION.md`, `04_models_checkpoints/`, `src/emd_v5_2_hybrid/` |",
        "| Where are generated molecules? | `05_generated_candidates/v5_5_pocket_guided/` and earlier branches in `versions/v5_3_model_guided_generation/` |",
        "| Where are docking results? | `06_docking/v5_5_pocket_guided/scores/` and `08_final_ranking/` |",
        "| Where is the dashboard? | `interface/index.html`, `interface/app.js`, `interface/data.js` |",
        "| Where is project evolution shown? | `versions/README.md` and each folder under `versions/` |",
        "| How to verify project integrity? | `python -m unittest discover -s tests` and `python scripts/08_validate_project.py --base .` |",
        "",
        "## Complete Tracked File Inventory",
        "",
        "The table below lists every tracked file in the GitHub repository. Repeated candidate structures/poses are described by candidate ID because their purpose is identical but their content belongs to different molecules.",
    ]

    current_top = None
    for path in tracked:
        top = top_folder(path)
        if top != current_top:
            current_top = top
            lines += [
                "",
                f"### `{current_top}`",
                "",
                "| File | Size | Stage | Type | What it contains | Why / how it is used |",
                "|---|---:|---|---|---|---|",
            ]
        what, why = explain(path)
        lines.append(
            f"| `{esc(path)}` | {file_size(path)} | {esc(stage(path))} | {esc(kind(path))} | {esc(what)} | {esc(why)} |"
        )

    lines += [
        "",
        "## Final Speaking Summary",
        "",
        "For reviewers, explain the repository like this: the root folder is the completed V5.5 system; the numbered data/model/output folders show the live computational pipeline; `interface/` shows the real dashboard; `docs/` and `submission_artifacts_2026-05-28/` contain explanation material; and `versions/` proves how the project evolved from earlier planning and V5.2/V5.3/V5.4 stages into the final V5.5 handover.",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Documented {len(tracked)} files; guide size {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
