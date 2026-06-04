# EMD V5.2 Hybrid Pipeline
# ========================
# Complete drug discovery pipeline for JAK2 macrocyclic inhibitor design
# using SE(3) equivariant flow matching + classical baselines.

__version__ = "5.2.1"
__project__ = "ElectroMacroDiff V5.2 Hybrid"

# ── M0: Configuration & Registry ──────────────────────────
from emd_pipeline.config_registry import (
    load_config,
    ensure_dirs,
    set_all_seeds,
    load_progress,
    save_progress,
    update_registry,
    save_environment_versions,
)

# ── M1: Data Collection ───────────────────────────────────
from emd_pipeline.data_collection import (
    fetch_chembl_jak2_activities,
    fetch_pdb_structure,
    curate_activities,
    scaffold_split,
    analyze_macrocycles,
)

# ── M2: Feature Engineering ───────────────────────────────
from emd_pipeline.feature_engineering import (
    compute_descriptors,
    generate_conformers,
    build_se3_graph_tensors,
)

# ── M3: SE(3) Flow Matching Model ─────────────────────────
from emd_pipeline.se3_flow_model import (
    MolecularGraphDataset,
    collate_fn,
    SE3FlowMatchingModel,
    SE3InvariantLayer,
    flow_matching_loss,
    train_se3_model,
    generate_se3_candidates,
    coords_to_smiles,
    plot_training_curve,
)

# ── M4: Candidate Generation ──────────────────────────────
from emd_pipeline.candidate_generation import (
    generate_selfies_candidates,
    generate_rdkit_candidates,
    filter_candidates,
    merge_candidates,
    compute_generation_metrics,
)

# ── M5: Docking ───────────────────────────────────────────
from emd_pipeline.docking import (
    prepare_receptor,
    dock_candidates,
    prepare_ligands_for_docking,
    run_vina_docking,
    extract_pocket_center,
    visualize_top_poses,
)

# ── M6: ADMET & Synthesis ─────────────────────────────────
from emd_pipeline.admet_synthesis import (
    compute_admet_scores,
    compute_synthesis_scores,
    compute_safety_proxy,
    apply_filter_flags,
)

# ── M7: Ranking & Report ──────────────────────────────────
from emd_pipeline.ranking_report import (
    consensus_ranking,
    generate_molecule_cards,
    generate_tpp_report,
    save_top_sdf,
)

# ── Visualization ─────────────────────────────────────────
from emd_pipeline.visualization import (
    plot_dataset_overview,
    plot_training_dashboard,
    plot_generation_comparison,
    plot_ranking_dashboard,
    draw_molecule_grid,
    render_3d_molecule,
)

# ── Colab Utilities ───────────────────────────────────────
from emd_pipeline.colab_utils import (
    check_gpu_status,
    get_optimal_batch_size,
    TimedSection,
    safe_save,
    check_drive_artifacts,
)

# ── Campaign Dashboard ────────────────────────────────────
from emd_pipeline.campaign_dashboard import (
    generate_daily_log,
    check_campaign_status,
)

__all__ = [
    # Config (M0)
    "load_config", "ensure_dirs", "set_all_seeds",
    "load_progress", "save_progress", "update_registry",
    "save_environment_versions",
    # Data (M1)
    "fetch_chembl_jak2_activities", "fetch_pdb_structure",
    "curate_activities", "scaffold_split", "analyze_macrocycles",
    # Features (M2)
    "compute_descriptors", "generate_conformers", "build_se3_graph_tensors",
    # Model (M3)
    "MolecularGraphDataset", "collate_fn", "SE3FlowMatchingModel",
    "SE3InvariantLayer", "flow_matching_loss",
    "train_se3_model", "generate_se3_candidates", "coords_to_smiles",
    "plot_training_curve",
    # Generation (M4)
    "generate_selfies_candidates", "generate_rdkit_candidates",
    "filter_candidates", "merge_candidates", "compute_generation_metrics",
    # Docking (M5)
    "prepare_receptor", "dock_candidates", "prepare_ligands_for_docking",
    "run_vina_docking", "extract_pocket_center", "visualize_top_poses",
    # ADMET (M6)
    "compute_admet_scores", "compute_synthesis_scores",
    "compute_safety_proxy", "apply_filter_flags",
    # Ranking (M7)
    "consensus_ranking", "generate_molecule_cards",
    "generate_tpp_report", "save_top_sdf",
    # Visualization
    "plot_dataset_overview", "plot_training_dashboard",
    "plot_generation_comparison", "plot_ranking_dashboard",
    "draw_molecule_grid", "render_3d_molecule",
    # Utils
    "check_gpu_status", "get_optimal_batch_size", "TimedSection",
    "safe_save", "check_drive_artifacts",
    # Dashboard
    "generate_daily_log", "check_campaign_status",
]
