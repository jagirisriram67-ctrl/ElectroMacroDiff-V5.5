"""
EMD V5.2 Hybrid - Pipeline Integration Test
============================================
Tests the entire pipeline end-to-end with synthetic data
to verify all modules work before running on real data in Colab.

Usage:
    python test_pipeline.py
"""

import os
import sys
import json
import time
import warnings
import tempfile
import shutil

warnings.filterwarnings("ignore")

# Ensure pipeline is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASSED = 0
FAILED = 0
SKIPPED = 0


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global PASSED, FAILED, SKIPPED
            try:
                print(f"\n{'='*60}")
                print(f"TEST: {name}")
                print(f"{'='*60}")
                result = func()
                if result == "skip":
                    SKIPPED += 1
                    print(f"  [SKIP] {name}")
                else:
                    PASSED += 1
                    print(f"  [PASS] {name}")
            except Exception as e:
                FAILED += 1
                print(f"  [FAIL] {name}: {e}")
                import traceback
                traceback.print_exc()
        return wrapper
    return decorator


# Create temp directory for test outputs
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_output")
os.makedirs(TEST_DIR, exist_ok=True)


# ================================================================
# M0: Config & Registry
# ================================================================

@test("M0: Load campaign config")
def test_config():
    from emd_pipeline.config_registry import load_config
    config = load_config("campaign_config.yaml")
    assert config["project"]["name"] == "EMD_V5_2_Hybrid"
    assert config["target"]["primary_pdb"] == "5AEP"
    assert config["seeds"]["random_seed"] == 42
    print(f"  Config loaded: {config['project']['name']}")


@test("M0: Create directory structure")
def test_dirs():
    from emd_pipeline.config_registry import ensure_dirs
    test_base = os.path.join(TEST_DIR, "drive_test")
    ensure_dirs(test_base)
    assert os.path.isdir(os.path.join(test_base, "00_project_registry"))
    assert os.path.isdir(os.path.join(test_base, "04_models_checkpoints", "se3_flow"))
    assert os.path.isdir(os.path.join(test_base, "08_final_ranking"))


@test("M0: Progress save/load")
def test_progress():
    from emd_pipeline.config_registry import load_progress, save_progress
    prog_path = os.path.join(TEST_DIR, "test_progress.json")
    save_progress(prog_path, {"step": "test", "value": 42})
    loaded = load_progress(prog_path)
    assert loaded["step"] == "test"
    assert loaded["value"] == 42
    assert "updated_at" in loaded


@test("M0: Registry update")
def test_registry():
    from emd_pipeline.config_registry import update_registry
    reg_path = os.path.join(TEST_DIR, "test_registry.csv")
    update_registry(reg_path, {"notebook": "test", "step": "init", "status": "ok"})
    import pandas as pd
    df = pd.read_csv(reg_path)
    assert len(df) == 1
    assert df.iloc[0]["notebook"] == "test"


@test("M0: Seed management")
def test_seeds():
    from emd_pipeline.config_registry import set_all_seeds
    set_all_seeds(42)
    import numpy as np
    a = np.random.rand()
    set_all_seeds(42)
    b = np.random.rand()
    assert a == b, "Seeds not deterministic"


# ================================================================
# M1: Data Collection (synthetic test)
# ================================================================

@test("M1: SMILES validation and curation")
def test_curation():
    import pandas as pd
    from emd_pipeline.data_collection import curate_activities

    raw_data = pd.DataFrame({
        "canonical_smiles": [
            "c1ccc2c(c1)cc1ccccc1n2",       # valid
            "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
            "INVALID_SMILES",                  # invalid
            "c1ccccc1",                        # benzene (MW too low)
            "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",  # steroid
        ],
        "activity_value": [100, 50, 200, 300, 75],
        "activity_units": ["nM", "nM", "nM", "nM", "nM"],
        "activity_type": ["IC50"] * 5,
        "source": ["test"] * 5,
    })

    curated = curate_activities(raw_data, mw_min=150, mw_max=900)
    assert len(curated) > 0, "No molecules passed curation"
    assert "canonical_smiles" in curated.columns
    assert "mol_id" in curated.columns
    assert "max_ring_size" in curated.columns
    print(f"  Curated: {len(curated)} molecules from {len(raw_data)} raw")


@test("M1: Scaffold split")
def test_split():
    import pandas as pd
    from emd_pipeline.data_collection import scaffold_split

    df = pd.DataFrame({
        "canonical_smiles": [
            "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1",
            "C1CCCCC1", "c1ccc(-c2ccccc2)cc1", "c1ccoc1",
            "c1ccsc1", "c1cc[nH]c1", "C1CCNCC1", "c1ccc(O)cc1",
        ],
        "mol_id": [f"M{i}" for i in range(10)],
    })

    split_df = scaffold_split(df, train_frac=0.6, val_frac=0.2, seed=42)
    assert "split" in split_df.columns
    assert set(split_df["split"].unique()).issubset({"train", "val", "test"})


@test("M1: PDB download (5AEP)")
def test_pdb():
    from emd_pipeline.data_collection import fetch_pdb_structure
    pdb_dir = os.path.join(TEST_DIR, "pdb")
    try:
        path = fetch_pdb_structure("5AEP", pdb_dir)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000
        print(f"  Downloaded: {os.path.getsize(path)} bytes")
    except Exception as e:
        print(f"  Network error (expected in offline): {e}")
        return "skip"


# ================================================================
# M2: Feature Engineering
# ================================================================

@test("M2: Descriptor computation")
def test_descriptors():
    import pandas as pd
    from emd_pipeline.feature_engineering import compute_descriptors

    df = pd.DataFrame({
        "canonical_smiles": [
            "c1ccc2c(c1)cc1ccccc1n2",
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O",
        ],
        "mol_id": ["T001", "T002", "T003"],
    })

    features = compute_descriptors(df)
    assert len(features) == 3
    assert "mw" in features.columns
    assert "qed" in features.columns
    assert "logp" in features.columns
    assert all(features["passes_rdkit"])


@test("M2: Conformer generation")
def test_conformers():
    import pandas as pd
    from emd_pipeline.feature_engineering import generate_conformers

    df = pd.DataFrame({
        "canonical_smiles": ["c1ccccc1", "CC(=O)O", "c1ccncc1"],
        "mol_id": ["C001", "C002", "C003"],
    })

    sdf_path = os.path.join(TEST_DIR, "test_conformers.sdf")
    mols = generate_conformers(df, output_sdf=sdf_path)
    assert len(mols) >= 2
    assert os.path.exists(sdf_path)


@test("M2: SE(3) graph tensor construction")
def test_graph_tensors():
    import pandas as pd
    from emd_pipeline.feature_engineering import generate_conformers, build_se3_graph_tensors

    df = pd.DataFrame({
        "canonical_smiles": [
            "c1ccc2c(c1)cc1ccccc1n2",
            "CC(=O)Oc1ccccc1C(=O)O",
            "c1ccncc1",
        ],
        "mol_id": ["G001", "G002", "G003"],
    })

    mols = generate_conformers(df)
    pt_path = os.path.join(TEST_DIR, "test_graphs.pt")
    idx_path = os.path.join(TEST_DIR, "test_graph_index.csv")

    graphs = build_se3_graph_tensors(mols, output_pt=pt_path, output_index=idx_path)
    assert len(graphs) >= 2
    assert "atom_features" in graphs[0]
    assert "coords" in graphs[0]
    assert "edge_index" in graphs[0]
    print(f"  Graph 0: atoms={graphs[0]['num_atoms']}, features={graphs[0]['atom_features'].shape}")


# ================================================================
# M3: SE(3) Flow Matching Model
# ================================================================

@test("M3: Dataset and DataLoader")
def test_dataloader():
    import torch
    from emd_pipeline.se3_flow_model import MolecularGraphDataset, collate_fn
    from torch.utils.data import DataLoader

    graphs = torch.load(os.path.join(TEST_DIR, "test_graphs.pt"), weights_only=False)
    dataset = MolecularGraphDataset(graphs, max_atoms=40)
    assert len(dataset) > 0

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert "atom_features" in batch
    assert batch["atom_features"].shape[0] <= 2
    assert batch["atom_features"].shape[1] == 40
    print(f"  Batch shapes: features={batch['atom_features'].shape}, coords={batch['coords'].shape}")


@test("M3: Model forward pass")
def test_model_forward():
    import torch
    from emd_pipeline.se3_flow_model import (
        MolecularGraphDataset, collate_fn, SE3FlowMatchingModel, flow_matching_loss
    )
    from torch.utils.data import DataLoader

    graphs = torch.load(os.path.join(TEST_DIR, "test_graphs.pt"), weights_only=False)
    dataset = MolecularGraphDataset(graphs, max_atoms=40)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    atom_dim = batch["atom_features"].shape[2]
    model = SE3FlowMatchingModel(
        atom_feature_dim=atom_dim, hidden_dim=32, num_layers=2,
        num_heads=2, max_atoms=40
    )

    device = torch.device("cpu")
    loss = flow_matching_loss(model, batch, device)
    assert torch.isfinite(loss)
    print(f"  Forward loss: {loss.item():.6f}")


@test("M3: Model backward pass + optimizer step")
def test_model_backward():
    import torch
    from emd_pipeline.se3_flow_model import (
        MolecularGraphDataset, collate_fn, SE3FlowMatchingModel, flow_matching_loss
    )
    from torch.utils.data import DataLoader

    graphs = torch.load(os.path.join(TEST_DIR, "test_graphs.pt"), weights_only=False)
    dataset = MolecularGraphDataset(graphs, max_atoms=40)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    atom_dim = batch["atom_features"].shape[2]
    model = SE3FlowMatchingModel(
        atom_feature_dim=atom_dim, hidden_dim=32, num_layers=2,
        num_heads=2, max_atoms=40
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = flow_matching_loss(model, batch, torch.device("cpu"))
    loss.backward()
    optimizer.step()

    # Verify gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed"
    print(f"  Backward + step OK, loss={loss.item():.6f}")


@test("M3: Checkpoint save/reload")
def test_checkpoint():
    import torch
    from emd_pipeline.se3_flow_model import SE3FlowMatchingModel

    model = SE3FlowMatchingModel(atom_feature_dim=21, hidden_dim=32, num_layers=2, num_heads=2, max_atoms=40)
    ckpt_path = os.path.join(TEST_DIR, "test_checkpoint.pt")

    ckpt = {"epoch": 5, "model_state_dict": model.state_dict(), "train_loss": 0.5}
    torch.save(ckpt, ckpt_path)

    model2 = SE3FlowMatchingModel(atom_feature_dim=21, hidden_dim=32, num_layers=2, num_heads=2, max_atoms=40)
    loaded = torch.load(ckpt_path, weights_only=False)
    model2.load_state_dict(loaded["model_state_dict"])
    assert loaded["epoch"] == 5
    print(f"  Checkpoint save/reload OK")


# ================================================================
# M4: Candidate Generation
# ================================================================

@test("M4: SELFIES generation")
def test_selfies():
    import pandas as pd
    from emd_pipeline.candidate_generation import generate_selfies_candidates

    seeds = pd.DataFrame({
        "canonical_smiles": ["c1ccccc1", "c1ccncc1", "c1ccc(O)cc1"],
        "mol_id": ["S001", "S002", "S003"],
    })

    result = generate_selfies_candidates(seeds, mutations_per_seed=5)
    assert len(result) > 0
    assert "canonical_smiles" in result.columns
    print(f"  Generated: {len(result)} SELFIES candidates")


@test("M4: RDKit generation")
def test_rdkit_gen():
    import pandas as pd
    from emd_pipeline.candidate_generation import generate_rdkit_candidates

    seeds = pd.DataFrame({
        "canonical_smiles": ["c1ccc(N)cc1", "c1ccc(O)cc1", "c1ccc(F)cc1"],
        "mol_id": ["R001", "R002", "R003"],
    })

    result = generate_rdkit_candidates(seeds, subs_per_seed=3)
    assert len(result) > 0
    print(f"  Generated: {len(result)} RDKit candidates")


@test("M4: Filtering and merging")
def test_filter_merge():
    import pandas as pd
    from emd_pipeline.candidate_generation import (
        generate_selfies_candidates, generate_rdkit_candidates,
        filter_candidates, merge_candidates
    )

    seeds = pd.DataFrame({
        "canonical_smiles": ["c1ccccc1", "c1ccncc1"],
        "mol_id": ["FM01", "FM02"],
    })

    selfies = generate_selfies_candidates(seeds, mutations_per_seed=3)
    rdkit = generate_rdkit_candidates(seeds, subs_per_seed=2)

    sf_filtered = filter_candidates(selfies, mw_range=(50, 500))
    rk_filtered = filter_candidates(rdkit, mw_range=(50, 500))

    merged = merge_candidates(sf_filtered, rk_filtered)
    assert len(merged) > 0
    assert "candidate_id" in merged.columns
    print(f"  Merged: {len(merged)} unique candidates")


# ================================================================
# M6: ADMET & Synthesis
# ================================================================

@test("M6: ADMET scores")
def test_admet():
    import pandas as pd
    from emd_pipeline.admet_synthesis import compute_admet_scores

    df = pd.DataFrame({
        "canonical_smiles": ["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccncc1"],
        "candidate_id": ["A001", "A002", "A003"],
    })

    admet = compute_admet_scores(df)
    assert len(admet) == 3
    assert "qed" in admet.columns
    assert "admet_score" in admet.columns
    assert all(admet["admet_score"] >= 0)


@test("M6: Synthesis scores")
def test_synthesis():
    import pandas as pd
    from emd_pipeline.admet_synthesis import compute_synthesis_scores

    df = pd.DataFrame({
        "canonical_smiles": ["c1ccccc1", "c1ccncc1"],
        "candidate_id": ["SY01", "SY02"],
    })

    synth = compute_synthesis_scores(df)
    assert len(synth) == 2
    assert "sa_score" in synth.columns
    assert "synthesis_score" in synth.columns


@test("M6: Safety proxy")
def test_safety():
    import pandas as pd
    from emd_pipeline.admet_synthesis import compute_safety_proxy

    df = pd.DataFrame({
        "canonical_smiles": ["c1ccccc1", "[O-][N+](=O)c1ccccc1"],  # benzene, nitrobenzene
        "candidate_id": ["SF01", "SF02"],
    })

    safety = compute_safety_proxy(df)
    assert len(safety) == 2
    # Nitrobenzene should have lower safety score
    nitro_score = safety[safety["candidate_id"] == "SF02"]["safety_proxy_score"].iloc[0]
    benz_score = safety[safety["candidate_id"] == "SF01"]["safety_proxy_score"].iloc[0]
    assert nitro_score < benz_score, "Safety proxy should flag nitro group"


# ================================================================
# M7: Ranking
# ================================================================

@test("M7: Consensus ranking")
def test_ranking():
    import pandas as pd
    from emd_pipeline.ranking_report import consensus_ranking

    docking = pd.DataFrame({
        "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
        "smiles": ["c1ccccc1", "c1ccncc1", "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(F)cc1"],
        "best_score": [-8.5, -7.2, -9.1, -6.0, -8.0],
    })
    admet = pd.DataFrame({
        "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
        "qed": [0.5, 0.6, 0.7, 0.4, 0.8],
        "admet_score": [0.6, 0.7, 0.8, 0.5, 0.9],
        "lipinski_violations": [0, 0, 0, 1, 0],
    })
    synth = pd.DataFrame({
        "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
        "sa_score": [2.0, 3.0, 2.5, 4.0, 1.5],
        "synthesis_score": [0.9, 0.8, 0.85, 0.6, 0.95],
    })
    safety = pd.DataFrame({
        "candidate_id": ["C1", "C2", "C3", "C4", "C5"],
        "safety_proxy_score": [1.0, 0.8, 0.9, 0.5, 1.0],
    })

    ranked = consensus_ranking(docking, admet, synth, safety)
    assert len(ranked) == 5
    assert ranked.iloc[0]["rank"] == 1
    assert "final_weighted_score" in ranked.columns
    assert "decision" in ranked.columns
    print(f"  Top candidate: {ranked.iloc[0]['candidate_id']} score={ranked.iloc[0]['final_weighted_score']:.4f}")


@test("M7: Molecule cards")
def test_cards():
    import pandas as pd
    from emd_pipeline.ranking_report import generate_molecule_cards

    ranked = pd.DataFrame({
        "rank": [1, 2, 3],
        "candidate_id": ["C1", "C2", "C3"],
        "smiles": ["c1ccccc1", "c1ccncc1", "c1ccc(O)cc1"],
        "best_score": [-9.1, -8.5, -7.2],
        "final_weighted_score": [0.85, 0.78, 0.72],
        "qed": [0.7, 0.5, 0.6],
        "sa_score": [2.5, 2.0, 3.0],
        "admet_score": [0.8, 0.6, 0.7],
        "safety_proxy_score": [0.9, 1.0, 0.8],
        "decision": ["CANDIDATE", "CANDIDATE", "CANDIDATE"],
        "main_risk": ["low risk", "low risk", "low risk"],
        "inchikey": ["", "", ""],
    })

    cards_path = os.path.join(TEST_DIR, "test_cards.xlsx")
    cards = generate_molecule_cards(ranked, cards_path, top_n=3)
    assert len(cards) == 3


@test("M7: TPP report generation")
def test_tpp():
    import pandas as pd
    from emd_pipeline.ranking_report import generate_tpp_report
    from emd_pipeline.config_registry import load_config

    config = load_config("campaign_config.yaml")

    ranked = pd.DataFrame({
        "rank": [1, 2, 3],
        "candidate_id": ["C1", "C2", "C3"],
        "smiles": ["c1ccccc1", "c1ccncc1", "c1ccc(O)cc1"],
        "best_score": [-9.1, -8.5, -7.2],
        "final_weighted_score": [0.85, 0.78, 0.72],
        "source_generator": ["SE3", "SELFIES", "RDKit"],
        "decision": ["CANDIDATE", "CANDIDATE", "CANDIDATE"],
        "main_risk": ["low risk", "low risk", "low risk"],
    })

    report_dir = os.path.join(TEST_DIR, "reports")
    path = generate_tpp_report(ranked, config, report_dir)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "ElectroMacroDiff" in content
    assert "JAK2" in content


# ================================================================
# M3+: SE(3) Generation (end-to-end)
# ================================================================

@test("M3+: SE(3) ODE generation")
def test_se3_generation():
    import torch
    from emd_pipeline.se3_flow_model import (
        MolecularGraphDataset, SE3FlowMatchingModel, generate_se3_candidates
    )

    graphs = torch.load(os.path.join(TEST_DIR, "test_graphs.pt"), weights_only=False)
    model = SE3FlowMatchingModel(
        atom_feature_dim=graphs[0]["atom_features"].shape[1],
        hidden_dim=32, num_layers=2, num_heads=2, max_atoms=40
    )

    generated = generate_se3_candidates(model, graphs[:2], num_steps=10, device="cpu")
    assert len(generated) > 0, "No molecules generated"
    assert generated[0]["coords"].shape[1] == 3
    print(f"  Generated {len(generated)} molecules, first has {generated[0]['num_atoms']} atoms")


@test("M3+: Coord-to-SMILES reconstruction")
def test_coord_to_smiles():
    import torch
    from emd_pipeline.se3_flow_model import (
        SE3FlowMatchingModel, generate_se3_candidates, coords_to_smiles
    )

    graphs = torch.load(os.path.join(TEST_DIR, "test_graphs.pt"), weights_only=False)
    model = SE3FlowMatchingModel(
        atom_feature_dim=graphs[0]["atom_features"].shape[1],
        hidden_dim=32, num_layers=2, num_heads=2, max_atoms=40
    )

    generated = generate_se3_candidates(model, graphs[:2], num_steps=5, device="cpu")
    smiles_results = coords_to_smiles(generated, graphs)
    assert len(smiles_results) > 0, "No SMILES reconstructed"
    print(f"  Reconstructed {len(smiles_results)} SMILES from {len(generated)} coords")
# ================================================================
# Visualization Module
# ================================================================

@test("VIZ: Dataset overview plot")
def test_viz_dataset():
    import pandas as pd
    from emd_pipeline.visualization import plot_dataset_overview
    df = pd.DataFrame({
        "mw": [300, 400, 350, 500, 250],
        "logp": [2.1, 3.5, 1.8, 4.2, 0.9],
        "qed": [0.6, 0.7, 0.5, 0.4, 0.8],
        "rotatable_bonds": [3, 5, 2, 7, 1],
        "max_ring_size": [6, 6, 5, 6, 5],
        "split": ["train", "train", "train", "val", "test"],
    })
    out = os.path.join(TEST_DIR, "viz_dataset.png")
    plot_dataset_overview(df, output_path=out)
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000


@test("VIZ: Ranking dashboard")
def test_viz_ranking():
    import pandas as pd
    from emd_pipeline.visualization import plot_ranking_dashboard
    ranked = pd.DataFrame({
        "rank": list(range(1, 11)),
        "candidate_id": [f"C{i}" for i in range(10)],
        "final_weighted_score": [0.9 - i * 0.05 for i in range(10)],
        "best_score": [-9 + i * 0.5 for i in range(10)],
        "admet_score": [0.8 - i * 0.03 for i in range(10)],
        "docking_score_norm": [0.9 - i * 0.04 for i in range(10)],
        "synthesis_score": [0.85 - i * 0.02 for i in range(10)],
        "safety_proxy_score": [1.0 - i * 0.05 for i in range(10)],
        "source_generator": ["SE3", "SELFIES", "RDKit"] * 3 + ["SE3"],
        "decision": ["CANDIDATE"] * 5 + ["BACKUP"] * 5,
    })
    out = os.path.join(TEST_DIR, "viz_ranking.png")
    plot_ranking_dashboard(ranked, output_path=out)
    assert os.path.exists(out)


@test("VIZ: Molecule grid")
def test_viz_grid():
    from emd_pipeline.visualization import draw_molecule_grid
    smiles = ["c1ccccc1", "c1ccncc1", "c1ccc(O)cc1", "c1ccc(N)cc1"]
    legends = ["Benzene", "Pyridine", "Phenol", "Aniline"]
    out = os.path.join(TEST_DIR, "viz_grid.png")
    grid = draw_molecule_grid(smiles, legends, output_path=out)
    assert grid is not None
    assert os.path.exists(out)


# ================================================================
# Campaign Dashboard
# ================================================================

@test("DASH: Daily log generation")
def test_daily_log():
    from emd_pipeline.campaign_dashboard import generate_daily_log
    log_path = os.path.join(TEST_DIR, "daily_log.md")
    report = generate_daily_log(TEST_DIR, output_path=log_path)
    assert os.path.exists(log_path)
    assert "ElectroMacroDiff" in report
    print(f"  Log generated: {len(report)} chars")


@test("DASH: Campaign status check")
def test_campaign_status():
    from emd_pipeline.campaign_dashboard import check_campaign_status
    status = check_campaign_status(TEST_DIR)
    assert "overall_pct" in status
    assert isinstance(status["overall_pct"], float)
    print(f"  Campaign status: {status['overall_pct']:.0f}% complete")


# ================================================================
# Colab Utilities
# ================================================================

@test("UTILS: GPU status check")
def test_gpu_status():
    from emd_pipeline.colab_utils import check_gpu_status
    info = check_gpu_status()
    assert "gpu_available" in info
    assert "device_name" in info


@test("UTILS: Batch size estimation")
def test_batch_size():
    from emd_pipeline.colab_utils import get_optimal_batch_size
    bs = get_optimal_batch_size(80, hidden_dim=128, gpu_memory_gb=16)
    assert 1 <= bs <= 32


@test("UTILS: TimedSection context manager")
def test_timed_section():
    from emd_pipeline.colab_utils import TimedSection
    with TimedSection("Test Section") as ts:
        time.sleep(0.05)


# ================================================================
# Run all tests
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ElectroMacroDiff V5.2 Hybrid - Pipeline Integration Tests")
    print("=" * 60)

    start_time = time.time()

    tests = [
        test_config, test_dirs, test_progress, test_registry, test_seeds,
        test_curation, test_split, test_pdb,
        test_descriptors, test_conformers, test_graph_tensors,
        test_dataloader, test_model_forward, test_model_backward, test_checkpoint,
        test_selfies, test_rdkit_gen, test_filter_merge,
        test_admet, test_synthesis, test_safety,
        test_ranking, test_cards, test_tpp,
        test_se3_generation, test_coord_to_smiles,
        test_viz_dataset, test_viz_ranking, test_viz_grid,
        test_daily_log, test_campaign_status,
        test_gpu_status, test_batch_size, test_timed_section,
    ]

    for t in tests:
        t()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)

    if FAILED == 0:
        print("\nAll pipeline modules verified! Ready for Colab deployment.")
    else:
        print(f"\n{FAILED} test(s) need attention before deployment.")

    # Cleanup
    try:
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    except:
        pass

    sys.exit(FAILED)
