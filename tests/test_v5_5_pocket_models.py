"""Unit tests for V5.5 pocket-conditioned models and features."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT / "src"))


class PocketFeatureTests(unittest.TestCase):
    """Test pocket electronic feature extraction."""

    def test_pocket_feature_vector_dimension(self):
        from emd_v5_2_hybrid.pocket_features import pocket_feature_vector, POCKET_FEATURE_DIM
        profile = {
            "pocket_atom_count": 235,
            "pocket_net_charge_proxy": -0.011,
            "hydrophobic_atom_count": 45,
            "hbond_donor_atom_count": 60,
            "hbond_acceptor_atom_count": 75,
            "aromatic_atom_count": 15,
            "top_contact_residues": [
                {"residue": "ARG:A:980", "atom_contacts": 16},
                {"residue": "ASP:A:939", "atom_contacts": 9},
            ],
        }
        vec = pocket_feature_vector(profile)
        self.assertEqual(len(vec), POCKET_FEATURE_DIM)

    def test_pocket_feature_values_in_range(self):
        from emd_v5_2_hybrid.pocket_features import pocket_feature_vector
        profile = {
            "pocket_atom_count": 235,
            "pocket_net_charge_proxy": -0.5,
            "hydrophobic_atom_count": 45,
            "hbond_donor_atom_count": 60,
            "hbond_acceptor_atom_count": 75,
            "aromatic_atom_count": 15,
            "top_contact_residues": [],
        }
        vec = pocket_feature_vector(profile)
        for i, val in enumerate(vec):
            self.assertTrue(-2.0 <= val <= 2.0, f"Feature {i} = {val} out of range")

    def test_pocket_feature_deterministic(self):
        from emd_v5_2_hybrid.pocket_features import pocket_feature_vector
        profile = {"pocket_atom_count": 100, "hydrophobic_atom_count": 20,
                    "hbond_donor_atom_count": 30, "hbond_acceptor_atom_count": 40,
                    "aromatic_atom_count": 10, "pocket_net_charge_proxy": 0.1,
                    "top_contact_residues": []}
        vec1 = pocket_feature_vector(profile)
        vec2 = pocket_feature_vector(profile)
        self.assertEqual(vec1, vec2)

    def test_aa_fingerprint_populates_correctly(self):
        from emd_v5_2_hybrid.pocket_features import pocket_feature_vector, AA_INDEX
        profile = {
            "pocket_atom_count": 100,
            "hydrophobic_atom_count": 20,
            "hbond_donor_atom_count": 30,
            "hbond_acceptor_atom_count": 40,
            "aromatic_atom_count": 10,
            "pocket_net_charge_proxy": 0.0,
            "top_contact_residues": [
                {"residue": "ARG:A:100", "atom_contacts": 10},
            ],
        }
        vec = pocket_feature_vector(profile)
        # ARG should have non-zero value in the AA fingerprint portion
        arg_idx = 12 + AA_INDEX["ARG"]  # 12 = offset of AA fingerprint in vector
        self.assertGreater(vec[arg_idx], 0.0)

    def test_load_real_pocket_profile(self):
        from emd_v5_2_hybrid.pocket_features import load_pocket_feature_vector, POCKET_FEATURE_DIM
        pocket_json = ROOT / "06_docking" / "v5_3_model_guided" / "scores" / "jak2_pocket_electronic_profile.json"
        if pocket_json.exists():
            vec = load_pocket_feature_vector(pocket_json)
            self.assertEqual(len(vec), POCKET_FEATURE_DIM)


class PocketAnchorGNNTests(unittest.TestCase):
    """Test PocketAnchorGNN model architecture."""

    def test_forward_pass_shape(self):
        import torch
        from emd_v5_2_hybrid.pocket_anchor_gnn import PocketAnchorGNN
        model = PocketAnchorGNN(node_dim=64, pocket_dim=32, hidden_dim=64, num_layers=2)
        x = torch.randn(10, 64)
        ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        ef = torch.randn(4, 6)
        pv = torch.randn(32)
        out = model(x, ei, ef, pv)
        self.assertEqual(out.shape, (10,))

    def test_output_per_atom(self):
        """Output should have one score per atom."""
        import torch
        from emd_v5_2_hybrid.pocket_anchor_gnn import PocketAnchorGNN
        for n_atoms in [5, 15, 30]:
            model = PocketAnchorGNN(node_dim=64, pocket_dim=32, hidden_dim=32, num_layers=2)
            x = torch.randn(n_atoms, 64)
            ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            ef = torch.randn(2, 6)
            pv = torch.randn(32)
            out = model(x, ei, ef, pv)
            self.assertEqual(out.shape[0], n_atoms)

    def test_empty_edges(self):
        """Model should handle molecules with no bonds."""
        import torch
        from emd_v5_2_hybrid.pocket_anchor_gnn import PocketAnchorGNN
        model = PocketAnchorGNN(node_dim=64, pocket_dim=32, hidden_dim=32, num_layers=2)
        x = torch.randn(3, 64)
        ei = torch.empty((2, 0), dtype=torch.long)
        ef = torch.empty((0, 6))
        pv = torch.randn(32)
        out = model(x, ei, ef, pv)
        self.assertEqual(out.shape, (3,))

    def test_checkpoint_save_load(self):
        import torch
        import tempfile
        from emd_v5_2_hybrid.pocket_anchor_gnn import PocketAnchorGNN, save_pocket_anchor_gnn_checkpoint, load_pocket_anchor_gnn
        model = PocketAnchorGNN(node_dim=64, pocket_dim=32, hidden_dim=32, num_layers=2)
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            path = Path(tmpdir) / "test_anchor_gnn.pt"
            save_pocket_anchor_gnn_checkpoint(path, model, {"test": True})
            loaded_model, ckpt = load_pocket_anchor_gnn(path)
            self.assertTrue(ckpt["test"])
            x = torch.randn(5, 64)
            ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            ef = torch.randn(2, 6)
            pv = torch.randn(32)
            model.eval()
            loaded_model.eval()
            out1 = model(x, ei, ef, pv)
            out2 = loaded_model(x, ei, ef, pv)
            self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


class PocketLinkerPolicyTests(unittest.TestCase):
    """Test PocketLinkerPolicy dual-head model."""

    def test_forward_pass_shapes(self):
        import torch
        from emd_v5_2_hybrid.pocket_linker_policy import PocketLinkerPolicy
        model = PocketLinkerPolicy(linker_feature_dim=16, pocket_dim=32, hidden_dim=64)
        x = torch.randn(4, 16)
        pv = torch.randn(32)
        size_logits, chemo_logits = model(x, pv)
        self.assertEqual(size_logits.shape, (4, 10))
        self.assertEqual(chemo_logits.shape, (4, 11))

    def test_pocket_broadcast_1d(self):
        """Pocket vector should broadcast from 1D to batch."""
        import torch
        from emd_v5_2_hybrid.pocket_linker_policy import PocketLinkerPolicy
        model = PocketLinkerPolicy(linker_feature_dim=16, pocket_dim=32, hidden_dim=32)
        x = torch.randn(8, 16)
        pv = torch.randn(32)  # 1D, should be broadcast
        size_logits, chemo_logits = model(x, pv)
        self.assertEqual(size_logits.shape[0], 8)

    def test_checkpoint_roundtrip(self):
        import torch
        import tempfile
        from emd_v5_2_hybrid.pocket_linker_policy import PocketLinkerPolicy, save_pocket_linker_policy_checkpoint, load_pocket_linker_policy
        model = PocketLinkerPolicy(linker_feature_dim=16, pocket_dim=32, hidden_dim=32)
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            path = Path(tmpdir) / "test_linker_policy.pt"
            save_pocket_linker_policy_checkpoint(path, model, {"epoch": 10})
            loaded, sl, cl, ckpt = load_pocket_linker_policy(path)
            self.assertEqual(ckpt["epoch"], 10)
            self.assertEqual(len(sl), 10)
            self.assertEqual(len(cl), 11)


class ValidityRewardTests(unittest.TestCase):
    """Test PocketValidityRewardModel."""

    def test_forward_pass(self):
        import torch
        from emd_v5_2_hybrid.validity_reward import PocketValidityRewardModel, REWARD_INPUT_DIM
        model = PocketValidityRewardModel(input_dim=REWARD_INPUT_DIM, hidden_dim=64)
        x = torch.randn(8, REWARD_INPUT_DIM)
        out = model(x)
        self.assertEqual(out.shape, (8,))

    def test_attempt_to_feature_vector(self):
        from emd_v5_2_hybrid.validity_reward import attempt_to_feature_vector, REWARD_INPUT_DIM
        pocket_vec = [0.0] * 32
        attempt = {
            "anchor_score_a": 0.8,
            "anchor_score_b": 0.6,
            "linker_length": 5,
            "target_ring": 16,
            "path_bonds": 11,
            "linker_probability": 0.7,
            "chemotype": "oxa",
        }
        vec = attempt_to_feature_vector(attempt, pocket_vec)
        self.assertEqual(len(vec), REWARD_INPUT_DIM)

    def test_attempt_label(self):
        from emd_v5_2_hybrid.validity_reward import attempt_label
        self.assertEqual(attempt_label("valid_output"), 1.0)
        self.assertEqual(attempt_label("ring_closure_failed"), 0.0)
        self.assertEqual(attempt_label("rejected_known_linker"), 0.0)


class DeviceHelperTests(unittest.TestCase):
    """Test device resolution utility."""

    def test_cpu_always_works(self):
        from emd_v5_2_hybrid.device_helper import resolve_device
        self.assertEqual(resolve_device("cpu"), "cpu")

    def test_auto_returns_string(self):
        from emd_v5_2_hybrid.device_helper import resolve_device
        device = resolve_device("auto")
        self.assertIsInstance(device, str)
        self.assertTrue(len(device) > 0)

    def test_device_summary(self):
        from emd_v5_2_hybrid.device_helper import device_summary
        summary = device_summary("cpu")
        self.assertEqual(summary["device"], "cpu")
        self.assertEqual(summary["device_type"], "cpu")


if __name__ == "__main__":
    unittest.main()
