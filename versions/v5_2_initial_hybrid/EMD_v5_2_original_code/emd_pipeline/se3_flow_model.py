"""
EMD V5.2 Hybrid — SE(3) Flow Matching Model (M3)
=================================================
Custom SE(3)-equivariant flow matching model for 3D molecular generation.
Designed to run on Google Colab T4 GPU with CPU fallback.
"""

import os
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════
# Dataset and DataLoader
# ═══════════════════════════════════════════════════════════

class MolecularGraphDataset(Dataset):
    """Dataset for SE(3) molecular graphs with padding/masking."""
    
    def __init__(self, graphs, max_atoms=80, split_ids=None):
        self.max_atoms = max_atoms
        if split_ids is not None:
            split_set = set(split_ids)
            self.graphs = [g for g in graphs if g["mol_id"] in split_set]
        else:
            self.graphs = graphs
        print(f"  Dataset: {len(self.graphs)} molecules, max_atoms={max_atoms}")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        g = self.graphs[idx]
        N = g["num_atoms"]
        F_atom = g["atom_features"].shape[1]
        
        # Pad atom features
        atom_feats = torch.zeros(self.max_atoms, F_atom)
        atom_feats[:N] = g["atom_features"][:N]
        
        # Pad coords
        coords = torch.zeros(self.max_atoms, 3)
        coords[:N] = g["coords"][:N]
        
        # Center coords
        coords[:N] = coords[:N] - coords[:N].mean(dim=0, keepdim=True)
        
        # Mask
        mask = torch.zeros(self.max_atoms)
        mask[:N] = 1.0
        
        return {
            "atom_features": atom_feats,
            "coords": coords,
            "mask": mask,
            "num_atoms": N,
            "mol_id": g["mol_id"],
        }


def collate_fn(batch):
    """Custom collation for molecular graph batches."""
    return {
        "atom_features": torch.stack([b["atom_features"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "num_atoms": torch.tensor([b["num_atoms"] for b in batch]),
        "mol_id": [b["mol_id"] for b in batch],
    }


# ═══════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════

class SE3InvariantLayer(nn.Module):
    """SE(3)-aware message passing layer using relative distances."""
    
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Coordinate update (equivariant)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h, x, mask):
        """
        Args:
            h: [B, N, D] node features
            x: [B, N, 3] coordinates
            mask: [B, N] atom mask
        Returns:
            h_out: [B, N, D] updated features
            x_out: [B, N, 3] updated coordinates
        """
        B, N, D = h.shape
        
        # Compute pairwise distances
        # x_i - x_j: [B, N, N, 3]
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        dist = dist + 1e-8  # avoid div by zero
        
        # Pairwise mask
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, N, N]
        
        # Message: concatenate [h_i, h_j, dist]
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        
        msg_input = torch.cat([h_i, h_j, dist], dim=-1)  # [B, N, N, 2D+1]
        msg = self.node_mlp(msg_input)  # [B, N, N, D]
        
        # Mask messages
        msg = msg * pair_mask.unsqueeze(-1)
        
        # Aggregate messages (mean over neighbors)
        num_neighbors = pair_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        h_agg = msg.sum(dim=2) / num_neighbors  # [B, N, D]
        
        # Update features with residual
        h_out = self.norm(h + h_agg)
        
        # Coordinate update (equivariant)
        coord_input = torch.cat([msg.mean(dim=2), dist.squeeze(-1).mean(dim=-1, keepdim=True)], dim=-1)
        coord_weight = self.coord_mlp(coord_input)  # [B, N, 1]
        
        # Weighted displacement
        weighted_diff = diff * pair_mask.unsqueeze(-1)
        disp = weighted_diff.mean(dim=2)  # [B, N, 3]
        x_out = x + coord_weight * disp * mask.unsqueeze(-1)
        
        return h_out, x_out


class SE3FlowMatchingModel(nn.Module):
    """SE(3) Flow Matching model for 3D molecular generation.
    
    Architecture:
    - Atom feature embedding with Xavier initialization
    - Sinusoidal time embedding (standard for diffusion/flow models)
    - Stack of SE3InvariantLayers
    - Velocity prediction head with zero-init (output starts near zero)
    """
    
    def __init__(self, atom_feature_dim, hidden_dim=128, num_layers=4,
                 num_heads=4, max_atoms=80):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms
        
        # Input embedding
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Sinusoidal time embedding (more expressive than learned)
        self.time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # SE(3) layers
        self.layers = nn.ModuleList([
            SE3InvariantLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Velocity prediction head
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for linear layers, zero-init for velocity output."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Zero-initialize velocity head output (model starts predicting ~zero velocity)
        nn.init.zeros_(self.vel_head[-1].weight)
        nn.init.zeros_(self.vel_head[-1].bias)
    
    def _sinusoidal_embedding(self, t):
        """Sinusoidal positional embedding for time t ∈ [0, 1]."""
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1) if half_dim > 1 else 0.0
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, time_dim]
        return emb
    
    def forward(self, atom_features, coords_t, t, mask):
        """
        Args:
            atom_features: [B, N, F] atom features
            coords_t: [B, N, 3] noised coordinates at time t
            t: [B] time values in [0, 1]
            mask: [B, N] atom mask
        
        Returns:
            velocity: [B, N, 3] predicted velocity field
        """
        # Embed atoms
        h = self.atom_embed(atom_features)  # [B, N, D]
        
        # Sinusoidal time embedding → MLP
        t_sinusoidal = self._sinusoidal_embedding(t)  # [B, D]
        t_emb = self.time_mlp(t_sinusoidal)  # [B, D]
        h = h + t_emb.unsqueeze(1)  # broadcast time to all atoms
        
        # SE(3) message passing
        x = coords_t
        for layer in self.layers:
            h, x = layer(h, x, mask)
        
        # Predict velocity
        velocity = self.vel_head(h)  # [B, N, 3]
        velocity = velocity * mask.unsqueeze(-1)
        
        return velocity


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def flow_matching_loss(model, batch, device):
    """Compute conditional flow matching loss.
    
    Samples time t, creates noisy coordinates, and computes MSE
    between predicted and target velocity fields.
    """
    atom_features = batch["atom_features"].to(device)
    coords_1 = batch["coords"].to(device)  # target (clean) coords
    mask = batch["mask"].to(device)
    B = coords_1.shape[0]
    
    # Sample time — avoid exact 0 and 1 for numerical stability
    t = torch.rand(B, device=device).clamp(1e-4, 1.0 - 1e-4)
    
    # Sample noise (source distribution = centered Gaussian)
    coords_0 = torch.randn_like(coords_1) * mask.unsqueeze(-1)
    # Center noise per molecule
    for i in range(B):
        n = int(mask[i].sum().item())
        if n > 0:
            coords_0[i, :n] = coords_0[i, :n] - coords_0[i, :n].mean(dim=0)
    
    # Interpolate: x_t = (1-t) * x_0 + t * x_1
    t_expand = t.view(B, 1, 1)
    coords_t = (1 - t_expand) * coords_0 + t_expand * coords_1
    
    # Target velocity: dx/dt = x_1 - x_0
    target_velocity = coords_1 - coords_0
    
    # Predict velocity
    pred_velocity = model(atom_features, coords_t, t, mask)
    
    # MSE loss (masked) — per-atom mean for stability
    diff = (pred_velocity - target_velocity) ** 2
    diff = diff * mask.unsqueeze(-1)
    
    # Per-molecule loss, then batch mean (more stable than global sum)
    per_mol = diff.sum(dim=(1, 2)) / (mask.sum(dim=1).clamp(min=1) * 3.0)
    loss = per_mol.mean()
    
    # NaN guard
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss


def train_se3_model(
    model, train_dataset, val_dataset, config,
    checkpoint_dir, device=None, progress_path=None
):
    """Train the SE(3) flow matching model with checkpointing.
    
    Features:
    - Automatic GPU/CPU detection
    - Exponential Moving Average (EMA) for generation quality
    - Linear warmup + cosine annealing LR schedule
    - Checkpoint saving every N epochs
    - Resume from checkpoint
    - Training curve logging
    - Gradient clipping
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Training on: {device}")
    model = model.to(device)
    
    # Config
    epochs = config.get("epochs_target", 200)
    batch_size = config.get("batch_size", 4)
    lr = config.get("learning_rate", 2e-4)
    weight_decay = config.get("weight_decay", 1e-5)
    grad_clip = config.get("gradient_clip", 1.0)
    ckpt_every = config.get("checkpoint_every", 10)
    val_every = config.get("validate_every", 5)
    seed = config.get("random_seed", 42)
    ema_decay = config.get("ema_decay", 0.999)
    warmup_epochs = config.get("warmup_epochs", max(1, epochs // 20))
    
    torch.manual_seed(seed)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )
    
    # Optimizer with warmup + cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Linear warmup then cosine annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # EMA (Exponential Moving Average) for generation quality
    import copy
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    
    def update_ema(ema, source, decay):
        with torch.no_grad():
            for ema_p, src_p in zip(ema.parameters(), source.parameters()):
                ema_p.data.mul_(decay).add_(src_p.data, alpha=1.0 - decay)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    log_records = []
    
    latest_ckpt = os.path.join(checkpoint_dir, "se3_latest_checkpoint.pt")
    if os.path.exists(latest_ckpt):
        print("Resuming from latest checkpoint...")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    
    global_step = 0
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses = []
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = flow_matching_loss(model, batch, device)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Update EMA model
            update_ema(ema_model, model, ema_decay)
            global_step += 1
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        train_loss = np.mean(train_losses) if train_losses else float("nan")
        epoch_time = time.time() - epoch_start
        
        # Validation — run every epoch for reliable monitoring
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                vl = flow_matching_loss(model, batch, device)
                if not (torch.isnan(vl) or torch.isinf(vl)):
                    val_losses.append(vl.item())
        val_loss = np.mean(val_losses) if val_losses else float("nan")
        
        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": epoch_time,
        }
        log_records.append(log_entry)
        
        # Print with val_loss display handling
        val_display = f"{val_loss:.6f}" if not np.isnan(val_loss) else "N/A"
        print(f"  Epoch {epoch}/{epochs}: train={train_loss:.6f} val={val_display} time={epoch_time:.1f}s")
        
        # Build checkpoint dict
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if not np.isnan(val_loss) else None,
            "config": config,
            "random_seed": seed,
        }
        
        # Always save latest
        torch.save(ckpt, latest_ckpt)
        
        # Save periodic numbered checkpoint
        if epoch % ckpt_every == 0 or epoch == epochs - 1:
            torch.save(ckpt, os.path.join(checkpoint_dir, f"se3_epoch_{epoch:04d}.pt"))
        
        # Best model tracking — use train_loss as fallback if val_loss unavailable
        check_loss = val_loss if not np.isnan(val_loss) else train_loss
        if not np.isnan(check_loss) and check_loss < best_val_loss:
            best_val_loss = check_loss
            torch.save(ckpt, os.path.join(checkpoint_dir, "se3_best_checkpoint.pt"))
            print(f"    New best model! loss={check_loss:.6f}")
        
        # Save progress
        if progress_path:
            from emd_pipeline.config_registry import save_progress
            save_progress(progress_path, {
                "last_completed_index": epoch,
                "status": "training",
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            })
    
    # Save training log
    log_df = pd.DataFrame(log_records)
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"Training log saved to {log_path}")
    
    return model, log_df


def plot_training_curve(log_df, output_path):
    """Plot training and validation loss curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss", linewidth=2)
    
    val_mask = log_df["val_loss"].notna()
    if val_mask.any():
        ax.plot(log_df.loc[val_mask, "epoch"], log_df.loc[val_mask, "val_loss"],
                label="Val Loss", linewidth=2, marker="o", markersize=4)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("SE(3) Flow Matching Training Curve", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to {output_path}")


# ═══════════════════════════════════════════════════════════
# Inference / Generation
# ═══════════════════════════════════════════════════════════

def generate_se3_candidates(model, seed_graphs, num_steps=50, device=None,
                             temperature=1.0):
    """Generate new molecular coordinates using trained flow matching model.
    
    Process:
    1. Start from noise (t=0)
    2. Integrate velocity field from t=0 to t=1
    3. Output final coordinates at t=1
    4. Attempt SMILES reconstruction via distance geometry
    
    Args:
        model: Trained SE3FlowMatchingModel
        seed_graphs: List of graph dicts to use as atom-feature templates
        num_steps: Number of ODE integration steps
        device: torch device
        temperature: Noise scaling (lower = more conservative)
    
    Returns:
        List of dicts with generated coordinates and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    generated = []
    dt = 1.0 / num_steps
    max_atoms = model.max_atoms
    
    for g_idx, g in enumerate(seed_graphs):
        try:
            N = g["num_atoms"]
            F_dim = g["atom_features"].shape[1] if g["atom_features"].dim() == 2 else g["atom_features"].shape[2]
            
            # Properly pad atom features to max_atoms (CRITICAL FIX)
            padded_feats = torch.zeros(1, max_atoms, F_dim, device=device)
            raw_feats = g["atom_features"]
            if raw_feats.dim() == 2:
                padded_feats[0, :N] = raw_feats[:N]
            else:
                padded_feats[0, :N] = raw_feats[0, :N]
            
            # Mask
            mask = torch.zeros(1, max_atoms, device=device)
            mask[0, :N] = 1.0
            
            # Start from scaled noise
            x = torch.randn(1, max_atoms, 3, device=device) * temperature
            x = x * mask.unsqueeze(-1)
            # Center the noise
            if N > 0:
                x[0, :N] = x[0, :N] - x[0, :N].mean(dim=0)
            
            # Euler ODE integration t: 0 → 1
            with torch.no_grad():
                for step in range(num_steps):
                    t_val = step * dt
                    t = torch.tensor([t_val], device=device)
                    v = model(padded_feats, x, t, mask)
                    
                    # NaN guard during integration
                    if torch.isnan(v).any():
                        print(f"  Warning: NaN velocity at step {step} for graph {g_idx}, skipping")
                        break
                    
                    x = x + v * dt
                    # Re-center after each step (keeps coordinates stable)
                    if N > 0:
                        x[0, :N] = x[0, :N] - x[0, :N].mean(dim=0)
            
            final_coords = x[0, :N].cpu().numpy()
            
            # Check for NaN/Inf in output
            if np.any(np.isnan(final_coords)) or np.any(np.isinf(final_coords)):
                print(f"  Skipping graph {g_idx}: NaN/Inf in output coordinates")
                continue
            
            generated.append({
                "coords": final_coords,
                "atom_features": g["atom_features"][:N].numpy() if g["atom_features"].dim() == 2 else g["atom_features"][0, :N].numpy(),
                "mol_id": g.get("mol_id", f"SE3_{g_idx:05d}"),
                "source_smiles": g.get("smiles", ""),
                "num_atoms": N,
                "source_generator": "SE3",
            })
        except Exception as e:
            print(f"  Generation failed for graph {g_idx}: {e}")
            continue
    
    print(f"SE3: Generated {len(generated)} candidates from {len(seed_graphs)} seeds")
    return generated


def coords_to_smiles(generated_candidates, seed_graphs):
    """Attempt to reconstruct SMILES from generated 3D coordinates.
    
    Strategy: Use the seed molecule's connectivity (bonds, atom types)
    but with new 3D coordinates from the SE(3) model. This produces
    perturbed conformations that may reveal new binding modes.
    
    For truly novel molecules, we'd need learned bond prediction,
    which is a future enhancement.
    
    Args:
        generated_candidates: Output from generate_se3_candidates
        seed_graphs: Original seed graphs (for bond connectivity)
    
    Returns:
        List of dicts with SMILES and metadata
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
    except ImportError:
        print("RDKit not available for SMILES reconstruction")
        return []
    
    results = []
    seed_map = {g.get("mol_id", ""): g for g in seed_graphs}
    
    for gen in generated_candidates:
        try:
            source_id = gen.get("mol_id", "")
            source_smi = gen.get("source_smiles", "")
            
            if not source_smi:
                # Try to find in seed map
                seed_g = seed_map.get(source_id)
                if seed_g:
                    source_smi = seed_g.get("smiles", "")
            
            if not source_smi:
                continue
            
            # Parse the source molecule
            mol = Chem.MolFromSmiles(source_smi)
            if mol is None:
                continue
            
            # Add hydrogens and embed with generated coordinates
            mol = Chem.AddHs(mol)
            
            # Create a conformer from generated coords
            conf = Chem.Conformer(mol.GetNumAtoms())
            N_gen = gen["num_atoms"]
            N_mol = mol.GetNumAtoms()
            
            # Map generated coords to heavy atoms
            heavy_idx = 0
            for i in range(N_mol):
                if mol.GetAtomWithIdx(i).GetAtomicNum() != 1:  # Not hydrogen
                    if heavy_idx < N_gen:
                        coord = gen["coords"][heavy_idx]
                        conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
                        heavy_idx += 1
                    else:
                        conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
                else:
                    conf.SetAtomPosition(i, (0.0, 0.0, 0.0))
            
            mol.AddConformer(conf, assignId=True)
            
            # Optimize hydrogen positions
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=100)
            except:
                pass
            
            # Get canonical SMILES (connectivity unchanged from source)
            mol_no_h = Chem.RemoveHs(mol)
            can_smi = Chem.MolToSmiles(mol_no_h)
            
            results.append({
                "candidate_id": f"SE3_{len(results):05d}",
                "source_generator": "SE3",
                "parent_mol_id": source_id,
                "canonical_smiles": can_smi,
                "valid_rdkit": True,
                "generation_notes": f"SE3 flow coords from {source_id}",
            })
            
        except Exception as e:
            continue
    
    print(f"SE3 coord-to-SMILES: {len(results)} reconstructed")
    return results
