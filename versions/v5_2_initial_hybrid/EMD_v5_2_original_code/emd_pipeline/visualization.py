"""
EMD V5.2 Hybrid — Visualization Module
=======================================
Molecular structure visualization, distribution plots,
training curves, and Colab-friendly 3D rendering.
"""

import os
import numpy as np
import pandas as pd


def plot_dataset_overview(curated_df, features_df=None, output_path=None):
    """Generate comprehensive dataset overview figure.
    
    Creates a 2x3 panel with MW, LogP, pActivity, QED, ring size,
    and train/val/test split distributions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    palette = ["#e94560", "#0f3460", "#16213e", "#533483", "#e94560", "#0f3460"]
    
    df = features_df if features_df is not None else curated_df

    panels = [
        ("mw", "Molecular Weight (Da)", palette[0]),
        ("logp", "LogP", palette[1]),
        ("qed", "QED Drug-likeness", palette[3]),
        ("rotatable_bonds", "Rotatable Bonds", palette[4]),
        ("max_ring_size", "Max Ring Size", palette[5]),
    ]

    for idx, (col, title, color) in enumerate(panels):
        if col not in df.columns:
            continue
        row, col_idx = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col_idx])
        ax.set_facecolor("#16213e")
        
        data = df[col].dropna()
        ax.hist(data, bins=25, color=color, alpha=0.85, edgecolor="#1a1a2e", linewidth=0.5)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(col, color="#aaa", fontsize=9)
        ax.set_ylabel("Count", color="#aaa", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        
        # Stats annotation
        ax.text(0.97, 0.95, f"n={len(data)}\nmed={data.median():.1f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#ccc", fontstyle="italic")

    # Split distribution (last panel)
    if "split" in curated_df.columns:
        ax = fig.add_subplot(gs[1, 2])
        ax.set_facecolor("#16213e")
        counts = curated_df["split"].value_counts()
        colors_pie = ["#e94560", "#533483", "#0f3460"]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.0f%%",
            colors=colors_pie[:len(counts)], textprops={"color": "white", "fontsize": 10}
        )
        ax.set_title("Train / Val / Test Split", color="white", fontsize=11, fontweight="bold")

    fig.suptitle("ElectroMacroDiff V5.2 Hybrid - Dataset Overview",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Dataset overview saved: {output_path}")
    plt.close()


def plot_training_dashboard(log_df, output_path=None):
    """Generate a training dashboard with loss curves, LR schedule, and epoch timing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Panel 1: Loss curves
    ax = axes[0]
    ax.plot(log_df["epoch"], log_df["train_loss"], color="#e94560", linewidth=2, label="Train")
    val_mask = log_df["val_loss"].notna()
    if val_mask.any():
        ax.plot(log_df.loc[val_mask, "epoch"], log_df.loc[val_mask, "val_loss"],
                color="#533483", linewidth=2, marker="o", markersize=4, label="Validation")
    ax.set_xlabel("Epoch", color="#aaa")
    ax.set_ylabel("Loss", color="#aaa")
    ax.set_title("Training & Validation Loss", color="white", fontweight="bold")
    ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white")

    # Panel 2: Learning rate
    ax = axes[1]
    if "lr" in log_df.columns:
        ax.plot(log_df["epoch"], log_df["lr"], color="#0f3460", linewidth=2)
        ax.set_xlabel("Epoch", color="#aaa")
        ax.set_ylabel("Learning Rate", color="#aaa")
        ax.set_title("Learning Rate Schedule", color="white", fontweight="bold")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))

    # Panel 3: Epoch time
    ax = axes[2]
    if "epoch_time_s" in log_df.columns:
        ax.bar(log_df["epoch"], log_df["epoch_time_s"], color="#e94560", alpha=0.7, width=0.8)
        ax.set_xlabel("Epoch", color="#aaa")
        ax.set_ylabel("Time (s)", color="#aaa")
        ax.set_title("Epoch Duration", color="white", fontweight="bold")

    fig.suptitle("SE(3) Flow Matching - Training Dashboard",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Training dashboard saved: {output_path}")
    plt.close()


def plot_generation_comparison(metrics_df, merged_df=None, output_path=None):
    """Plot generation comparison across SE(3), SELFIES, and RDKit sources."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#888", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333")

    src_colors = {"SE3": "#e94560", "SELFIES": "#533483", "RDKit": "#0f3460",
                  "se3": "#e94560", "selfies": "#533483", "rdkit": "#0f3460"}

    if merged_df is not None and "source_generator" in merged_df.columns:
        sources = merged_df["source_generator"].unique()

        # Panel 1: Count by source
        ax = axes[0]
        counts = merged_df["source_generator"].value_counts()
        colors = [src_colors.get(s, "#888") for s in counts.index]
        ax.bar(counts.index, counts.values, color=colors, alpha=0.85, edgecolor="#1a1a2e")
        ax.set_title("Candidates by Source", color="white", fontweight="bold")
        ax.set_ylabel("Count", color="#aaa")

        # Panel 2: QED by source
        ax = axes[1]
        if "qed" in merged_df.columns:
            for src in sources:
                data = merged_df[merged_df["source_generator"] == src]["qed"].dropna()
                ax.hist(data, bins=15, alpha=0.6, label=src,
                        color=src_colors.get(src, "#888"), edgecolor="#1a1a2e")
            ax.set_title("QED Distribution by Source", color="white", fontweight="bold")
            ax.set_xlabel("QED", color="#aaa")
            ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white")

        # Panel 3: SA Score by source
        ax = axes[2]
        if "sa_score" in merged_df.columns:
            for src in sources:
                data = merged_df[merged_df["source_generator"] == src]["sa_score"].dropna()
                ax.hist(data, bins=15, alpha=0.6, label=src,
                        color=src_colors.get(src, "#888"), edgecolor="#1a1a2e")
            ax.set_title("SA Score by Source", color="white", fontweight="bold")
            ax.set_xlabel("SA Score", color="#aaa")
            ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white")

    fig.suptitle("Hybrid Generation Comparison",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Generation comparison saved: {output_path}")
    plt.close()


def plot_ranking_dashboard(ranked_df, output_path=None):
    """Generate final ranking dashboard with score breakdown and top candidates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")

    # Panel 1: Top 10 final scores
    ax = axes[0, 0]
    top10 = ranked_df.head(10)
    colors = ["#e94560" if d == "CANDIDATE" else "#533483" if d == "BACKUP" else "#444"
              for d in top10.get("decision", ["ARCHIVE"] * len(top10))]
    y_pos = range(len(top10))
    ax.barh(y_pos, top10["final_weighted_score"], color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top10["candidate_id"], fontsize=8, color="#ccc")
    ax.set_xlabel("Final Weighted Score", color="#aaa")
    ax.set_title("Top 10 Candidates", color="white", fontweight="bold")
    ax.invert_yaxis()

    # Panel 2: Docking vs ADMET scatter
    ax = axes[0, 1]
    if "best_score" in ranked_df.columns and "admet_score" in ranked_df.columns:
        valid = ranked_df.dropna(subset=["best_score", "admet_score"])
        scatter_colors = ["#e94560" if r <= 5 else "#533483" if r <= 10 else "#444"
                          for r in valid["rank"]]
        ax.scatter(valid["best_score"], valid["admet_score"], c=scatter_colors, alpha=0.7, s=50, edgecolors="#1a1a2e")
        ax.set_xlabel("Docking Score (kcal/mol)", color="#aaa")
        ax.set_ylabel("ADMET Score", color="#aaa")
        ax.set_title("Docking vs ADMET", color="white", fontweight="bold")

    # Panel 3: Score component radar for top 3
    ax = axes[1, 0]
    components = ["docking_score_norm", "admet_score", "synthesis_score", "safety_proxy_score"]
    comp_labels = ["Docking", "ADMET", "Synthesis", "Safety"]
    available = [c for c, l in zip(components, comp_labels) if c in ranked_df.columns]
    labels = [l for c, l in zip(components, comp_labels) if c in ranked_df.columns]

    if available and len(ranked_df) >= 3:
        x = np.arange(len(available))
        width = 0.25
        for i, (_, row) in enumerate(ranked_df.head(3).iterrows()):
            vals = [row.get(c, 0) for c in available]
            color = ["#e94560", "#533483", "#0f3460"][i]
            ax.bar(x + i * width, vals, width, label=f"#{row.get('rank', i+1)}",
                   color=color, alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels, fontsize=9, color="#ccc")
        ax.set_ylabel("Score", color="#aaa")
        ax.set_title("Top 3 Score Breakdown", color="white", fontweight="bold")
        ax.legend(facecolor="#16213e", edgecolor="#333", labelcolor="white", fontsize=9)

    # Panel 4: Source distribution pie
    ax = axes[1, 1]
    if "source_generator" in ranked_df.columns:
        top20 = ranked_df.head(20)
        src_counts = top20["source_generator"].value_counts()
        src_colors_pie = ["#e94560", "#533483", "#0f3460", "#888"]
        ax.pie(src_counts.values, labels=src_counts.index, autopct="%1.0f%%",
               colors=src_colors_pie[:len(src_counts)],
               textprops={"color": "white", "fontsize": 10})
        ax.set_title("Source Distribution (Top 20)", color="white", fontweight="bold")

    fig.suptitle("ElectroMacroDiff V5.2 Hybrid - Final Ranking Dashboard",
                 color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Ranking dashboard saved: {output_path}")
    plt.close()


def draw_molecule_grid(smiles_list, legends=None, output_path=None,
                        mols_per_row=4, img_size=(350, 300)):
    """Draw a grid of 2D molecule depictions from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    mols = []
    valid_legends = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            mols.append(mol)
            if legends:
                valid_legends.append(legends[i] if i < len(legends) else "")
            else:
                valid_legends.append("")

    if not mols:
        print("No valid molecules to draw")
        return None

    grid = Draw.MolsToGridImage(
        mols, molsPerRow=mols_per_row,
        subImgSize=img_size, legends=valid_legends if legends else None
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid.save(output_path)
        print(f"Molecule grid saved: {output_path}")

    return grid


def render_3d_molecule(smiles, viewer_size=(400, 300)):
    """Render a 3D molecule view using py3Dmol (for Colab/Jupyter).
    
    Returns the py3Dmol view object for display in notebooks.
    """
    try:
        import py3Dmol
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if mol.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        mb = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=viewer_size[0], height=viewer_size[1])
        viewer.addModel(mb, "mol")
        viewer.setStyle({"stick": {"colorscheme": "cyanCarbon"}})
        viewer.setBackgroundColor("#1a1a2e")
        viewer.zoomTo()
        return viewer

    except ImportError:
        print("py3Dmol not available - install with: pip install py3Dmol")
        return None
