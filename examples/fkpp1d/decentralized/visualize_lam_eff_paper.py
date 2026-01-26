"""
Paper-Quality Visualization for Lambda-Effort Analysis in FKPP1D Control.
Reads pre-computed CSV data and generates publication-ready plots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

# --- Path Setup ---
FIGURES_DIR = Path(__file__).parent / "figures" / "conjecture"
CSV_PATH = FIGURES_DIR / "conjecture_data_windowed.csv"

# --- Paper Style ---
def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

# --- Plotting ---
def plot_lambda_effort_paper(df):
    """Generate paper-quality plots for lambda-effort analysis."""
    setup_paper_style()
    
    colors = ['#2c3e50', '#2980b9', '#27ae60', '#e67e22']
    markers = ['o', 's', '^', 'D']
    
    def format_effort_axis(ax):
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(framealpha=0.9)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    # --- Plot 1: Tracking MSE ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    for i, l in enumerate(df['lambda'].unique()):
        sub = df[df['lambda'] == l]
        ax1.semilogy(sub['n_agents'], sub['mse'], marker=markers[i % len(markers)], 
                     markersize=6, label=f'$\\lambda_u={l}$', 
                     color=colors[i % len(colors)], linewidth=2)
    
    ax1.set_xlabel("Number of Agents ($N$)")
    ax1.set_ylabel("Final $L^2$ Error")
    ax1.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Training $N$')
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.legend(framealpha=0.9)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "paper_scaling_mse.pdf")
    fig1.savefig(FIGURES_DIR / "paper_scaling_mse.png", dpi=300)
    print(f"✓ Saved: paper_scaling_mse.pdf/png")

    # --- Plot 2: Squared Effort ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for i, l in enumerate(df['lambda'].unique()):
        sub = df[df['lambda'] == l]
        ax2.loglog(sub['n_agents'], sub['total_effort_sq'], marker=markers[i % len(markers)],
                   markersize=6, label=f'$\\lambda_u={l}$', 
                   color=colors[i % len(colors)], linewidth=2)
    
    ax2.set_xlabel("Number of Agents ($N$)")
    ax2.set_ylabel("Mean $\\sum u_i^2$")
    format_effort_axis(ax2)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "paper_scaling_effort_sq.pdf")
    fig2.savefig(FIGURES_DIR / "paper_scaling_effort_sq.png", dpi=300)
    print(f"✓ Saved: paper_scaling_effort_sq.pdf/png")

    # --- Plot 3: Absolute Effort ---
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    for i, l in enumerate(df['lambda'].unique()):
        sub = df[df['lambda'] == l]
        ax3.loglog(sub['n_agents'], sub['total_effort_abs'], marker=markers[i % len(markers)],
                   markersize=6, label=f'$\\lambda_u={l}$', 
                   color=colors[i % len(colors)], linewidth=2)
    
    ax3.set_xlabel("Number of Agents ($N$)")
    ax3.set_ylabel("Mean $\\sum |u_i|$")
    format_effort_axis(ax3)
    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / "paper_scaling_effort_abs.pdf")
    fig3.savefig(FIGURES_DIR / "paper_scaling_effort_abs.png", dpi=300)
    print(f"✓ Saved: paper_scaling_effort_abs.pdf/png")
    
    plt.close('all')


def main():
    print("=" * 60)
    print("FKPP1D Lambda-Effort Paper Visualization")
    print("=" * 60)
    
    # Load CSV data
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Lambda values: {sorted(df['lambda'].unique())}")
    print(f"Agent counts: {sorted(df['n_agents'].unique())}")
    
    # Generate plots
    plot_lambda_effort_paper(df)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
