"""
drift.py
========
Brokerage Drift Visualizations — Two simple, clear plots:
  1. Temporal Betweenness over time (line chart per broker)
  2. Burt Constraint over time (line chart per broker)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from typing import Dict, Optional


# ── Style ──────────────────────────────────────────────────────────
_BG       = "#0f1117"
_PANEL_BG = "#181c25"
_GRID     = "#2a2e3a"
_TEXT     = "#e0e0e0"
_WARN     = "#ff6b6b"
_GOLD     = "#ffd700"

_COLORS = [
    "#6ec6ff", "#ff6b9d", "#c084fc", "#fbbf24",
    "#34d399", "#f87171", "#60a5fa", "#a78bfa",
    "#fb923c", "#2dd4bf", "#e879f9", "#38bdf8",
]


def _style(fig, ax):
    """Apply dark theme."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT, labelsize=10)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for s in ax.spines.values():
        s.set_color(_GRID)
    ax.grid(True, color=_GRID, alpha=0.35, linewidth=0.5)


def _save(fig, path):
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  → Saved: {path}")
    plt.close(fig)


# ── Plot 1: Temporal Betweenness ───────────────────────────────────
def plot_betweenness(diag, deps_dict, save_path=None, top_n=10):
    """
    Simple line chart: temporal betweenness for the top N brokers
    over weekly time bins.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    _style(fig, ax)

    avg = diag.groupby('node_id')['temporal_betweenness'].mean()
    top = avg.nlargest(top_n).index.tolist()

    for i, nid in enumerate(top):
        nd = diag[diag['node_id'] == nid].sort_values('bin_id')
        vals = nd['temporal_betweenness'].values
        bins = nd['bin_id'].values
        color = _COLORS[i % len(_COLORS)]

        # Smooth for readability
        smooth = uniform_filter1d(vals, size=5) if len(vals) > 5 else vals
        dept = deps_dict.get(str(nid), deps_dict.get(nid, 'unknown'))
        label = f"Node {nid} ({dept})"

        ax.plot(bins, smooth, color=color, linewidth=1.8, alpha=0.85, label=label)

        # Label at right end
        ax.annotate(str(nid), xy=(bins[-1], smooth[-1]),
                    fontsize=8, color=color, fontweight='bold',
                    xytext=(6, 0), textcoords='offset points', va='center')

    ax.set_xlabel('Time Bin (Week)', fontsize=12)
    ax.set_ylabel('Temporal Betweenness (normalized)', fontsize=12)
    ax.set_title('Temporal Betweenness Over Time — Top 10 Brokers',
                 fontsize=14, fontweight='bold', pad=14)

    ax.legend(fontsize=8, facecolor=_PANEL_BG, edgecolor=_GRID,
              labelcolor=_TEXT, loc='upper left', ncol=2)

    _save(fig, save_path)


# ── Plot 2: Burt Constraint ───────────────────────────────────────
def plot_constraint(diag, deps_dict, save_path=None, top_n=10):
    """
    Simple line chart: Burt constraint for the top N brokers
    over weekly time bins.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    _style(fig, ax)

    # Use the same top brokers (by betweenness) so both plots are comparable
    avg_betw = diag.groupby('node_id')['temporal_betweenness'].mean()
    top = avg_betw.nlargest(top_n).index.tolist()

    for i, nid in enumerate(top):
        nd = diag[diag['node_id'] == nid].sort_values('bin_id')
        vals = nd['constraint'].values
        bins = nd['bin_id'].values
        color = _COLORS[i % len(_COLORS)]

        smooth = uniform_filter1d(vals, size=5) if len(vals) > 5 else vals
        dept = deps_dict.get(str(nid), deps_dict.get(nid, 'unknown'))
        label = f"Node {nid} ({dept})"

        ax.plot(bins, smooth, color=color, linewidth=1.8, alpha=0.85, label=label)

        ax.annotate(str(nid), xy=(bins[-1], smooth[-1]),
                    fontsize=8, color=color, fontweight='bold',
                    xytext=(6, 0), textcoords='offset points', va='center')

    ax.set_xlabel('Time Bin (Week)', fontsize=12)
    ax.set_ylabel('Burt Constraint (0 = unconstrained, 1 = fully trapped)', fontsize=12)
    ax.set_title('Burt Constraint Over Time — Top 10 Brokers',
                 fontsize=14, fontweight='bold', pad=14)

    ax.legend(fontsize=8, facecolor=_PANEL_BG, edgecolor=_GRID,
              labelcolor=_TEXT, loc='upper left', ncol=2)

    _save(fig, save_path)


# ── Main Entrypoint ──────────────────────────────────────────────
def plot_brokerage_drift(diagnostics: pd.DataFrame,
                         targets: pd.DataFrame = None,
                         preds_a: pd.DataFrame = None,
                         preds_b: pd.DataFrame = None,
                         deps_dict: Dict = None,
                         output_dir: str = 'outputs',
                         show_plot: bool = True,
                         save_path: Optional[str] = None):
    """
    Generate 2 simple brokerage drift plots:
      1. temporal_betweenness.png — line chart of betweenness over time
      2. burt_constraint.png     — line chart of constraint over time

    Parameters
    ----------
    diagnostics : DataFrame  – sena_diagnostics.csv
    deps_dict   : dict       – {node_id_str: department}
    output_dir  : str        – directory to save plots
    """
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if deps_dict is None:
        deps_dict = {}

    print("Generating 2 drift plots...")

    plot_betweenness(
        diagnostics, deps_dict,
        save_path=str(out / "drift_temporal_betweenness.png")
    )
    plot_constraint(
        diagnostics, deps_dict,
        save_path=str(out / "drift_burt_constraint.png")
    )

    print("Drift plots done.")
