"""
drift.py
========
Brokerage Drift Visualizations
Six individual publication-quality plots showing how broker influence
erodes over time, who drifts, when they drift, and how well the model
predicts it.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, Optional


# ── Style constants ────────────────────────────────────────────────
_BG       = "#0f1117"
_PANEL_BG = "#181c25"
_GRID     = "#2a2e3a"
_TEXT     = "#e0e0e0"
_ACCENT   = "#6ec6ff"
_WARN     = "#ff6b6b"
_OK       = "#6bffb8"
_GOLD     = "#ffd700"

_DEPT_PALETTE = [
    "#6ec6ff", "#ff6b9d", "#c084fc", "#fbbf24",
    "#34d399", "#f87171", "#60a5fa", "#a78bfa",
    "#fb923c", "#2dd4bf", "#e879f9", "#38bdf8",
]


def _apply_dark_style(fig, ax):
    """Apply consistent dark theme to a single figure+axis."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(True, color=_GRID, alpha=0.4, linewidth=0.5)


def _dept_color_map(departments):
    """Map unique departments to palette colors."""
    unique = sorted(set(d for d in departments if d != 'unknown'))
    return {d: _DEPT_PALETTE[i % len(_DEPT_PALETTE)] for i, d in enumerate(unique)}


def _save_fig(fig, save_path):
    """Save and close a figure."""
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"  → Saved: {save_path}")
    plt.close(fig)


# ── Plot 1: Broker Trajectory Spaghetti ────────────────────────────
def plot_broker_trajectories(diag, dept_map, deps_dict,
                             save_path=None, top_n=12):
    """Spaghetti plot: temporal betweenness trajectories of top brokers,
    highlighting drift events with red markers.
    Includes a broker roster table on the right side."""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor(_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.02)

    ax = fig.add_subplot(gs[0])
    _apply_dark_style(fig, ax)

    ax_table = fig.add_subplot(gs[1])
    ax_table.set_facecolor(_BG)
    ax_table.axis('off')

    node_avg = diag.groupby('node_id')['temporal_betweenness'].mean()
    top_nodes = node_avg.nlargest(top_n).index.tolist()

    # Collect roster data while plotting
    roster = []

    for rank, nid in enumerate(top_nodes, 1):
        nd = diag[diag['node_id'] == nid].sort_values('bin_id')
        dept = deps_dict.get(str(nid), deps_dict.get(nid, 'unknown'))
        color = dept_map.get(dept, '#888888')
        vals = nd['temporal_betweenness'].values
        bins = nd['bin_id'].values

        smooth = uniform_filter1d(vals, size=5) if len(vals) > 5 else vals
        ax.plot(bins, smooth, color=color, alpha=0.7, linewidth=1.4)

        # Label at the endpoint of each line
        ax.annotate(f'{nid}', xy=(bins[-1], smooth[-1]),
                    fontsize=7, color=color, fontweight='bold',
                    xytext=(5, 0), textcoords='offset points',
                    va='center', ha='left')

        # Mark drift events (drops > 0.1)
        deltas = np.diff(vals)
        drift_count = int(np.sum(deltas < -0.1))
        drift_idx = np.where(deltas < -0.1)[0] + 1
        if len(drift_idx) > 0:
            ax.scatter(bins[drift_idx], smooth[drift_idx],
                       color=_WARN, s=28, zorder=5, alpha=0.85,
                       edgecolors='white', linewidths=0.4)

        roster.append({
            'rank': rank,
            'node_id': str(nid),
            'dept': dept,
            'mean_betw': node_avg[nid],
            'drift_events': drift_count,
            'color': color,
        })

    ax.set_xlabel('Time Bin (Week)', fontsize=11)
    ax.set_ylabel('Temporal Betweenness', fontsize=11)
    ax.set_title('Broker Influence Trajectories — Top 12 Brokers',
                 fontsize=13, fontweight='bold', pad=12)

    legend_els = [
        Line2D([0], [0], color='#aaaaaa', lw=1.4, label='Broker trajectory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_WARN,
               markersize=6, label='Drift event (Δ > 0.1)', linestyle='None'),
    ]
    ax.legend(handles=legend_els, loc='upper left', fontsize=9,
              facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT)

    # ── Broker Roster Table ────────────────────────────────────────
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(0, 1)

    # Title
    ax_table.text(0.5, 0.97, 'Top 12 Brokers', fontsize=12,
                  fontweight='bold', color=_TEXT, ha='center', va='top',
                  transform=ax_table.transAxes)

    # Column headers
    headers = ['#', 'Node', 'Dept', 'Avg Betw', 'Drifts']
    col_x = [0.02, 0.12, 0.30, 0.58, 0.84]
    header_y = 0.91

    for hdr, x in zip(headers, col_x):
        ax_table.text(x, header_y, hdr, fontsize=8.5, fontweight='bold',
                      color='#9ca3af', ha='left', va='top',
                      transform=ax_table.transAxes)

    # Separator line
    sep_y = header_y - 0.025
    ax_table.plot([0.01, 0.99], [sep_y, sep_y], color=_GRID, linewidth=0.8,
                  transform=ax_table.transAxes, clip_on=False)

    # Rows
    row_height = 0.058
    for i, r in enumerate(roster):
        y = header_y - 0.05 - i * row_height

        # Color indicator dot
        ax_table.plot(0.05, y - 0.005, 'o', color=r['color'], markersize=5,
                      transform=ax_table.transAxes)

        # Rank
        ax_table.text(col_x[0], y, f"{r['rank']}", fontsize=8,
                      color=_TEXT, ha='left', va='top',
                      transform=ax_table.transAxes)
        # Node ID
        ax_table.text(col_x[1], y, r['node_id'], fontsize=8,
                      color=r['color'], fontweight='bold', ha='left', va='top',
                      transform=ax_table.transAxes)
        # Department
        ax_table.text(col_x[2], y, r['dept'], fontsize=7.5,
                      color='#9ca3af', ha='left', va='top',
                      transform=ax_table.transAxes)
        # Mean betweenness
        ax_table.text(col_x[3], y, f"{r['mean_betw']:.3f}", fontsize=8,
                      color=_TEXT, ha='left', va='top',
                      transform=ax_table.transAxes)
        # Drift count
        drift_color = _WARN if r['drift_events'] > 10 else _TEXT
        ax_table.text(col_x[4], y, f"{r['drift_events']}", fontsize=8,
                      color=drift_color, fontweight='bold', ha='left', va='top',
                      transform=ax_table.transAxes)

    _save_fig(fig, save_path)


# ── Plot 2: Drift Event Waterfall ─────────────────────────────────
def plot_drift_waterfall(targets, deps_dict, dept_map, save_path=None):
    """Strip plot: each drift event as a dot on a node×time grid,
    colored by department."""
    fig, ax = plt.subplots(figsize=(12, 7))
    _apply_dark_style(fig, ax)

    drift_events = targets[targets['target_a'] == 1].copy()
    drift_events['dept'] = drift_events['node_id'].astype(str).map(
        lambda x: deps_dict.get(x, deps_dict.get(int(x) if x.isdigit() else x, 'unknown'))
    )

    drifters = drift_events.groupby('node_id')['bin_id'].min().sort_values()
    node_y = {nid: i for i, nid in enumerate(drifters.index)}

    for _, row in drift_events.iterrows():
        y = node_y.get(row['node_id'], 0)
        dept = row['dept']
        color = dept_map.get(dept, '#888888')
        ax.scatter(row['bin_id'], y, color=color, s=8, alpha=0.7, edgecolors='none')

    ax.set_xlabel('Time Bin (Week)', fontsize=11)
    ax.set_ylabel('Drifting Nodes (ordered by first drift)', fontsize=11)
    ax.set_title('Drift Event Timeline — When Each Node Loses Influence',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_yticks([])

    # Department legend
    dept_handles = [Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=dept_map[d], markersize=6,
                           label=d, linestyle='None')
                    for d in sorted(dept_map.keys())]
    if dept_handles:
        ax.legend(handles=dept_handles, loc='upper left', fontsize=7,
                  facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT,
                  ncol=2)

    _save_fig(fig, save_path)


# ── Plot 3: Population Drift Rate ─────────────────────────────────
def plot_population_drift_rate(targets, save_path=None):
    """Time-series of the fraction of nodes experiencing drift per window."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_style(fig, ax)

    rate = targets.groupby('bin_id').agg(
        drift_a_rate=('target_a', 'mean'),
        drift_b_rate=('target_b', 'mean'),
    ).reset_index()

    if len(rate) > 5:
        rate['drift_a_smooth'] = uniform_filter1d(rate['drift_a_rate'].values, size=5)
        rate['drift_b_smooth'] = uniform_filter1d(rate['drift_b_rate'].values, size=5)
    else:
        rate['drift_a_smooth'] = rate['drift_a_rate']
        rate['drift_b_smooth'] = rate['drift_b_rate']

    ax.fill_between(rate['bin_id'], rate['drift_a_smooth'],
                    alpha=0.2, color=_ACCENT)
    ax.plot(rate['bin_id'], rate['drift_a_smooth'],
            color=_ACCENT, linewidth=2, label='Target A: Betweenness Drop')

    ax.fill_between(rate['bin_id'], rate['drift_b_smooth'],
                    alpha=0.2, color=_GOLD)
    ax.plot(rate['bin_id'], rate['drift_b_smooth'],
            color=_GOLD, linewidth=2, label='Target B: Constraint Rise')

    ax.set_xlabel('Time Bin (Week)', fontsize=11)
    ax.set_ylabel('Drift Fraction', fontsize=11)
    ax.set_title('Population-Level Drift Rate Over Time',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, min(1.0, rate[['drift_a_smooth', 'drift_b_smooth']].max().max() * 1.5))
    ax.legend(fontsize=10, facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT)

    _save_fig(fig, save_path)


# ── Plot 4: Phase Portrait ────────────────────────────────────────
def plot_phase_portrait(targets, deps_dict, dept_map, save_path=None):
    """Constraint vs Betweenness scatter colored by drift outcome.
    Arrows show the direction of change (current → next window)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    _apply_dark_style(fig, ax)

    sample = targets.dropna(subset=['temporal_betweenness', 'constraint',
                                     'next_betweenness', 'next_constraint'])
    if len(sample) > 1000:
        sample = sample.sample(1000, random_state=42)

    non_drift = sample[(sample['target_a'] == 0) & (sample['target_b'] == 0)]
    ax.scatter(non_drift['temporal_betweenness'], non_drift['constraint'],
               c='#555555', s=12, alpha=0.25, edgecolors='none', label='Stable')

    drift_a = sample[sample['target_a'] == 1]
    ax.scatter(drift_a['temporal_betweenness'], drift_a['constraint'],
               c=_WARN, s=24, alpha=0.6, edgecolors='white', linewidths=0.3,
               label='Drift A (Betw. Drop)', marker='v')

    drift_b = sample[sample['target_b'] == 1]
    ax.scatter(drift_b['temporal_betweenness'], drift_b['constraint'],
               c=_GOLD, s=24, alpha=0.6, edgecolors='white', linewidths=0.3,
               label='Drift B (Constr. Rise)', marker='^')

    # Draw arrows for dramatic drift-A cases
    big_drifters = drift_a.nlargest(30, 'temporal_betweenness')
    for _, row in big_drifters.iterrows():
        dx = row['next_betweenness'] - row['temporal_betweenness']
        dy = row['next_constraint'] - row['constraint']
        ax.annotate('', xy=(row['temporal_betweenness'] + dx,
                            row['constraint'] + dy),
                    xytext=(row['temporal_betweenness'], row['constraint']),
                    arrowprops=dict(arrowstyle='->', color=_WARN, lw=0.7, alpha=0.5))

    ax.set_xlabel('Temporal Betweenness', fontsize=11)
    ax.set_ylabel('Burt Constraint', fontsize=11)
    ax.set_title('Phase Portrait — Betweenness vs Constraint with Drift Vectors',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9, facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT,
              loc='upper right')

    _save_fig(fig, save_path)


# ── Plot 5: Drift Risk Heatmap ────────────────────────────────────
def plot_risk_heatmap(targets, preds_a, save_path=None):
    """Heatmap of predicted drift risk over time for the most-at-risk nodes."""
    fig, ax = plt.subplots(figsize=(14, 7))
    _apply_dark_style(fig, ax)

    n_preds = len(preds_a)
    tail = targets.tail(n_preds).copy()
    tail['pred_risk'] = preds_a['y_pred'].values

    pivot = tail.pivot_table(index='node_id', columns='bin_id',
                             values='pred_risk', aggfunc='mean')

    top_risky = pivot.mean(axis=1).nlargest(25).index
    pivot_top = pivot.loc[top_risky].fillna(0)
    pivot_top = pivot_top[sorted(pivot_top.columns)]
    pivot_top = pivot_top.loc[pivot_top.mean(axis=1).sort_values(ascending=False).index]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'drift_risk', ['#181c25', '#1a3a5c', '#ff6b6b', '#ff2222'], N=256)

    im = ax.imshow(pivot_top.values, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   interpolation='nearest')

    ax.set_xlabel('Time Bin (Week)', fontsize=11)
    ax.set_ylabel('Node ID', fontsize=11)
    ax.set_title('Predicted Drift Risk Heatmap — Top 25 At-Risk Nodes',
                 fontsize=13, fontweight='bold', pad=12)

    n_cols = len(pivot_top.columns)
    tick_step = max(1, n_cols // 12)
    ax.set_xticks(range(0, n_cols, tick_step))
    ax.set_xticklabels([str(pivot_top.columns[i]) for i in range(0, n_cols, tick_step)],
                       fontsize=8)
    ax.set_yticks(range(len(pivot_top)))
    ax.set_yticklabels([str(n) for n in pivot_top.index], fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Predicted Drift Probability', fontsize=10, color=_TEXT)
    cbar.ax.tick_params(labelsize=8, colors=_TEXT)

    _save_fig(fig, save_path)


# ── Plot 6: Model Evaluation Curves ──────────────────────────────
def plot_model_evaluation(preds_a, preds_b, save_path=None):
    """Side-by-side ROC and PR curves for both targets."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(_BG)
    for ax in [ax_roc, ax_pr]:
        ax.set_facecolor(_PANEL_BG)
        ax.tick_params(colors=_TEXT, labelsize=9)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.grid(True, color=_GRID, alpha=0.4, linewidth=0.5)

    for preds, label, color in [
        (preds_a, 'Target A: Betw. Drop', _ACCENT),
        (preds_b, 'Target B: Constr. Rise', _GOLD),
    ]:
        y_true = preds['y_true'].values
        y_pred = preds['y_pred'].values
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                        label=f'{label} (AUC = {roc_auc:.3f})')

            prec, rec, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = auc(rec, prec)
            ax_pr.plot(rec, prec, color=color, linewidth=2,
                       label=f'{label} (AP = {pr_auc:.3f})')

    ax_roc.plot([0, 1], [0, 1], color='#555555', linestyle='--', linewidth=0.8)
    ax_roc.set_xlabel('False Positive Rate', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate', fontsize=11)
    ax_roc.set_title('ROC Curves — Drift Prediction Model',
                     fontsize=13, fontweight='bold', pad=12)
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.05)
    ax_roc.legend(fontsize=10, facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT,
                  loc='lower right')

    ax_pr.set_xlabel('Recall', fontsize=11)
    ax_pr.set_ylabel('Precision', fontsize=11)
    ax_pr.set_title('Precision-Recall Curves — Drift Prediction Model',
                    fontsize=13, fontweight='bold', pad=12)
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.05)
    ax_pr.legend(fontsize=10, facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT,
                 loc='upper right')

    plt.tight_layout()
    _save_fig(fig, save_path)


# ── Main Entrypoint ──────────────────────────────────────────────
def plot_brokerage_drift(diagnostics: pd.DataFrame,
                         targets: pd.DataFrame,
                         preds_a: pd.DataFrame,
                         preds_b: pd.DataFrame,
                         deps_dict: Dict,
                         output_dir: str = 'outputs',
                         show_plot: bool = True,
                         save_path: Optional[str] = None):
    """
    Generate 6 separate brokerage drift plots, each saved as its own image.

    Plots:
        1. broker_trajectories.png  – Top broker influence over time
        2. drift_waterfall.png      – When each node experiences drift
        3. population_drift_rate.png – Fraction of nodes drifting per window
        4. phase_portrait.png       – Betweenness vs constraint with drift arrows
        5. drift_risk_heatmap.png   – Predicted risk over time (top 25 nodes)
        6. model_evaluation.png     – ROC + PR curves

    Parameters
    ----------
    diagnostics : DataFrame  – sena_diagnostics.csv
    targets     : DataFrame  – modeling_targets.csv
    preds_a     : DataFrame  – preds_target_a.csv (y_true, y_pred)
    preds_b     : DataFrame  – preds_target_b.csv (y_true, y_pred)
    deps_dict   : dict       – {node_id_str: department}
    output_dir  : str        – directory to save individual plots
    show_plot   : bool       – whether to display interactively
    save_path   : str | None – (legacy) combined dashboard path, ignored
    """
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dept_map = _dept_color_map(deps_dict.values())

    print("Generating 6 individual drift plots...")

    plot_broker_trajectories(
        diagnostics, dept_map, deps_dict,
        save_path=str(out / "drift_01_broker_trajectories.png")
    )
    plot_drift_waterfall(
        targets, deps_dict, dept_map,
        save_path=str(out / "drift_02_event_waterfall.png")
    )
    plot_population_drift_rate(
        targets,
        save_path=str(out / "drift_03_population_drift_rate.png")
    )
    plot_phase_portrait(
        targets, deps_dict, dept_map,
        save_path=str(out / "drift_04_phase_portrait.png")
    )
    plot_risk_heatmap(
        targets, preds_a,
        save_path=str(out / "drift_05_risk_heatmap.png")
    )
    plot_model_evaluation(
        preds_a, preds_b,
        save_path=str(out / "drift_06_model_evaluation.png")
    )

    print("All 6 drift plots generated.")
