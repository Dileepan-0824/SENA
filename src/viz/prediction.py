"""
prediction.py
=============
Generates a solid 'Early Warning' visualization that directly links
the model's prediction with the actual network behavior.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

_BG = "#0f1117"
_PANEL_BG = "#181c25"
_GRID = "#2a2e3a"
_TEXT = "#e0e0e0"

def plot_prediction_alert(targets, preds_a, diagnostics, save_path=None):
    """
    Creates a 2-panel chart for the top 5 predicted-to-drift nodes:
    Top Panel: The model's predicted probability of drift (Early Warning)
    Bottom Panel: The actual temporal betweenness (Reality)
    """
    # 1. Align predictions with the true nodes and bins
    n_preds = len(preds_a)
    tail = targets.tail(n_preds).copy()
    tail['pred_risk'] = preds_a['y_pred'].values
    
    # 2. Find the nodes with the highest peak predicted risk
    peak_risks = tail.groupby('node_id')['pred_risk'].max()
    top_risky_nodes = peak_risks.nlargest(2).index.tolist()
    
    # Filter data for just these nodes
    pred_data = tail[tail['node_id'].isin(top_risky_nodes)]
    diag_data = diagnostics[diagnostics['node_id'].isin(top_risky_nodes)]
    
    # We only want to plot the time period covered by the predictions (the test set)
    start_bin = tail['bin_id'].min()
    end_bin = tail['bin_id'].max()
    diag_data = diag_data[(diag_data['bin_id'] >= start_bin) & (diag_data['bin_id'] <= end_bin)]
    
    # Setup Figure
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor(_BG)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)
    
    ax_pred = plt.subplot(gs[0])
    ax_true = plt.subplot(gs[1], sharex=ax_pred)
    
    colors = ["#ff6b6b", "#6ec6ff", "#fbbf24", "#34d399", "#c084fc"]
    
    # --- TOP PANEL: Predicted Risk ---
    ax_pred.set_facecolor(_PANEL_BG)
    ax_pred.axhline(y=0.5, color='#ff2222', linestyle='--', alpha=0.5, label='High Risk Threshold')
    
    for i, nid in enumerate(top_risky_nodes):
        node_preds = pred_data[pred_data['node_id'] == nid].sort_values('bin_id')
        ax_pred.plot(node_preds['bin_id'], node_preds['pred_risk'], 
                     color=colors[i], linewidth=2.5, label=f"Node {nid}", alpha=0.85)

    ax_pred.set_title("1. Model Prediction (Early Warning of Drift)", fontsize=14, color=_TEXT, fontweight='bold', pad=15)
    ax_pred.set_ylabel("Predicted Risk of Drift (Probability)", fontsize=12, color=_TEXT)
    ax_pred.tick_params(colors=_TEXT)
    ax_pred.set_ylim(-0.05, 1.05)
    for s in ax_pred.spines.values(): s.set_color(_GRID)
    ax_pred.grid(True, color=_GRID, alpha=0.5)
    ax_pred.legend(facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT, loc='upper left')

    # --- BOTTOM PANEL: Actual Ground Truth ---
    ax_true.set_facecolor(_PANEL_BG)
    
    for i, nid in enumerate(top_risky_nodes):
        node_true = diag_data[diag_data['node_id'] == nid].sort_values('bin_id')
        # Smooth slightly for readability
        vals = node_true['temporal_betweenness'].rolling(window=3, min_periods=1).mean()
        ax_true.plot(node_true['bin_id'], vals, 
                     color=colors[i], linewidth=2.5, alpha=0.85)
                     
        # Highlight ACTUAL drops (where target A is true)
        actual_drops = pred_data[(pred_data['node_id'] == nid) & (pred_data['target_a'] == 1)]
        for _, drop in actual_drops.iterrows():
            drop_bin = drop['bin_id']
            # Find the value in the true data at that bin
            val_at_drop = node_true[node_true['bin_id'] == drop_bin]['temporal_betweenness']
            if not val_at_drop.empty:
                ax_true.scatter(drop_bin, val_at_drop.iloc[0], color='red', s=100, zorder=5, marker='v')

    ax_true.scatter([], [], color='red', s=100, marker='v', label='Actual Drift Event (Betweenness Dropped)')
    ax_true.set_title("2. Actual Network Behavior (Reality)", fontsize=14, color=_TEXT, fontweight='bold', pad=15)
    ax_true.set_ylabel("Temporal Betweenness (Network Influence)", fontsize=12, color=_TEXT)
    ax_true.set_xlabel("Time Bin (Week)", fontsize=12, color=_TEXT)
    ax_true.tick_params(colors=_TEXT)
    for s in ax_true.spines.values(): s.set_color(_GRID)
    ax_true.grid(True, color=_GRID, alpha=0.5)
    ax_true.legend(facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT, loc='upper left')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__ == "__main__":
    _data_dir = pd.read_csv('data/processed/modeling_targets.csv')
    pass 
