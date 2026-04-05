"""
prediction.py
=============
Generates solid visualization that maps the model's prediction 
(Early Warning Risk) directly to the actual drift events (Next-Week Crashes)
for both Target A (Betweenness) and Target B (Constraint).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_BG = "#0f1117"
_PANEL_BG = "#181c25"
_GRID = "#2a2e3a"
_TEXT = "#e0e0e0"

def plot_prediction_alert(targets, preds_a, preds_b, diagnostics, save_prefix="outputs/drift_prediction"):
    """
    Creates faceted charts for the top 4 nodes most at risk.
    Each node gets its own panel. Left Y-axis = True Metric. Right Y-axis = Predicted Risk.
    """
    df = targets.copy()
    n_preds = len(preds_a)
    df = df.tail(n_preds).copy()
    
    # 1. Align Predictions
    df['risk_a'] = preds_a['y_pred'].values
    df['risk_b'] = preds_b['y_pred'].values
    
    start_bin = df['bin_id'].min()
    end_bin = df['bin_id'].max()

    def create_facet_plot(target_name, metric_col, risk_col, is_drop_target=True):
        # Find top 4 nodes with highest peak risk for this target
        top_nodes = df.groupby('node_id')[risk_col].max().nlargest(4).index.tolist()
        node_df = df[df['node_id'].isin(top_nodes)]
        
        fig = plt.figure(figsize=(14, 12))
        fig.patch.set_facecolor(_BG)
        gs = gridspec.GridSpec(4, 1, hspace=0.3)
        fig.suptitle(f"{target_name}: 1-Week Early Warning Prediction Map\nTop 4 At-Risk Brokers", 
                     fontsize=18, color=_TEXT, fontweight='bold', y=0.95)
        
        colors = ["#ff6b6b", "#6ec6ff", "#fbbf24", "#34d399"]
        
        for i, nid in enumerate(top_nodes):
            ax1 = plt.subplot(gs[i])
            ax1.set_facecolor(_PANEL_BG)
            ax2 = ax1.twinx()
            
            node_data = node_df[node_df['node_id'] == nid].sort_values('bin_id')
            bins = node_data['bin_id'].values
            true_vals = node_data[metric_col].values
            risks = node_data[risk_col].values
            
            # Add smoothing to true metric for readability
            smoothed_true = pd.Series(true_vals).rolling(3, min_periods=1).mean()
            
            # Left Axis: True Metric (Line)
            l1 = ax1.plot(bins, smoothed_true, color=colors[i], linewidth=2.5, label=f"Actual {metric_col.replace('_', ' ').title()}", alpha=0.9)
            
            # Right Axis: Predicted Risk (Filled Area)
            l2 = ax2.plot(bins, risks, color='#ff2222', linestyle='--', linewidth=1.5, alpha=0.5, label='Predicted Risk (Next Week)')
            ax2.fill_between(bins, 0, risks, color='#ff2222', alpha=0.15)
            
            # Mark the actual target event
            # Target is exactly "did it happen at t+1". 
            # We want to show the model risking at t, and the event firing at t+1.
            target_col = 'target_a' if risk_col == 'risk_a' else 'target_b'
            events = node_data[node_data[target_col] == 1]
            
            event_xs = []
            event_ys = []
            for _, event in events.iterrows():
                # The event actually manifests in the true metric.
                # If target is 1 at bin t, the drop/spike occurred heading INTO bin t from t-1.
                event_xs.append(event['bin_id'])
                val = smoothed_true.iloc[list(bins).index(event['bin_id'])]
                event_ys.append(val)
                
            if is_drop_target:    
                scatter = ax1.scatter(event_xs, event_ys, color='red', s=120, marker='v', zorder=5, label='Actual Event Occurred')
            else:
                scatter = ax1.scatter(event_xs, event_ys, color='red', s=120, marker='^', zorder=5, label='Actual Event Occurred')

            ax1.set_ylabel("True Metric", color=colors[i], fontsize=11)
            ax2.set_ylabel("Predicted Risk", color='#ff2222', fontsize=11)
            ax1.tick_params(colors=_TEXT)
            ax2.tick_params(colors=_TEXT)
            
            ax1.set_title(f"Broker Node {nid}", color=_TEXT, loc='left', pad=5)
            for s in ax1.spines.values(): s.set_color(_GRID)
            for s in ax2.spines.values(): s.set_color(_GRID)
            ax1.grid(True, color=_GRID, alpha=0.5)
            ax2.set_ylim(0, 1.0)
            
            if i == 0:
                lines = l1 + l2 + [scatter]
                labels = [l.get_label() for l in l1+l2] + ['Actual Event Occurred']
                ax1.legend(lines, labels, loc='upper right', facecolor=_PANEL_BG, edgecolor=_GRID, labelcolor=_TEXT)
                
        plt.savefig(f"{save_prefix}_{target_col}.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

    create_facet_plot("Target A (Betweenness Drop)", "temporal_betweenness", "risk_a", is_drop_target=True)
    create_facet_plot("Target B (Constraint Spike)", "constraint", "risk_b", is_drop_target=False)

if __name__ == "__main__":
    targets = pd.read_csv('data/processed/modeling_targets.csv')
    preds_a = pd.read_csv('data/processed/preds_target_a.csv')
    preds_b = pd.read_csv('data/processed/preds_target_b.csv')
    diagnostics = pd.read_csv('data/processed/sena_diagnostics.csv')
    
    plot_prediction_alert(targets, preds_a, preds_b, diagnostics)
    print("Prediction plots generated.")
