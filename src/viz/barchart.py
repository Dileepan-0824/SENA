"""
barchart.py
===========
Cross-layer closure rate per department (bar chart)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from typing import Dict, Optional, List, Tuple

def plot_cross_layer_closure(closure_data: pd.DataFrame,
                            figsize: Tuple[int, int] = (12, 8),
                            save_path: Optional[str] = None, show_plot: bool = True):
    """
    Create bar chart of cross-layer closure rate per department
    
    Parameters:
    - closure_data: DataFrame with columns ['department', 'closure_rate']
    - figsize: Figure size tuple
    - save_path: Path to save figure
    - show_plot: Whether to display plot
    """
    
    if closure_data.empty:
        print("No closure data provided")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort departments by closure rate
    closure_data = closure_data.sort_values('closure_rate', ascending=False)
    
    # Create color palette
    colors = sns.color_palette("viridis", len(closure_data))
    
    # Create bar chart
    bars = ax.bar(range(len(closure_data)), closure_data['closure_rate'], 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    # Setting an adaptive offset instead of a hardcoded +0.01 which shatters the plot when Y max is 0.01
    y_max = max(closure_data['closure_rate']) if not closure_data.empty else 1
    offset = y_max * 0.02
    
    for i, (bar, rate) in enumerate(zip(bars, closure_data['closure_rate'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{rate:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Department', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-layer Closure Rate', fontsize=12, fontweight='bold')
    ax.set_title('Cross-layer Closure Rate per Department', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(range(len(closure_data)))
    ax.set_xticklabels(closure_data['department'], rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits
    ax.set_ylim(0, max(closure_data['closure_rate']) * 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cross-layer closure bar chart saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def compute_cross_layer_closure(email_graph: nx.Graph, proximity_graph: nx.Graph,
                              node_departments: Dict[str, str]) -> pd.DataFrame:
    """
    Compute cross-layer closure rate per department
    
    Cross-layer closure measures how many connections in one layer
    are also present in the other layer, normalized by total possible.
    
    Parameters:
    - email_graph: NetworkX graph for email layer
    - proximity_graph: NetworkX graph for proximity layer
    - node_departments: Dict mapping node -> department
    
    Returns:
    - DataFrame with columns ['department', 'closure_rate']
    """
    
    departments = list(set(node_departments.values()))
    if 'unknown' in departments:
        departments.remove('unknown')
    
    results = []
    
    for dept in departments:
        # Get nodes in this department
        dept_nodes = [node for node, d in node_departments.items() if d == dept]
        
        if len(dept_nodes) < 2:
            results.append({'department': dept, 'closure_rate': 0.0})
            continue
        
        # Create subgraphs for this department
        email_subgraph = email_graph.subgraph(dept_nodes)
        proximity_subgraph = proximity_graph.subgraph(dept_nodes)
        
        # Get edge sets
        email_edges = set(email_subgraph.edges())
        proximity_edges = set(proximity_subgraph.edges())
        
        # Normalize edges (make undirected for comparison)
        email_edges_undirected = {tuple(sorted(edge)) for edge in email_edges}
        proximity_edges_undirected = {tuple(sorted(edge)) for edge in proximity_edges}
        
        # Compute closure metrics using edge weights
        if len(email_edges_undirected) == 0 and len(proximity_edges_undirected) == 0:
            closure_rate = 0.0
        else:
            intersection_weight_sum = 0
            union_weight_sum = 0
            
            all_edges = email_edges_undirected | proximity_edges_undirected
            for edge in all_edges:
                u, v = edge
                # check both graphs regardless of order
                w_email = email_subgraph[u][v].get('weight', 1) if email_subgraph.has_edge(u, v) else 0
                if w_email == 0 and email_subgraph.has_edge(v, u): w_email = email_subgraph[v][u].get('weight', 1)
                
                w_prox = proximity_subgraph[u][v].get('weight', 1) if proximity_subgraph.has_edge(u, v) else 0
                if w_prox == 0 and proximity_subgraph.has_edge(v, u): w_prox = proximity_subgraph[v][u].get('weight', 1)
                
                # Intersection is the minimum weight (bottleneck) across both layers
                intersection_weight_sum += min(w_email, w_prox)
                # Union is the maximum weight across both layers
                union_weight_sum += max(w_email, w_prox)

            closure_rate = intersection_weight_sum / union_weight_sum if union_weight_sum > 0 else 0.0
        
        results.append({'department': dept, 'closure_rate': closure_rate})
    
    return pd.DataFrame(results)

def create_sample_closure_data() -> pd.DataFrame:
    """Create sample closure data for testing"""
    departments = ['DCAR', 'DG', 'DISQ', 'DMCT', 'DMI', 'DSE', 'DST', 'SCOM', 'SDOC', 'SFLE']
    
    # Generate realistic closure rates (higher for some departments)
    np.random.seed(42)
    closure_rates = []
    for dept in departments:
        if dept in ['DCAR', 'DMI', 'DSE']:  # Higher closure for these
            rate = np.random.beta(8, 2)  # Biased toward high values
        else:
            rate = np.random.beta(2, 3)  # Biased toward low values
        
        closure_rates.append(rate)
    
    return pd.DataFrame({
        'department': departments,
        'closure_rate': closure_rates
    })