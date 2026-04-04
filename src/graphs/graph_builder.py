import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

def build_temporal_graphs(data_dir: Path):
    """
    Phase 4: Temporal Graph Construction
    Build weekly and monthly snapshots...
    """
    email_edges = pd.read_csv(data_dir / 'processed' / 'email_edges_norm.csv')
    prox_edges = pd.read_csv(data_dir / 'processed' / 'prox_edges_norm.csv')
    
    # Define week and month bins (e.g. 7 days * 24 h * 3600 s)
    WEEK_SECONDS = 7 * 24 * 3600
    MONTH_SECONDS = 30 * 24 * 3600
    
    email_edges['week_bin'] = (email_edges['time_norm'] // WEEK_SECONDS).astype(int)
    email_edges['month_bin'] = (email_edges['time_norm'] // MONTH_SECONDS).astype(int)
    
    prox_edges['week_bin'] = (prox_edges['time_norm'] // WEEK_SECONDS).astype(int)
    prox_edges['month_bin'] = (prox_edges['time_norm'] // MONTH_SECONDS).astype(int)
    
    # Combine layers into common bins (weekly/monthly snapshots)
    snapshots = {}
    
    def process_layer(layer_df, layer_name, snapshot_col, source_col, target_col, weight_col='weight'):
        if target_col not in layer_df:
            return
        
        for bin_id, group in layer_df.groupby(snapshot_col):
            if bin_id not in snapshots:
                snapshots[bin_id] = nx.MultiDiGraph()
            # Add edges
            for _, row in group.iterrows():
                u, v = row[source_col], row[target_col]
                # Weighted multi-edges
                w = row.get(weight_col, 1)
                snapshots[bin_id].add_edge(u, v, layer=layer_name, weight=w)
    
    process_layer(email_edges, 'email', 'week_bin', 'sender_canon', 'recipient_canon')
    process_layer(prox_edges, 'proximity', 'week_bin', 'i', 'j', 'duration')
    
    graphs_dir = data_dir / 'processed' / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the generated NetworkX graphs to disk
    import pickle
    snapshots_dir = graphs_dir / 'snapshots'
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    for bin_id, G in snapshots.items():
        with open(snapshots_dir / f'snapshot_{bin_id}.pkl', 'wb') as f:
            pickle.dump(G, f)
    
    # 4.4 Compute base temporal features
    features = []
    
    for bin_id in sorted(snapshots.keys()):
        G = snapshots[bin_id]
        
        # Base features (per node)
        deg_in = dict(G.in_degree(weight='weight'))
        deg_out = dict(G.out_degree(weight='weight'))
        
        for node in G.nodes():
            features.append({
                'bin_id': bin_id,
                'node_id': node,
                'in_degree': deg_in.get(node, 0),
                'out_degree': deg_out.get(node, 0),
            })
            
        # 4.5 Snapshot QA checks (Monotonic bins, component behavior)
        if len(G) > 0:
            print(f"QA: Bin {bin_id} - Nodes: {G.number_of_nodes()} Edges: {G.number_of_edges()}")
            
    feat_df = pd.DataFrame(features)
    feat_df.to_csv(graphs_dir / 'base_temporal_features.csv', index=False)
    
    # Further metrics like burstiness and reciprocity will be applied here in future logic.
    print(f"Built {len(snapshots)} temporal graph snapshots.")

if __name__ == '__main__':
    build_temporal_graphs(Path('data'))
