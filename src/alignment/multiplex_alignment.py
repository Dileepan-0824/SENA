import pandas as pd
import numpy as np

def align_nodes(data_dir):
    """
    Phase 3: Multiplex Alignment
    - 3.1 Use department labels primarily from proximity nodes
    - 3.2 Treat email layer as department-agnostic when missing
    - 3.4 Strict vs Relaxed union
    """
    # Load processed
    harmonization = pd.read_csv(data_dir / 'processed' / 'harmonization.csv')
    node_deps = pd.read_csv(data_dir / 'node_departments.csv')
    email_edges = pd.read_csv(data_dir / 'processed' / 'email_edges_norm.csv')
    prox_edges = pd.read_csv(data_dir / 'processed' / 'prox_edges_norm.csv')
    
    # Map department labels from proximity nodes to the canonical nodes
    prox_nodes_with_dept = node_deps.set_index('node_id')['department'].to_dict()
    
    # Strict overlap scenario
    overlapping_nodes = set(prox_nodes_with_dept.keys()).intersection(set(harmonization['canonical_id']))
    
    # We will compute overlap nodes and assign department flags
    def assign_department(node):
        return prox_nodes_with_dept.get(node, 'unknown')
        
    # We produce outputs for strict (only intersecting nodes) and relaxed (union)
    
    # Relaxed union
    relaxed_nodes = pd.DataFrame({'canonical_id': list(set(harmonization['canonical_id']).union(set(prox_nodes_with_dept.keys())))})
    relaxed_nodes['department'] = relaxed_nodes['canonical_id'].apply(assign_department)
    relaxed_nodes.to_csv(data_dir / 'processed' / 'aligned_nodes_relaxed.csv', index=False)
    
    # Strict union
    strict_nodes = pd.DataFrame({'canonical_id': list(overlapping_nodes)})
    strict_nodes['department'] = strict_nodes['canonical_id'].apply(assign_department)
    strict_nodes.to_csv(data_dir / 'processed' / 'aligned_nodes_strict.csv', index=False)
    
    print(f"Alignment strategy completed: {len(strict_nodes)} strict nodes, {len(relaxed_nodes)} relaxed nodes")

if __name__ == '__main__':
    from pathlib import Path
    align_nodes(Path('data'))
