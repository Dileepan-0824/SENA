import pandas as pd
import numpy as np
from pathlib import Path

def build_targets(data_dir: Path):
    """
    Phase 6: Custom Objective Definition
    - Target A: significant drop in temporal betweenness rank.
    - Target B: significant increase in structural constraint.
    """
    diagnostics = pd.read_csv(data_dir / 'processed' / 'sena_diagnostics.csv')
    
    # 6.2 Target A: rank drop
    # 6.3 Target B: constraint increase
    diagnostics.sort_values(['node_id', 'bin_id'], inplace=True)
    
    # Calculate deltas for next window (shift -1 for target)
    diagnostics['next_betweenness'] = diagnostics.groupby('node_id')['temporal_betweenness'].shift(-1)
    diagnostics['next_constraint'] = diagnostics.groupby('node_id')['constraint'].shift(-1)
    
    # Define significant thresholds (e.g. drop of more than 0.1 for rank, increase of 0.2 for constraint)
    diagnostics['target_a'] = (diagnostics['temporal_betweenness'] - diagnostics['next_betweenness'] > 0.1).astype(int)
    diagnostics['target_b'] = (diagnostics['next_constraint'] - diagnostics['constraint'] > 0.2).astype(int)
    
    # Drop NAs (last sequence)
    modeling_data = diagnostics.dropna(subset=['next_betweenness', 'next_constraint'])
    
    # 6.4 Publish thresholding and class-balance diagnostics
    print(f"Target A Class Balance: {modeling_data['target_a'].mean()}")
    print(f"Target B Class Balance: {modeling_data['target_b'].mean()}")
    
    modeling_data.to_csv(data_dir / 'processed' / 'modeling_targets.csv', index=False)
    print("Targets built successfully.")

if __name__ == '__main__':
    build_targets(Path('data'))
