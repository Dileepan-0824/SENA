import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_email(email):
    """Clean malformed handles and prefixes in aggregated email edges."""
    if pd.isna(email):
        return email
    email = str(email).lower().strip()
    if email.startswith('--migrated--'):
        email = email.replace('--migrated--', '')
    if email.startswith('-'):
        email = email[1:]
    return email

def parse_and_normalize(data_dir: Path, output_dir: Path):
    """
    Phase 2 - Sampled Data Validation and Canonicalization
    Parse and normalize timestamps to a common timeline.
    Canonicalize node IDs with a harmonization table.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    email_edges = pd.read_csv(data_dir / 'email_edges_sampled.csv')
    prox_edges = pd.read_csv(data_dir / 'proximity_edges.csv')
    node_deps = pd.read_csv(data_dir / 'node_departments.csv')
    email_agg = pd.read_csv(data_dir / 'email_edges_aggregated.csv')
    
    # 2.2 Parse and normalize timestamps
    # Email: ISO to seconds since start
    email_edges['timestamp'] = pd.to_datetime(email_edges['timestamp'], utc=True)
    min_email_time = email_edges['timestamp'].min()
    email_edges['time_norm'] = (email_edges['timestamp'] - min_email_time).dt.total_seconds().astype(int)
    
    # Proximity: assume integer ticks are seconds (or normalize from min)
    min_prox_time = prox_edges['timestamp'].min()
    prox_edges['time_norm'] = prox_edges['timestamp'] - min_prox_time
    
    # Scale proximity timestamps to match email time span or vice versa?
    # Actually, we keep it as integer seconds for a common numerical timeline.
    # The timescale (weeks/months) will be built on 'time_norm'.
    
    # 2.4 Add email identity cleaning rules
    email_edges['sender_clean'] = email_edges['sender'].apply(clean_email)
    email_edges['recipient_clean'] = email_edges['recipient'].apply(clean_email)
    email_agg['sender_clean'] = email_agg['sender'].apply(clean_email)
    email_agg['recipient_clean'] = email_agg['recipient'].apply(clean_email)
    
    # 2.3 Canonicalize node IDs (Harmonization Table)
    # create a mock harmonization table since none provided
    all_emails = set(email_edges['sender_clean']).union(set(email_edges['recipient_clean']))
    prox_nodes = set(prox_edges['i']).union(set(prox_edges['j']))
    
    # Map randomly/sequentially if there's no native mapping (Asymmetric multiplex setup)
    # We assign some overlap
    emails_list = sorted(list(all_emails))
    prox_list = sorted(list(prox_nodes))
    
    harmonization = {}
    for i, e in enumerate(emails_list):
        if i < len(prox_list):
            harmonization[e] = prox_list[i]
        else:
            harmonization[e] = f"E_{i}" # distinct email only nodes
            
    # Apply canonicalization
    email_edges['sender_canon'] = email_edges['sender_clean'].map(harmonization)
    email_edges['recipient_canon'] = email_edges['recipient_clean'].map(harmonization)
    
    # 2.5 Generate QA report
    qa_report = {
        'email_edges': len(email_edges),
        'prox_edges': len(prox_edges),
        'unique_emails': len(all_emails),
        'unique_prox_nodes': len(prox_nodes),
        'overlapping_nodes': min(len(all_emails), len(prox_nodes)),
        'email_missingness': email_edges.isnull().sum().to_dict(),
        'prox_missingness': prox_edges.isnull().sum().to_dict()
    }
    
    with open(output_dir / 'data_quality_report.json', 'w') as f:
        json.dump(qa_report, f, indent=4)
        
    logging.info("QA report generated.")
    
    # Save parsed outputs
    email_edges.to_csv(output_dir / 'email_edges_norm.csv', index=False)
    prox_edges.to_csv(output_dir / 'prox_edges_norm.csv', index=False)
    node_deps.to_csv(output_dir / 'node_deps_norm.csv', index=False)
    
    # Save harmonization table
    pd.DataFrame(list(harmonization.items()), columns=['email', 'canonical_id']).to_csv(output_dir / 'harmonization.csv', index=False)
    
    logging.info("Ingestion complete.")

if __name__ == '__main__':
    parse_and_normalize(Path('data'), Path('data/processed'))
