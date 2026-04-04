import pickle
from pathlib import Path
import networkx as nx
import numpy as np

data_dir = Path("data")
email_graph = nx.Graph()
prox_graph = nx.Graph()
snapshots_dir = data_dir / 'processed' / 'graphs' / 'snapshots'
for snap_file in snapshots_dir.glob("snapshot_*.pkl"):
    with open(snap_file, "rb") as f:
        G = pickle.load(f)
        for u, v, k, data in G.edges(keys=True, data=True):
            u, v = str(u), str(v)
            if data.get("layer") == "email":
                email_graph.add_edge(u, v)
            else:
                prox_graph.add_edge(u, v)

valid_email_nodes = [n for n, d in email_graph.degree() if d > 0]
valid_prox_nodes = [n for n, d in prox_graph.degree() if d > 0]
overlap_nodes = list(set(valid_email_nodes) & set(valid_prox_nodes))

if len(overlap_nodes) > 35:
    np.random.seed(42)
    sample_nodes = list(np.random.choice(overlap_nodes, 35, replace=False))
else:
    sample_nodes = overlap_nodes

sub_email = email_graph.subgraph(sample_nodes)
sub_prox = prox_graph.subgraph(sample_nodes)
print("Random sample email edges:", sub_email.number_of_edges())
print("Random sample prox edges:", sub_prox.number_of_edges())

# What about grabbing a dense component?
# E.g., largest connected component in intersection graph
inter_edges = [(u, v) for u, v in email_graph.edges() if prox_graph.has_edge(u, v)]
H = nx.Graph(inter_edges)
if H.number_of_nodes() > 0:
    largest_cc = max(nx.connected_components(H), key=len)
    top_35 = list(largest_cc)[:35]
    print("Connected sample email edges:", email_graph.subgraph(top_35).number_of_edges())
    print("Connected sample prox edges:", prox_graph.subgraph(top_35).number_of_edges())

