import pandas as pd
import networkx as nx

# graph_file = "datasets/graph.csv"
graph_file = "datasets/citations.csv"
df = pd.read_csv(graph_file, sep="\t", names=["source", "target"])

Graphtype = nx.Graph()
G = nx.from_pandas_edgelist(df, create_using=Graphtype)
centrality = nx.algorithms.centrality.harmonic_centrality(G)
print(centrality)