import networkx as nx
import numpy as np
import pandas as pd

from centrality import compute_centralities


EPS_ = 1e-5

def get_ground_truth(graph_file):
    df = pd.read_csv(graph_file, sep="\t", names=["source", "target"])

    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df, create_using=Graphtype)

    # Closeness centrality
    closeness_cent = nx.algorithms.centrality.closeness_centrality(G)
    # print("closeness_cent:")
    # print(sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:10])

    # Harmonic centrality
    harmonic_cent = nx.algorithms.centrality.harmonic_centrality(G)
    # print("harmonic_cent:")
    # print(sorted(harmonic_cent.items(), key=lambda x: x[1], reverse=True)[:10])
    return closeness_cent, harmonic_cent

def get_MAPE(approx, ground_truth):
    # Compute mean absolute percentage error
    error, n  = 0., 0.
    for key in approx:
        error += abs((approx[key] - ground_truth[key])/(ground_truth[key] + EPS_))
        n += 1.
    return 100.*error/n

if __name__=="__main__":
    # graph_file = "datasets/graph.csv"
    # graph_file = "datasets/citations.csv"
    graph_file = "datasets/emails.csv"
    
    # Our centrality
    print("Computing approximate centrality")
    approx_centralities = compute_centralities(graph_file='datasets/emails.csv', sphere_dir='datasets/emails/')
    print("DONE")
    
    # Ground truth
    print("Computing ground truth...")
    closeness_cent_gt, harmonic_cent_gt = get_ground_truth(graph_file)
    print("DONE")

    # Compute errors
    closeness_error = get_MAPE(approx=approx_centralities["closeness_centrality"],
                               ground_truth=closeness_cent_gt)
    print("closeness_error:", closeness_error, "%")

    harmonic_error = get_MAPE(approx=approx_centralities["harmonic_centralities"],
                               ground_truth=harmonic_cent_gt)
    print("harmonic_error:", harmonic_error, "%")