import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import SpectralClustering

from clustering import spectral_cluster
from plot_graph import plot_graph


def read_graph(file):
    connections = pd.read_csv(file, sep=None, names=['ori', 'dest'])
    graph = nx.from_pandas_edgelist(connections, source='ori', target='dest') 
    # nx.draw(graph)
    # plt.show()
    return graph


if __name__ == "__main__":
    graph = read_graph("datasets/example1.dat")
    k = 4

    # Get adjacency matrix
    A = nx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
    np.fill_diagonal(A, 0)
    plt.matshow(A)
    plt.show()

    # Run our spectral clustering
    guessed_labels = spectral_cluster(A=A, k=4)

    # Plot our spectral clustering
    plot_graph(graph, guessed_labels)

    # Run sklearn spectral clustering for comparison
    clustering = SpectralClustering(n_clusters=4,
                                    random_state=0).fit(A)
    sklearn_labels = clustering.labels_

    print('---------- METRICS ----------')
    ARI = metrics.adjusted_rand_score(sklearn_labels, guessed_labels)
    print(f'ARI: {ARI}')