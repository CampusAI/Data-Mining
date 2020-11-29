import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import SpectralClustering

from clustering import spectral_cluster

def read_graph(file):
    connections = pd.read_csv(file, sep=None, names=['ori', 'dest'], engine="python")
    graph = nx.from_pandas_edgelist(connections, source='ori', target='dest') 
    # nx.draw(graph)
    # plt.show()
    return graph


if __name__=="__main__":
    path = "/home/oleguer/Documents/p6/Data-Mining/Graph-Spectra/"
    graph = read_graph(path + "datasets/example1.dat")
    k = 4

    # Get adjacency matrix
    A = nx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
    np.fill_diagonal(A, 0)
    plt.matshow(A)
    plt.show()

    guessed_labels = spectral_cluster(A=A, k=4)

    # Run sklearn spectral clustering for comparison
    clustering = SpectralClustering(n_clusters=4,
                                    affinity='precomputed',
                                    random_state=0).fit(A)
    sklearn_labels = clustering.labels_

    print('---------- METRICS ----------')
    ARI = metrics.adjusted_rand_score(sklearn_labels, guessed_labels)
    print(f'ARI: {ARI}')