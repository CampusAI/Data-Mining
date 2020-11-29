import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.tests.test_qhull import points
from sklearn import metrics
from sklearn.cluster import SpectralClustering

from clustering import spectral_cluster
from clustering import epsilon_ball_adjacency
from clustering import fully_connected_adjacency
from plot_graph import plot_graph
from data import load_fake_data
from data import plot_clusters


def read_graph(file):
    connections = pd.read_csv(file, sep=None, names=['ori', 'dest', 'weight'], engine="python")
    graph = nx.from_pandas_edgelist(connections, source='ori', target='dest') 
    # nx.draw(graph)
    # plt.show()
    return graph


def read_points(file, delimiter=None, only2d=True):
    points = np.genfromtxt(file, delimiter=delimiter)
    if only2d:
        return points[:, :2]
    else:
        return points


def evaluate_clustering_from_graph(graph):
    """Perform our clustering and scipy clustering and evaluate.

    Args:
        graph (networkx.Graph):
    """
    # Get adjacency matrix
    A = nx.linalg.graphmatrix.adjacency_matrix(graph).toarray()
    np.fill_diagonal(A, 0)

    # Run our spectral clustering
    guessed_labels = spectral_cluster(A=A, k=k, plot_adjacency=False, plot_eigvals=True)

    labels_dict = {node: guessed_label for node, guessed_label in zip(graph.nodes, guessed_labels)}

    # Plot our spectral clustering
    plot_graph(graph, labels_dict)

    # Run sklearn spectral clustering for comparison
    clustering = SpectralClustering(n_clusters=k,
                                    affinity='precomputed',
                                    random_state=0).fit(A)
    sklearn_labels = clustering.labels_

    print('---------- METRICS ----------')
    ARI = metrics.adjusted_rand_score(sklearn_labels, guessed_labels)
    print(f'ARI: {ARI}')


def evaluate_clustering_from_points(points, k, mode='fully-connected', sigma=1., epsilon=0.2):
    if mode == 'fully-connected':
        A = fully_connected_adjacency(points, sigma)
    elif mode == 'epsilon-ball':
        A = epsilon_ball_adjacency(points, epsilon)
    else:
        raise ValueError(f'Given mode {mode} not recognized. Use fully-connected or epsilon-ball')
    np.fill_diagonal(A, 0)

    # Run our spectral clustering
    guessed_labels = spectral_cluster(A=A, k=k, plot_adjacency=False, plot_eigvals=True)

    labels_dict = {i: guessed_labels[i] for i in range(guessed_labels.shape[0])}

    # Plot our spectral clustering
    plot_clusters(points, guessed_labels)

    # Run sklearn spectral clustering for comparison
    clustering = SpectralClustering(n_clusters=k,
                                    affinity='precomputed',
                                    random_state=0).fit(A)
    sklearn_labels = clustering.labels_

    print('---------- METRICS ----------')
    ARI = metrics.adjusted_rand_score(sklearn_labels, guessed_labels)
    print(f'ARI: {ARI}')


if __name__ == "__main__":
    graph_ = read_graph('datasets/example1.dat')
    k = 4

    evaluate_clustering_from_graph(graph_)