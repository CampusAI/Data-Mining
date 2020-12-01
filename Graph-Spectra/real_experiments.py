import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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


def read_points(file, only2d=True):
    points = np.loadtxt(file)
    if only2d:
        points = points[:, :2]
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
    guessed_labels, k = spectral_cluster(A=A, k=k, plot_adjacency=False, plot_eigvals=True)

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
    print(f'ARI scipy: {ARI}')


def evaluate_clustering_from_points(points, k, mode='fully-connected', max_k=None, sigma=1.,
                                    epsilon=0.2, knn=3, use_mod=True, real_labels=None):
    if mode == 'fully-connected':
        A = fully_connected_adjacency(points, sigma=sigma)
    elif mode == 'epsilon-ball':
        A = epsilon_ball_adjacency(points, epsilon)
    elif mode == 'self-tuning':
        A = fully_connected_adjacency(points, k=knn, use_mod=use_mod)
    else:
        raise ValueError(f'Given mode {mode} not recognized. Use fully-connected or epsilon-ball')
    np.fill_diagonal(A, 0)

    # Run our spectral clustering
    guessed_labels, k = spectral_cluster(A=A, k=k, max_k=max_k, plot_adjacency=False,
                                         plot_eigvals=False)

    # Plot our spectral clustering
    plot_clusters(points, guessed_labels)

    # Run sklearn spectral clustering for comparison
    clustering = SpectralClustering(n_clusters=k,
                                    affinity='precomputed',
                                    random_state=0).fit(A)
    sklearn_labels = clustering.labels_

    print('---------- METRICS ----------')
    ARI = metrics.adjusted_rand_score(sklearn_labels, guessed_labels)
    print(f'ARI scipy: {ARI}')
    if real_labels is not None:
        ARI_real = metrics.adjusted_rand_score(sklearn_labels, real_labels)
        print(f'ARI real labels: {ARI_real}')


if __name__ == "__main__":
    points_ = read_points(file='datasets/Compound.txt', only2d=True)
    #points_, labels_ = load_fake_data(dims=2, n_clusters=(10, 15), cluster_stds=0.03)
    #k = np.unique(labels_).shape[0]

    evaluate_clustering_from_points(points_, k=6, max_k=15, mode='self-tuning',
                                    use_mod=True, knn=7, epsilon=3., sigma=0.4)
