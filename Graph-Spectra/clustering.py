import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import spatial
from sklearn import cluster
from sklearn import metrics

from data import load_fake_data, plot_clusters


def fully_connected_adjacency(data, sigma):
    def weight(a, b):
        return np.exp(-(np.linalg.norm(a - b) ** 2) / (2 * (sigma ** 2)))
    return spatial.distance.cdist(data, data, metric=weight)


def epsilon_ball_adjacency(data, epsilon):
    def weight(a, b):
        return 0. if np.linalg.norm(a-b) > epsilon else 1.
    return spatial.distance.cdist(data, data, metric=weight)


def knn_adjacency(data, k):
    distances = spatial.distance_matrix(data, data)
    sorted_neighbors = distances.argsort()
    distances[:] = 0
    for i in range(distances.shape[0]):
        distances[i, sorted_neighbors[1:k+1]] = 1.
    return distances


def get_adjacency(data, method, epsilon=0.1, sigma=1.):
    if method == 'epsilon-ball':
        return epsilon_ball_adjacency(data, epsilon=epsilon)
    elif method == 'fully-connected':
        return fully_connected_adjacency(data, sigma=sigma)


def get_sym_norm_laplacian(D, A):
    res = np.zeros((D.shape[0], D.shape[0]))
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i == j and D[i, i] != 0.:
                res[i, j] = 1.
            elif i != j and A[i, j] != 0.:
                res[i, j] = -1. / np.sqrt(D[i, i] * D[j, j])
            else:
                res[i, j] = 0.
    return res


def spectral_cluster(data, k, method='fully-connected', epsilon=0.1, sigma=1.,
                     normalize_laplacian=False):
    """Perfomr spectral clustering on the data for the given number of clusters.

    Args:
        data (numpy.ndarray): Numpy array of shape (N, d) with d being the dimensionality of
            the points.
        k (int): Number of clusters.
        method (str): 'fully-connected' or 'epsilon-ball'.
        epsilon (float): Size of the epsilon ball if using 'epsilon-ball' method.
        sigma (float): Value of sigma if using the 'fully-connected' method.
        normalize_laplacian (bool): Whether to normalize the laplacian.

    Returns:
        labels (numpy.ndarray): A numpy array with the cluster assigment of each given point.
    """
    # 1. Affinity matrix
    A = get_adjacency(data=data, method=method, epsilon=epsilon, sigma=sigma)

    # 2. L matrix
    D = np.diag(np.sum(A, axis=0))
    if normalize_laplacian:
        L = get_sym_norm_laplacian(D, A)
    else:
        L = D - A

    # 3. Get k-largest eigenvals
    eigen_vals, eigen_vecs = np.linalg.eig(L)
    eigen_vecs = eigen_vecs[:, (-eigen_vals).argsort()[:k]]  # Sort by eigen_val descending

    # 4. Normalize rows
    eigen_vecs = eigen_vecs / np.linalg.norm(eigen_vecs, axis=1)[:, None]

    # 5. Clustering
    kmeans = cluster.KMeans(n_clusters=k)
    labels = kmeans.fit(eigen_vecs).labels_
    return labels


if __name__ == "__main__":
    data, real_labels = load_fake_data(dims=2, points_per_cluster=(15, 40), n_clusters=(3, 10))
    k = np.unique(real_labels).shape[0]

    guessed_labels = spectral_cluster(data=data, k=k, method='epsilon-ball', sigma=1.,
                                      normalize_laplacian=False)

    plot_clusters(data, guessed_labels)

    print('---------- METRICS ----------')
    ARI = metrics.adjusted_rand_score(real_labels, guessed_labels)
    print(f'ARI: {ARI}')
