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
    """ method (str): 'fully-connected' or 'epsilon-ball'.
    """
    if method == 'epsilon-ball':
        return epsilon_ball_adjacency(data, epsilon=epsilon)
    elif method == 'fully-connected':
        return fully_connected_adjacency(data, sigma=sigma)


def spectral_cluster(A, k, plot_adjacency=False, plot_eigvals=False):
    """Perform spectral clustering on the data for the given number of clusters.

    Args:
        A (np.array(n, n)): Adjacency matrix
        k (int): Number of clusters.
        plot_adjacency (bool): Whether to plot the adjacency matrix.
        plot_eigvals (bool): Whether to plot the eigenvalues.

    Returns:
        labels (numpy.ndarray): A numpy array with the cluster assigment of each given point.
    """
    if plot_adjacency:
        plt.matshow(A)
        plt.show()

    # 1. L matrix
    D_root_inv = np.diag(np.power(np.sum(A, axis=0), -0.5))  # Compute Diagonal matrix D^-0.5
    L = np.dot(np.dot(D_root_inv, A), D_root_inv)

    # 2. Get k-largest eigenvals
    eigen_vals, eigen_vecs = np.linalg.eig(L)

    # print("eigen_vals", np.round(eigen_vals[(-eigen_vals).argsort()], decimals=3))
    eigen_vecs = eigen_vecs[:, (-eigen_vals).argsort()[:k]] # Sort by eigen_val descending

    # Sort eigenvalues increasing and estimate number of clusters
    eigen_vals.sort()
    eigen_vals = eigen_vals[::-1]
    print(f'Suggested number of clusters:')
    print(f'    By eigenvalues count: {len([v for v in eigen_vals if v == 1.])}')
    print(f'    By eigengap: {np.argmin(np.diff(eigen_vals)) + 1}')
    if plot_eigvals:
        plt.plot(range(eigen_vals.shape[0]), eigen_vals, 'o')
        plt.show()

    # 3. Normalize rows
    eigen_vecs = eigen_vecs / np.linalg.norm(eigen_vecs, axis=1)[:, None]

    # 4. Clustering
    kmeans = cluster.KMeans(n_clusters=k)
    labels = kmeans.fit(eigen_vecs).labels_
    # plt.scatter(eigen_vecs[:, 0], eigen_vecs[:, 1], c=labels)
    # plt.show()
    return labels


if __name__ == "__main__":
    data, real_labels = load_fake_data(dims=2, points_per_cluster=(10, 50), n_clusters=(4, 6), seed=3)
    k = np.unique(real_labels).shape[0]

    # Get adjacency matrix
    A = get_adjacency(data=data, method='epsilon-ball')
    np.fill_diagonal(A, 0)

    # Guess labels
    guessed_labels = spectral_cluster(A=A, k=k)

    plot_clusters(data, guessed_labels)

    print('---------- METRICS ----------')
    ARI = metrics.adjusted_rand_score(real_labels, guessed_labels)
    print(f'ARI: {ARI}')
