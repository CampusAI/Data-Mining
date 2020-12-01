import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import spatial
from sklearn import cluster
from sklearn import metrics

from data import load_fake_data, plot_clusters


def fully_connected_adjacency(data, sigma=None, use_mod=True, k=5):
    """Return an adjacency matrix A from the negative exp distance.

    The adjacency matrix A is given by:
        A_{i,j} = exp(-||data[i] - data[j]||^2 / (sigma[i] * sigma[j]))
    where sigma[i] = sigma if sigma is not None, otherwise it is given by:
        sigma[i] = ||data[i] - data[l]||^2 where data[l] is the k-nearest neighbor of data[i].
    For reference, see:
        https://papers.nips.cc/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf

    Args:
        data (numpy.ndarray): Numpy array of points of shape (N, d).
        sigma (float): Sigma value to use when computing the adjacency matrix. If None,
            it will be automatically computed for each data point as described above.
        k (int): The value of the k-nearest neighbor used to compute sigma, if sigma is None.

    Returns:
        A (numpy.ndarray): A (N, N) numpy array representing the adjacency matrix.
    """
    if sigma is not None:
        def weight(a, b):
            return np.exp(-(np.linalg.norm(a - b) ** 2) / (2 * (sigma ** 2)))
        return spatial.distance.cdist(data, data, metric=weight)
    else:
        n = data.shape[0]
        sigmas = np.empty(n)
        distances = spatial.distance_matrix(data, data)
        sorted_neighbors = distances.argsort()
        for i in range(n):
            if use_mod:
                k_neighborhood = data[sorted_neighbors[i, :k]]
                k_neighborhood_dists = spatial.distance_matrix(k_neighborhood, k_neighborhood)
                neighbors = np.tril(k_neighborhood_dists)
                sigmas[i] = np.mean(neighbors)
            else:
                sigmas[i] = np.linalg.norm(data[i] - data[sorted_neighbors[i, k]])
        distances = np.exp(-(distances**2) / (sigmas[:, np.newaxis] * sigmas))
        return distances


def epsilon_ball_adjacency(data, epsilon):
    def weight(a, b):
        return 0. if np.linalg.norm(a-b) > epsilon else 1.
    return spatial.distance.cdist(data, data, metric=weight)


def get_adjacency(data, method, epsilon=0.1, sigma=1.):
    """ method (str): 'fully-connected' or 'epsilon-ball'.
    """
    if method == 'epsilon-ball':
        return epsilon_ball_adjacency(data, epsilon=epsilon)
    elif method == 'fully-connected':
        return fully_connected_adjacency(data, sigma=sigma)


def spectral_cluster(A, k, tol=1e-3, max_k=None, plot_adjacency=False, plot_eigvals=False):
    """Perform spectral clustering on the data for the given number of clusters.

    Args:
        A (np.array(n, n)): Adjacency matrix
        k (Union[int, str]): The number of clusters or the cluster estimation method to use to
            automatically determine it. Available methods: '1-eigenvals', 'eigengap'.
        tol (float): When using k='1-eigenvals', k is equal to the number of eigenvalues that
            are in the range [1-tol, 1+tol].
        max_k (int): Maximum number of clusters to detect in automatic mode.
        plot_adjacency (bool): Whether to plot the adjacency matrix.
        plot_eigvals (bool): Whether to plot the eigenvalues.

    Returns:
        (labels, k) (tuple): A tuple containing an array of labels and the number of clusters used.
    """
    if plot_adjacency:
        plt.matshow(A)
        plt.show()

    # 1. L matrix
    D_root_inv = np.diag(np.power(np.sum(A, axis=1), -0.5))  # Compute Diagonal matrix D^-0.5
    L = np.dot(np.dot(D_root_inv, A), D_root_inv)

    # 2. Get k-largest eigenvals
    eigen_vals, eigen_vecs = np.linalg.eig(L)
    sorted_eigenvals = np.sort(eigen_vals)[::-1]
    if plot_eigvals:
        plt.plot(range(sorted_eigenvals.shape[0]), sorted_eigenvals, 'o')
        plt.show()

    if isinstance(k, str):
        if k == '1-eigenvals':
            k = len([v for v in sorted_eigenvals if 1.-tol < v < 1.+tol])
            if max_k is not None:
                k = min(k, max_k)
        elif k == 'eigengap':
            if max_k is not None:
                k = np.argmin(np.diff(sorted_eigenvals[:max_k])) + 1
            else:
                k = np.argmin(np.diff(sorted_eigenvals)) + 1
        else:
            raise ValueError(f'Cluster estimation method \'{k}\' not recognized.')
    print(f'Clustering with {k} clusters.')

    # Get k eigenvectors
    eigen_vecs = eigen_vecs[:, (-eigen_vals).argsort()[:k]]

    # 3. Normalize rows
    eigen_vecs = eigen_vecs / np.linalg.norm(eigen_vecs, axis=1)[:, None]

    # 4. Clustering
    kmeans = cluster.KMeans(n_clusters=k)
    labels = kmeans.fit(eigen_vecs).labels_
    return labels, k