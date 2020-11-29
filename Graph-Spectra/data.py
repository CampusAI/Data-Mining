from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import truncnorm


def load_fake_data(dims, cluster_means=None, cluster_stds=0.05, low=0.,
                   high=1., n_clusters=(2, 5), points_per_cluster=(15, 50), seed=None,
                   log_info=True):
    """Create a fake dataset of points to cluster.

    Args:
        dims (int): Dimensionality of the space.
        cluster_means (np.array): Array of shape (c, dims) containing the mean of each cluster
            where c is the number of clusters. If None, clusters will be randomly generated.
        cluster_stds (Union[np.array, float]): Numpy array of shape (c, dims) with standard
            deviations of the clusters. If float, the same std is used for all clusters.
        low (float): Lower bound for the randomly generated values in each point dimension.
        high (float): Upper bound for the randomly generated values in each point dimension.
        n_clusters (tuple): Tuple with min and max number of clusters to be randomly generated.
            This argument is silently discarded if cluster_means is not None.
        points_per_cluster (tuple): Tuple with min and max number of random points for each
            cluster to generate.
        seed (int): Seed for reproducibility.
        log_info (bool): Whether to print clusters info.

    Returns:
        (points, labels): A tuple where points is a numpy array of shape (N, dims), and labels
        is a numpy array of shape (N, ) with the label (from 0 to n_clusters) for each point.
    """
    if seed is not None:
        np.random.seed(seed)
    if cluster_means is None:
        if n_clusters[0] == n_clusters[1]:
            n_clusters = n_clusters[0]
        else:
            n_clusters = np.random.randint(n_clusters[0], n_clusters[1])
        cluster_means = low + np.random.random((n_clusters, dims)) * (high - low)
    else:
        n_clusters = cluster_means.shape[0]
    cluster_info = ''
    points = []
    labels = []
    for c in range(n_clusters):
        cluster_info += f'---------- CLUSTER {c} --------------\n'
        cluster_size = np.random.randint(points_per_cluster[0], points_per_cluster[1])
        cluster_info += f'Mean: {cluster_means[c]}\n'
        cluster_info += f'Number of points: {cluster_size}\n'
        scale = cluster_stds[c] if isinstance(cluster_stds, np.ndarray) else cluster_stds
        cluster_points = truncnorm.rvs(
            a=(low - cluster_means[c]) / scale,
            b=(high - cluster_means[c]) / scale,
            loc=cluster_means[c], scale=scale,
            size=(cluster_size, dims)
        )
        cluster_info += f'Points: {cluster_points.shape}\n'
        points.extend(cluster_points)
        labels.extend([c] * cluster_size)
    if log_info:
        print(cluster_info)
    points = np.array(points)
    labels = np.array(labels)
    shuffled_indices = np.random.choice(range(points.shape[0]), points.shape[0], replace=False)
    return points[shuffled_indices], labels[shuffled_indices]


def plot_clusters(points, labels, low=0, high=1):
    """Plot the given clusters.

    Args:
        points (numpy.ndarray): Numpy array of shape (N, 2) of points to be plotted.
        labels (numpy.ndarray): Numpy array of shape (N, ) containing the label of each point.
        low (float): Lower bound of the plot axis.
        high (float): Upper bound of the plot axis.
    """
    for label in np.unique(labels):
        cluster_points = points[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    points_, labels_ = load_fake_data(dims=2)
    plot_clusters(points_, labels_)