import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import sklearn.cluster

from data import load_fake_data, plot_clusters

def get_data():
    return np.array([[1, 0], [2, 0], [3, 0], [7, 0], [8, 0], [9, 0]])

def distance(a, b, sigma):
    return np.exp(-(np.linalg.norm(a-b)**2)/(2*sigma**2))


def spectral_cluster(data, k):
    # 1. Affinity matrix
    A = scipy.spatial.distance.cdist(data, data, metric=distance, sigma=1)  # Compute Affinity matrix
    np.fill_diagonal(A, 0)  # Fill diagonal with zeros

    # 2. L matrix
    D_root_inv = np.diag(np.power(np.sum(A, axis=0), -0.5))  # Compute Diagonal matrix D^-0.5
    L = np.dot(np.dot(D_root_inv, A), D_root_inv)
    
    # 3. Get k-largest eigenvals
    eigen_vals, eigen_vecs = np.linalg.eig(L)
    eigen_vecs = eigen_vecs[(-eigen_vals).argsort()[:k]]  # Sort by eigen_val in descending order
    eigen_vecs = eigen_vecs.T

    # 4. Normalize rows
    eigen_vecs = eigen_vecs / np.linalg.norm(eigen_vecs, axis=1)[:, None]

    # 5. Clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(eigen_vecs)
    return kmeans.labels_


if __name__=="__main__":
    k = 2

    data, real_labels = load_fake_data(dims=2, seed=0)
    guessed_labels = spectral_cluster(data=data, k=k)
    plot_clusters(data, guessed_labels)

    accuracy = 1. - (np.count_nonzero(guessed_labels)/len(guessed_labels))
    print("Accuracy: ", accuracy)
