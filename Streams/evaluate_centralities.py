from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from centrality import compute_centralities
import loader

plt.style.use('ggplot')

EPS_ = 1e-5


def get_ground_truth(graph_file):
    graph = loader.load_graph(graph_file)
    G = nx.Graph(graph)

    # Closeness centrality
    closeness_cent = nx.algorithms.centrality.closeness_centrality(G, wf_improved=False)
    # print("closeness_cent:")
    # print(sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:10])

    # Harmonic centrality
    harmonic_cent = nx.algorithms.centrality.harmonic_centrality(G)
    # print("harmonic_cent:")
    # print(sorted(harmonic_cent.items(), key=lambda x: x[1], reverse=True)[:10])

    return closeness_cent, harmonic_cent


def get_MAPE(approx, ground_truth):
    # Compute mean absolute percentage error
    error, n = 0., 0.
    for key in approx:
        error += abs((approx[key] - ground_truth[key])) / (ground_truth[key] + EPS_)
        n += 1
    return 100. * error / n


def plot_centralities(ground_truth, approx, title, ylabel):
    sorted_keys = sorted(ground_truth.keys(), key=lambda k: ground_truth[k], reverse=True)
    x_axis = range(len(sorted_keys))
    plt.plot(x_axis, [ground_truth[k] for k in sorted_keys], label='Ground truth')
    plt.plot(x_axis, [approx[k] for k in sorted_keys], label='Approximate')
    plt.legend(fontsize=25)
    plt.title(title, fontsize=30)
    plt.xlabel('Nodes', fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.show()


def compute_sorting_stats(closeness_gt, harmonic_gt, closeness_approx, harmonic_approx,
                          top_percent=5):
    # Sorting the keys of each dictionary by
    sorted_closeness_gt_keys = sorted(closeness_gt.keys(), key=lambda k: closeness_gt[k])
    sorted_harmonic_gt_keys = sorted(harmonic_gt.keys(), key=lambda k: harmonic_gt[k])
    sorted_closeness_approx_keys = sorted(closeness_approx.keys(), key=lambda k: closeness_approx[k])
    sorted_harmonic_approx_keys = sorted(harmonic_approx.keys(), key=lambda k: harmonic_approx[k])

    mse_closeness = 0.
    for i, k in enumerate(sorted_closeness_gt_keys):
        mse_closeness += abs(i - sorted_closeness_approx_keys.index(k))
    mse_closeness /= len(sorted_closeness_gt_keys)

    mse_harmonic = 0.
    for i, k in enumerate(sorted_harmonic_gt_keys):
        mse_harmonic += abs(i - sorted_harmonic_approx_keys.index(k))
    mse_harmonic /= len(harmonic_gt)

    n = len(sorted_closeness_gt_keys)
    top_percent = int(n * top_percent / 100)
    top_percent_closeness_gt = set(sorted_closeness_gt_keys[:top_percent])
    top_percent_harmonic_gt = set(sorted_harmonic_gt_keys[:top_percent])
    top_percent_closeness_approx = set(sorted_closeness_approx_keys[:top_percent])
    top_percent_harmonic_approx = set(sorted_harmonic_approx_keys[:top_percent])

    closeness_intersect = top_percent_closeness_gt.intersection(top_percent_closeness_approx)
    closeness_union = top_percent_closeness_gt.union(top_percent_closeness_approx)
    harmonic_intersect = top_percent_harmonic_gt.intersection(top_percent_harmonic_approx)
    harmonic_union = top_percent_harmonic_gt.union(top_percent_harmonic_approx)
    closeness_jaccard = len(closeness_intersect) / len(closeness_union)
    harmonic_jaccard = len(harmonic_intersect) / len(harmonic_union)

    return mse_closeness, mse_harmonic, closeness_jaccard, harmonic_jaccard


if __name__ == "__main__":
    # graph_file = "datasets/graph.csv"
    # graph_file = "datasets/citations.csv"
    graph_file = "datasets/emails.csv"
    
    # Our centrality
    print("Computing approximate centrality")
    approx_centralities = compute_centralities(
        graph_file='datasets/emails.csv',
        sphere_dir='datasets/emails/'
    )
    print("DONE")
    
    # Ground truth
    print("Computing ground truth...")
    closeness_cent_gt, harmonic_cent_gt = get_ground_truth(graph_file)
    print("DONE")

    plot_centralities(closeness_cent_gt, approx_centralities['closeness_centrality'],
                      'Closeness centrality', ylabel='Closeness centrality')
    plot_centralities(harmonic_cent_gt, approx_centralities['harmonic_centrality'],
                      'Harmonic centrality', ylabel='Harmonic centrality')

    # Compute errors
    closeness_error = get_MAPE(approx=approx_centralities["closeness_centrality"],
                               ground_truth=closeness_cent_gt)

    print("closeness_error:", closeness_error, "%")

    harmonic_error = get_MAPE(approx=approx_centralities["harmonic_centrality"],
                              ground_truth=harmonic_cent_gt)
    print("harmonic_error:", harmonic_error, "%")

    mse_closeness, mse_harmonic, closeness_jac, harmonic_jac = compute_sorting_stats(
        closeness_gt=closeness_cent_gt,
        harmonic_gt=harmonic_cent_gt,
        closeness_approx=approx_centralities['closeness_centrality'],
        harmonic_approx=approx_centralities['harmonic_centrality'],
        top_percent=10
    )

    print('MSE sorting closeness: ' + str(mse_closeness))
    print('MSE sorting harmonic: ' + str(mse_harmonic))
    print('Jaccard closeness: ' + str(closeness_jac))
    print('Jaccard harmonic: ' + str(harmonic_jac))