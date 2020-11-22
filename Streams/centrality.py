import os

from tqdm import tqdm

import loader


def compute_centralities(graph_file, sphere_dir):
    graph = loader.load_graph(graph_file, transpose=True)
    max_t = max(int(f) for f in os.listdir(sphere_dir))

    sum_distance = {node: 0 for node in graph}
    sum_reciprocal = {node: 0 for node in graph}
    coreachable = {node: 0 for node in graph}

    for node in tqdm(graph):
        for t in range(1, max_t + 1):
            counter_t = loader.load_sphere(sphere_dir, radius=t - 1, node=node)
            counter_t_next = loader.load_sphere(sphere_dir, radius=t, node=node)
            t_size = counter_t.size()
            t_next_size = counter_t_next.size()
            sum_distance[node] += float(t) * (t_next_size - t_size)
            sum_reciprocal[node] += 1./t * (t_next_size - t_size)
            if t == max_t - 1:
                coreachable[node] = float(t_next_size)

    closeness_centralities = {node: 1. / sum_distance[node] for node in graph}
    lin_centralities = {node: coreachable[node] / sum_distance[node] for node in graph}
    harmonic_centralities = sum_reciprocal
    return {
        'closeness_centrality': closeness_centralities,
        'lin_centrality': lin_centralities,
        'harmonic_centralities': harmonic_centralities
    }


if __name__ == '__main__':
    centralities = compute_centralities(graph_file='datasets/emails.csv', sphere_dir='datasets/emails/')