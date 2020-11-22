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
            coreachable[node] += t_next_size - t_size

    closeness_centralities = {
        node: coreachable[node] / sum_distance[node] if sum_distance[node] > 0 else 0
        for node in graph
    }
    lin_centralities = {  # If sum_distance[node] = 0 then coreachable[node] = 0
        node: coreachable[node]**2 / sum_distance[node] if sum_distance[node] > 0 else 1
        for node in graph
    }
    harmonic_centralities = sum_reciprocal
    return {
        'closeness_centrality': closeness_centralities,
        'lin_centrality': lin_centralities,
        'harmonic_centrality': harmonic_centralities
    }
