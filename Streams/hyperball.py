import copy

import pandas as pd
from tqdm import tqdm

from counter import Counter
import loader


def get_file(graph_file, radius, node):
    filename = ""
    filename += graph_file.split(".")[0]
    filename += "/"
    filename += str(radius)
    filename += "/"
    filename += str(node)
    return filename


if __name__ == "__main__":
    graph_file = "datasets/emails.csv"
    graph = loader.load_graph(graph_file)

    b = 5
    counters = {}
    for node in tqdm(graph):
        counter = Counter(b=b)
        counter.hash_add(node)
        counters[node] = counter

    stop = False
    t = 0
    while not stop:
        stop = False
        print("t: ", t)
        changed = 0
        for node in tqdm(graph):
            a = copy.deepcopy(counters[node])
            for successor in graph[node]:
                changed += a.union(counters[successor])
            a.save(get_file(graph_file=graph_file, radius=t, node=node))
        print(f'Changed in this iteration {changed}')
        if changed == 0:
            break
        # Update counters
        for node in graph:
            counters[node].load(get_file(graph_file=graph_file, radius=t, node=node))
        t += 1
