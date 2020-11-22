import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import counter


def load_graph(path, transpose=False):
    graph = pd.read_csv(path, sep=None, names=['ori', 'dest'])
    nodes = pd.unique(graph[['ori', 'dest']].values.ravel('K'))
    if transpose:
        from_col, to_col = 'dest', 'ori'
    else:
        from_col, to_col = 'ori', 'dest'
    nodes_map = {}
    for node in tqdm(nodes):
        nodes_map[node] = [row[to_col] for _, row in graph.loc[graph[from_col] == node].iterrows()]
    return nodes_map


def load_sphere(dir, radius, node):
    c = counter.Counter(b=5)
    c.load(os.path.join(dir, str(radius), str(node)))
    return c