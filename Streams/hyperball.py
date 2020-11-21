import copy

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from counter import Counter


def get_file(graph_file, radius, node):
    filename = ""
    filename += graph_file.split(".")[0]
    filename += "/"
    filename += str(radius)
    filename += "/"
    filename += str(node)
    return filename


if __name__ == "__main__":
    experiment_name = "test"
    graph_file = "datasets/citations.csv"
    graph = pd.read_csv(graph_file, sep="\t", names=['ori', 'dest'])
    writer = SummaryWriter()

    nodes = pd.unique(graph[['ori', 'dest']].values.ravel('K'))
    nodes_map = {}

    b = 5
    counters = {}
    for node in tqdm(nodes):
        counter = Counter(b=b)
        counter.hash_add(node)
        counters[node] = counter
        nodes_map[node] = [row['dest'] for _, row in graph.loc[graph['ori'] == node].iterrows()]

    stop = False
    t = 0
    while not stop:
        stop = False
        print("t: ", t)
        changed = 0
        for node in tqdm(nodes_map):
            a = copy.deepcopy(counters[node])
            for successor in nodes_map[node]:
                changed += a.union(counters[successor])
            a.save(get_file(graph_file=graph_file, radius=t, node=node))
        print(f'Changed in this iteration {changed}')
        if changed == 0:
            break
        # Update counters
        for node in nodes:
            counters[node].load(get_file(graph_file=graph_file, radius=t, node=node))
        t += 1

    writer.close()
    # for node in nodes:
    #     print(node, ": ", counters[node].size())
