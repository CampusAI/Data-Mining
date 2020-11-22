import copy

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
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
    graph = loader.load_graph(graph_file, transpose=True)
    writer = SummaryWriter()

    b = Counter.DEFAULT_REGISTERS
    counters = {}
    for node in tqdm(graph):
        counter = Counter(b=b)
        counter.hash_add(node)
        counters[node] = counter

    t = 0
    while True:
        print("t: ", t)
        changed = 0
        for node in tqdm(graph):
            a = copy.deepcopy(counters[node])
            for successor in graph[node]:
                changed += a.union(counters[successor])
            a.save(get_file(graph_file=graph_file, radius=t, node=node))
        writer.add_scalar(f'changes', changed, t)
        if changed == 0:
            break
        # Update counters
        for node in graph:
            counters[node].load(get_file(graph_file=graph_file, radius=t, node=node))
        t += 1

    writer.close()
