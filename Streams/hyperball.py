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
    
    b = 5
    counters = {}
    for node in nodes:
        counter = Counter(b=b)
        counter.hash_add(node)
        counters[node] = counter

    stop = False
    t = 0
    while not stop:
        stop = True
        print("t: ", t)
        for node in tqdm(nodes):
            a = copy.deepcopy(counters[node])
            for _, row in graph.loc[graph['ori'] == node].iterrows():
                a.union(counters[row['dest']])
            # print("- node:", node, ", elems:", a.size() - counters[node].size())
            a.save(get_file(graph_file=graph_file, radius=t, node=node))
            stop = stop and (a == counters[node])
        # Update counters
        for node in nodes:
            counters[node].load(get_file(graph_file=graph_file, radius=t, node=node))
        t += 1

    writer.close()
    # for node in nodes:
    #     print(node, ": ", counters[node].size())