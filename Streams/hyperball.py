import copy

import pandas as pd
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
    graph_file = "datasets/citations.csv"
    graph = pd.read_csv(graph_file, sep="\t", names=['ori', 'dest'])
    print(graph)

    nodes = pd.unique(graph[['ori', 'dest']].values.ravel('K'))
    
    b = 5
    counters = {}
    connections = {}
    last_update = {}
    for node in tqdm(nodes):
        counter = Counter(b=b)
        counter.hash_add(node)
        counters[node] = counter
        # connections[node] = [row['dest'] for _, row in graph.loc[graph['ori'] == node].iterrows()]
        # print(node, connections[node])
        last_update[node] = 0
    # Free memory
    # del graph

    stop = False
    t = 0
    while not stop:
        stop = True
        print("t: ", t)
        for node in tqdm(nodes):
            a = copy.deepcopy(counters[node])
            # for connection in connections:
            #     a.union(counters[connection])
            for _, row in graph.loc[graph['ori'] == node].iterrows():
                a.union(counters[row['dest']])
            # print("- node:", node, ", elems:", a.size() - counters[node].size())
            counter_changed = not (a == counters[node])
            # if counter_changed:
                # last_update[node] = t + 1
            a.save(get_file(graph_file=graph_file, radius=t, node=node))
            stop = stop and not counter_changed

        # Update counters
        print(last_update)
        for node in nodes:
            # if last_update[node] == t + 1:
            counters[node].load(get_file(graph_file=graph_file, radius=t, node=node))
        t += 1

    # # for node in nodes:
    # #     print(node, ": ", counters[node].size())