import networkx
import numpy as np
from matplotlib import pyplot as plt


def connections_to_graph_dict(connections_array):
    """Transforms a numpy array of shape (N, 2) of connections x->y into a dictionary.

    Args:
        connections_array (numpy.ndarray): Array of shape (N, 2) where in each row the two
            elements are nodes in the graph.

    Returns:
        A dictionary with the nodes as keys and lists of successors as values.
    """
    graph_dict = {}
    for connection in connections_array:
        if connection[0] not in graph_dict:
            graph_dict[connection[0]] = []
        graph_dict[connection[0]].append(connection[1])
    return graph_dict


def plot_graph(graph, labels=None):
    """Plot the graph with different colors if labels are provided.

    Args:
        graph (networkx.Graph): A Graph object.
        labels (dict): Dictionary where keys are node ids and values their cluster assignments.
    """
    if labels is not None:
        unique_labels = set([v for _, v in labels.items()])
        colors = np.arange(0, 1, 1. / len(unique_labels))
        colors_list = [colors[labels[node]] for node in graph.nodes]
    else:
        colors_list = None
    pos = networkx.spring_layout(graph)
    networkx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), node_color=colors_list,
                                 node_size=500)
    networkx.draw_networkx_labels(graph, pos)
    networkx.draw_networkx_edges(graph, pos, edgelist=graph.edges, edge_color='r', arrows=True)
    plt.show()

