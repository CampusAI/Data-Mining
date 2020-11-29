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


def plot_graph(graph):
    pos = networkx.spring_layout(graph)
    networkx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'), node_size=500)
    networkx.draw_networkx_labels(graph, pos)
    networkx.draw_networkx_edges(graph, pos, edgelist=graph_.edges, edge_color='r', arrows=True)
    plt.show()


if __name__ == '__main__':
    connections_ = np.genfromtxt('datasets/example2.dat', delimiter=',')
    graph_dict_ = connections_to_graph_dict(connections_)
    graph_ = networkx.Graph(graph_dict_)

    plot_graph(graph_)