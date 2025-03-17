import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from neural_lam import create_graph
from neural_lam.datastore.mdp import MDPDatastore


def main(args):
    dirname = os.path.dirname(args.datastore)
    datastore = MDPDatastore(args.datastore)
    output_path = os.path.join(dirname, "graph")
    xy = datastore.get_xy(category="state", stacked=False)

    graph = create_grid_graph(xy)

    edges_list = list(graph.edges)
    edges_tensor = torch.tensor(edges_list, dtype=torch.long).T
    edges_tensor = torch.cat([edges_tensor, edges_tensor.flip(0)], dim=1)

    os.makedirs(output_path, exist_ok=True)
    torch.save(edges_tensor, os.path.join(output_path, "edges.pt"))


def create_grid_graph(xy):
    """
    Create a grid-based graph where each node is connected to its immediate neighbors.

    Parameters:
    xy (numpy.ndarray): A (M, N, 2) array representing the coordinates of a regular grid.

    Returns:
    networkx.Graph: A graph where nodes are connected to their direct neighbors.
    """
    M, N, _ = xy.shape
    G = nx.Graph()

    node_index = lambda i, j: i * N + j

    for i in range(M):
        for j in range(N):
            node_id = node_index(i, j)
            G.add_node(node_id, pos=(xy[i, j, 0], xy[i, j, 1]))

            if j < N - 1:
                G.add_edge(node_id, node_index(i, j + 1))

            if i < M - 1:
                G.add_edge(node_id, node_index(i + 1, j))

    return G


def plot_grid_graph(G):
    """
    Plots a grid-based graph with nodes and edges.

    Parameters:
    G (networkx.Graph): The graph to plot.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.get_node_attributes(G, "pos")  # Extract node positions

    # Draw the graph
    nx.draw(G, pos, node_size=30, edge_color="gray", with_labels=False)

    plt.title("Grid Graph Visualization")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("datastore")
    args = parser.parse_args()

    main(args)
