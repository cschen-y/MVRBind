import networkx as nx
import numpy as np


def compute_eccentricity_with_zero(G):
    eccentricity = {}
    for node in G.nodes():
        try:
            distances = nx.single_source_shortest_path_length(G, node)
            eccentricity[node] = max(distances.values())
        except nx.NetworkXNoPath:
            eccentricity[node] = 0
    return eccentricity

def get_node_topology(adj_matrix):
    G = nx.from_numpy_matrix(adj_matrix)
    DG = dict(G.degree())
    NC = {node: np.mean([DG[neighbor] for neighbor in G.neighbors(node)]) if DG[node] > 0 else 0
          for node in G.nodes()}
    BC = nx.betweenness_centrality(G)
    CL = nx.closeness_centrality(G)
    EC = compute_eccentricity_with_zero(G)
    all_properties = []
    for node in G.nodes():
        node_properties = [DG[node], NC[node], BC[node], CL[node], EC[node]]
        all_properties.append(node_properties)

    return all_properties