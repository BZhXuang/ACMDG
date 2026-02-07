import copy
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np

np.random.seed(0)



def graph_initialization(data):
    feature_space = copy.deepcopy(data)
    dict_mapping = {}
    for i in range(len(feature_space)):
        dict_mapping[i] = i
    skeleton = nx.Graph()
    while (True):
        representatives, skeleton, dict_mapping = clustering_loop(feature_space, dict_mapping, skeleton)
        feature_space = data[representatives]
        if len(representatives) == 1:
            break
    representative = representatives[0]
    return skeleton, representative


def clustering_loop(feature_space, dict_mapping, skeleton):
    Graph = nx.Graph()
    representatives = []
    edges = nearest_neighbor_cal(feature_space)
    Graph.add_weighted_edges_from(edges)
    S = [Graph.subgraph(c).copy() for c in nx.connected_components(Graph)]
    for sub_S in S:
        representative = representatives_cal(sub_S)
        representatives.append(representative)
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
    for i in range(len(representatives)):
        representatives[i] = dict_mapping[representatives[i]]
    skeleton.add_weighted_edges_from(edges)
    dict_mapping = {}
    for i in range(len(representatives)):
        dict_mapping[i] = representatives[i]
    return representatives, skeleton, dict_mapping


def nearest_neighbor_cal(feature_space):
    neighbors = NearestNeighbors(n_neighbors=2).fit(feature_space)
    distance, nearest_neighbors = neighbors.kneighbors(feature_space, return_distance=True)
    distance = distance[:, 1]
    nearest_neighbors = nearest_neighbors.tolist()
    for i in range(len(nearest_neighbors)):
        nearest_neighbors[i].append(distance[i])
    return nearest_neighbors


def representatives_cal(sub_S):
    degree_dict = dict(sub_S.degree)
    max_degree = max(degree_dict.values())
    nodes_with_max_degree = [node for node, degree in degree_dict.items() if degree == max_degree]
    min_weighted_degree_sum = float('inf')
    min_weighted_degree_node = None
    for node in nodes_with_max_degree:
        weighted_degree_sum = sum(weight for _, _, weight in sub_S.edges(data='weight', nbunch=node))
        if weighted_degree_sum < min_weighted_degree_sum:
            min_weighted_degree_node = node
            min_weighted_degree_sum = weighted_degree_sum
    representative = min_weighted_degree_node
    return representative