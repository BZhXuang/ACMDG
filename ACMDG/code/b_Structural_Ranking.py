import numpy as np
import heapq


np.random.seed(0)

def calculate_single_node_score(node_and_skeleton_data):
    node, adj_list = node_and_skeleton_data
    K = len(adj_list.get(node, []))
    one_hop_neighbors = set(adj_list.get(node, []))
    two_hop_neighbors = set()
    for neighbor in one_hop_neighbors:
        two_hop_neighbors.update(adj_list.get(neighbor, []))
    two_hop_neighbors.discard(node)
    two_hop_neighbors.difference_update(one_hop_neighbors)
    D = len(two_hop_neighbors)
    N_val = D + K
    u = K / N_val if N_val != 0 else 0
    C = K + u * D
    return node, C


def calculate_node_scores(skeleton):
    adj_list = {node: list(skeleton.neighbors(node)) for node in skeleton.nodes()}
    tasks = [(node, adj_list) for node in skeleton.nodes()]
    node_scores = {}
    for task in tasks:
        node, C = calculate_single_node_score(task)
        node_scores[node] = C
    return node_scores


def structural_ranking(skeleton, start_node):
    node_scores = calculate_node_scores(skeleton)
    traversed_nodes_set = {start_node}
    traversed_nodes_list = [start_node]
    candidate_nodes = []
    for neighbor in skeleton.neighbors(start_node):
        if neighbor not in traversed_nodes_set:
            heapq.heappush(candidate_nodes, (-node_scores.get(neighbor, 0.0), neighbor))

    while candidate_nodes:
        neg_score, current_node = heapq.heappop(candidate_nodes)
        if current_node not in traversed_nodes_set:
            traversed_nodes_set.add(current_node)
            traversed_nodes_list.append(current_node)
            for neighbor in skeleton.neighbors(current_node):
                if neighbor not in traversed_nodes_set:
                    heapq.heappush(candidate_nodes, (-node_scores.get(neighbor, 0.0), neighbor))
    return traversed_nodes_list


def order_allocation_change(skeleton, representative):
    decision_list = structural_ranking(skeleton, representative)
    for i in range(len(decision_list)):
        skeleton.nodes[decision_list[i]]['ranking'] = i
    return skeleton, decision_list