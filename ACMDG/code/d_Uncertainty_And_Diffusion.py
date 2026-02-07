import copy
import math
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import euclidean
import heapq

from c_Interaction_And_Propagation import influence_model_propagation
from c_Interaction_And_Propagation import  connections_cal
from c_Interaction_And_Propagation import  interaction_process

np.random.seed(0)

def k_nearest_neighbors_cal(data, k):
    neighbors = NearestNeighbors(n_neighbors=k).fit(data)
    k_nearest_neighbors = neighbors.kneighbors(data, return_distance=False)
    return k_nearest_neighbors


def uncertainty_cal(predict_labels, k_nearest_neighbors, candidates, k):
    uncertainty_dict = dict()
    for candidate in candidates:
        k_nearest_neighbor = k_nearest_neighbors[candidate]
        uncertainty = uncertainty_one_node(predict_labels, k_nearest_neighbor, k)
        uncertainty_dict[candidate] = uncertainty
    return uncertainty_dict


def uncertainty_one_node(predict_labels, k_nearest_neighbor, k):
    dict_labels = {}
    for i in range(len(k_nearest_neighbor)):
        point = k_nearest_neighbor[i]
        if predict_labels[point] not in dict_labels.keys():
            dict_labels[predict_labels[point]] = [point]
        else:
            dict_labels[predict_labels[point]].append(point)
    sum_entropy = 0
    for m in dict_labels.keys():
        proportion = len(dict_labels[m]) / k
        if proportion != 0:
            sum_entropy += proportion * math.log2(proportion)
    uncertainty = -sum_entropy
    if uncertainty < 1e-9:
        uncertainty = 0.0
    return uncertainty


def first_n_nodes_cal(candidates, n):
    if n > len(candidates):
        n = len(candidates)
    sliced_list = []
    heap = [(-value, key) for key, value in candidates.items()]
    heapq.heapify(heap)
    temp_candidates = copy.deepcopy(candidates)
    for _ in range(n):
        if not heap:
            break
        neg_value, key = heapq.heappop(heap)
        value = -neg_value
        if value < 1e-9:
            heapq.heappush(heap, (neg_value, key))
            break
        sliced_list.append(key)
        del temp_candidates[key]
    remaining_zero_uncertainty_keys = list(temp_candidates.keys())
    while heap:
        _, key = heapq.heappop(heap)
        remaining_zero_uncertainty_keys.append(key)
    return sliced_list, remaining_zero_uncertainty_keys


def precompute_nn_indices(data):
    neighbors_model = NearestNeighbors(n_neighbors=2).fit(data)
    nearest_neighbor_indices = neighbors_model.kneighbors(data, return_distance=False)[:, 1]
    nn_indices = {i: index for i, index in enumerate(nearest_neighbor_indices)}
    return nn_indices


def neighborhood_learning_diffusion_with_uncertainty(
        skeleton, data, predict_labels, neighborhood, neighborhood_r,
        neighborhood_r_behind, k_nearest_neighbors, count, order_remaining,
        real_labels, record, n, k, omega, nn_indices):
    iter_count = record[-1][0]['iter'] + 1
    n_samples = len(data)
    neighbors_nn_model = NearestNeighbors(n_neighbors=2).fit(data)
    nearest_neighbor_indices_for_interaction = neighbors_nn_model.kneighbors(data, return_distance=False)[:, 1]
    queried_nodes = set([node for sublist in neighborhood for node in sublist])
    all_candidates_pool_dict = {node: 0 for node in order_remaining}
    current_wave_nodes = queried_nodes.copy()

    while True:
        if not all_candidates_pool_dict:
            break

        new_nodes_to_query = set()
        for node in current_wave_nodes:
            nearest_neighbor = nn_indices.get(node)
            if nearest_neighbor is not None and nearest_neighbor not in queried_nodes:
                new_nodes_to_query.add(nearest_neighbor)

        uncertainty_dict = uncertainty_cal(predict_labels, k_nearest_neighbors, all_candidates_pool_dict.keys(), k)
        nodes_for_interaction = []
        remaining_zero_uncertainty_nodes = []

        nn_hit_nodes = new_nodes_to_query & all_candidates_pool_dict.keys()
        if nn_hit_nodes:
            nn_uncertainty_dict = {node: uncertainty_dict[node] for node in nn_hit_nodes}
            high_uncertainty_nn_nodes, zero_uncertainty_nn_nodes = first_n_nodes_cal(nn_uncertainty_dict, n)
            nodes_for_interaction.extend(high_uncertainty_nn_nodes)
            remaining_zero_uncertainty_nodes.extend(zero_uncertainty_nn_nodes)

        if not nodes_for_interaction or len(nodes_for_interaction) < n:
            global_uncertainty_dict = {
                node: score for node, score in uncertainty_dict.items()
                if node not in nodes_for_interaction
            }
            needed_count = n - len(nodes_for_interaction)
            high_uncertainty_global_nodes, zero_uncertainty_global_nodes = first_n_nodes_cal(
                global_uncertainty_dict, needed_count
            )
            nodes_for_interaction.extend(high_uncertainty_global_nodes)
            remaining_zero_uncertainty_nodes.extend(zero_uncertainty_global_nodes)

        needed_count = n - len(nodes_for_interaction)
        if needed_count > 0:
            zero_uncertainty_nodes_set = set(remaining_zero_uncertainty_nodes)
            zero_uncertainty_nodes_set.difference_update(set(nodes_for_interaction))
            ranking_candidates = []
            for node in zero_uncertainty_nodes_set:
                ranking = skeleton.nodes[node].get('ranking', float('inf'))
                ranking_candidates.append((ranking, node))
            ranking_candidates.sort(key=lambda x: x[0])
            nodes_for_interaction.extend([
                node for _, node in ranking_candidates[:needed_count]
            ])

        nodes_for_interaction = nodes_for_interaction[:n]
        if not nodes_for_interaction:
            break

        current_wave_nodes = set(nodes_for_interaction)
        for node in nodes_for_interaction:
            queried_nodes.add(node)
            if node in all_candidates_pool_dict:
                del all_candidates_pool_dict[node]

            nearest_neighbor_for_interaction = nearest_neighbor_indices_for_interaction[node]
            local_neighborhood_index = -1
            if nearest_neighbor_for_interaction in queried_nodes:
                for idx, nh in enumerate(neighborhood):
                    if nearest_neighbor_for_interaction in nh:
                        local_neighborhood_index = idx
                        break

            connections_modified = []
            if local_neighborhood_index >= 0:
                local_representative = neighborhood_r[local_neighborhood_index][0]
                distance_to_rep = euclidean(data[node], data[local_representative])
                connections_modified.append([node, local_representative, distance_to_rep, local_neighborhood_index])

            all_connections = connections_cal(data, node, neighborhood_r).tolist()
            other_connections = []
            for conn in all_connections:
                conn_index = int(conn[3])
                if local_neighborhood_index >= 0 and conn_index == local_neighborhood_index:
                    continue
                other_connections.append(conn)
            connections_modified.extend(other_connections)

            connections = np.array(connections_modified)
            neighborhood, neighborhood_r, neighborhood_r_behind, count = interaction_process(connections, real_labels,
                                                                                             neighborhood, count,
                                                                                             neighborhood_r,
                                                                                             neighborhood_r_behind,
                                                                                             skeleton, omega)

        predict_labels = influence_model_propagation(skeleton, neighborhood)
        ari = adjusted_rand_score(real_labels, predict_labels)
        nmi = normalized_mutual_info_score(real_labels, predict_labels)
        record.append([{"iter": iter_count, "interaction": count, "ari": ari, "nmi": nmi}])
        print("iteration: %d, queries: %d, ari: %f, nmi: %f" % (iter_count, count, ari, nmi))
        iter_count += 1
        if ari == 1.0 or len(queried_nodes) == n_samples:
            break
    return record
