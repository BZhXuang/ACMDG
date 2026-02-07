import copy
import csv
import math
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.spatial.distance import euclidean
import heapq
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

np.random.seed(0)


################################################################################
## a_Graph_Initialization.py ##
################################################################################

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


################################################################################
## b_Structural_Ranking.py ##
################################################################################

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


################################################################################
## c_Interaction_And_Propagation.py ##
################################################################################

def neighborhood_initialization(data, decision_list, representative, real_labels, skeleton, m, omega):
    count = 0
    neighborhood = [[representative]]
    neighborhood_r = [[representative]]
    neighborhood_r_behind = [[representative]]
    nodes = decision_list[1:m]
    decision_list_remaining = decision_list[:]
    decision_list_remaining.remove(representative)
    for node in nodes:
        decision_list_remaining.remove(node)
        connections = connections_cal(data, node, neighborhood_r)
        neighborhood, neighborhood_r, neighborhood_r_behind, count = interaction_process(connections, real_labels,
                                                                                         neighborhood, count,
                                                                                         neighborhood_r,
                                                                                         neighborhood_r_behind,
                                                                                         skeleton, omega)
    return neighborhood, neighborhood_r, neighborhood_r_behind, count, decision_list_remaining


def connections_cal(data, node, neighborhood_r):
    connections = []
    for i in range(len(neighborhood_r)):
        distances = []
        for neighbor in neighborhood_r[i]:
            distances.append(euclidean(data[node], data[neighbor]))
        index = np.argmin(distances)
        connections.append([node, neighborhood_r[i][index], distances[index], i])
    connections = np.array(connections)
    sorted_indices = np.argsort(connections[:, 2])
    connections = connections[sorted_indices]
    return connections


def interaction_process(connections, real_labels, neighborhood, count, neighborhood_r, neighborhood_r_behind, skeleton,
                        l):
    flag = False
    node1 = int(connections[0][0])
    for i in range(len(connections)):
        node2 = int(connections[i][1])
        neighborhood_index = int(connections[i][3])
        if real_labels[node1] == real_labels[node2]:
            flag = True
            if len(neighborhood[neighborhood_index]) < l:
                neighborhood[neighborhood_index].append(node1)
                neighborhood_r[neighborhood_index].append(node1)
                if skeleton.nodes[node1]['ranking'] > skeleton.nodes[neighborhood_r_behind[neighborhood_index][0]][
                    'ranking']:
                    neighborhood_r_behind[neighborhood_index] = [node1]
            if len(neighborhood[neighborhood_index]) >= l:
                neighborhood[neighborhood_index].append(node1)
                if skeleton.nodes[node1]['ranking'] < skeleton.nodes[neighborhood_r_behind[neighborhood_index][0]][
                    'ranking']:
                    node_to_remove = neighborhood_r_behind[neighborhood_index][0]
                    neighborhood_r[neighborhood_index].remove(node_to_remove)
                    neighborhood_r[neighborhood_index].append(node1)
                    a = []
                    for j in neighborhood_r[neighborhood_index]:
                        a.append(skeleton.nodes[j]['ranking'])
                    c = neighborhood_r[neighborhood_index][np.argmax(a)]
                    neighborhood_r_behind[neighborhood_index] = [c]
            count = count + 1
            break
        if real_labels[node1] != real_labels[node2]:
            count = count + 1
    if not flag:
        neighborhood.append([node1])
        neighborhood_r.append([node1])
        neighborhood_r_behind.append([node1])
    return neighborhood, neighborhood_r, neighborhood_r_behind, count


def influence_model_propagation(skeleton, neighborhood):
    for i in range(len(skeleton.nodes)):
        skeleton.nodes[i]['state'] = 'inactive'
        skeleton.nodes[i]['label'] = 'unclear'
    nodes_in_neighborhood = []
    for i in range(len(neighborhood)):
        for node in neighborhood[i]:
            skeleton.nodes[node]['state'] = 'active'
            skeleton.nodes[node]['label'] = i
            nodes_in_neighborhood.append(node)
    for node in nodes_in_neighborhood:
        container = [node]
        label = skeleton.nodes[node]['label']
        while (True):
            if not container:
                break
            current_node = container.pop()
            neighbors = list(skeleton.neighbors(current_node))
            for neighbor in neighbors:
                if (skeleton.nodes[neighbor]['ranking'] > skeleton.nodes[node]['ranking']) and (
                        skeleton.nodes[neighbor]['state'] == 'inactive'):
                    container.append(neighbor)
                    skeleton.nodes[neighbor]['state'] = 'active'
                    skeleton.nodes[neighbor]['label'] = label
    predicted_labels = []
    for i in range(len(skeleton.nodes)):
        labels = skeleton.nodes[i]['label']
        predicted_labels.append(labels)
    return predicted_labels


################################################################################
## d_Uncertainty_And_Diffusion.py ##
################################################################################

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


################################################################################
## e_Main_And_Data_Process.py ##
################################################################################

def generate_data(path):
    df = pd.read_csv(path, header=None)
    data = np.array(df)
    col = len(data[0])
    label_col = col - 1
    labels = data[:, label_col]
    data = data[:, :label_col]
    data = np.array(data, dtype=float)
    unique_labels = np.unique(labels)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    labels = np.array([label_mapping.get(l, -1) for l in labels])
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, labels


def data_process(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data + random_matrix
    return data


def NNGAMD(data, real_labels, k, omega, beta):
    m = int(len(data) * (1 / beta))
    if m == 0: m = 1
    n = int(len(data) * (1 / beta))
    if n == 0: n = 1

    k_nearest_neighbors = k_nearest_neighbors_cal(data, k)
    nn_indices = precompute_nn_indices(data)

    skeleton, representative = graph_initialization(data)
    record = [[{"iter": 0, "interaction": 0, "ari": 0, "nmi": 0}]]

    skeleton, order = order_allocation_change(skeleton, representative)

    neighborhood, neighborhood_r, neighborhood_r_behind, count, order_remaining = neighborhood_initialization(data,
                                                                                                              order,
                                                                                                              representative,
                                                                                                              real_labels,
                                                                                                              skeleton,
                                                                                                              m, omega)
    predict_labels = influence_model_propagation(skeleton, neighborhood)
    record.append([{"iter": 1, "interaction": count, "ari": adjusted_rand_score(real_labels, predict_labels),
                    "nmi": normalized_mutual_info_score(real_labels, predict_labels)}])

    record_behind = neighborhood_learning_diffusion_with_uncertainty(
        skeleton, data, predict_labels, neighborhood, neighborhood_r,
        neighborhood_r_behind, k_nearest_neighbors, count, order_remaining,
        real_labels, record, n, k, omega, nn_indices)

    return record_behind


if __name__ == '__main__':
    k = 10
    beta = 100
    omega = 100
    path = '../dataset/arrhythmia/arrhythmia.csv'

    print(f"Loading data from: {path}")
    try:
        data, real_labels = generate_data(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    data = data_process(data)
    ARI_record = NNGAMD(data, real_labels, k, omega, beta)

    tlt = path.split('/')[-1].split('.')[0]
    output_dir = "../results"
    csv_title = f"{output_dir}/ACMDG-" + tlt + "-result.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    final_record = [item for sublist in ARI_record for item in sublist]
    with open(csv_title, 'w', newline='') as csvfile:
        fieldnames = ['Iteration', 'Query_Count', 'ARI_Value', 'NMI_Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for step in final_record:
            writer.writerow({
                'Iteration': step['iter'],
                'Query_Count': step['interaction'],
                'ARI_Value': step['ari'],
                'NMI_Value': step['nmi']
            })

    print(f"Process finished. Results saved to {csv_title}")