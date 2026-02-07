import numpy as np
from scipy.spatial.distance import euclidean


np.random.seed(0)


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