import csv
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

from a_Graph_Initialization import graph_initialization
from b_Structural_Ranking import order_allocation_change
from c_Interaction_And_Propagation import neighborhood_initialization
from c_Interaction_And_Propagation import influence_model_propagation
from d_Uncertainty_And_Diffusion import neighborhood_learning_diffusion_with_uncertainty
from d_Uncertainty_And_Diffusion import k_nearest_neighbors_cal
from d_Uncertainty_And_Diffusion import precompute_nn_indices


np.random.seed(0)


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