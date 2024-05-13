import pennylane as qml
import torch
#import tensorflow as tf
#import numpy as np
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

from torch.utils.data import Dataset, DataLoader
import time


def get_adjacency_matrix(edge_index, edge_weight, num_nodes):

    # Initialize the adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    # Populate the adjacency matrix
    for i, (src, dest) in enumerate(edge_index.T):
        adjacency_matrix[src, dest] = edge_weight[i]
    
    return adjacency_matrix


def get_compressed_graphs(x, edge, edge_weight, batch):
    num_graphs = int(batch.max())+1
    accumulated_num_nodes = 0
    all_node_features = []
    all_adjacency_matrices = []
    for i in range(num_graphs):
        num_nodes = len(batch[batch==i])
        node_features = x[batch==i]

        edge_index_graph = edge[:, (accumulated_num_nodes <= edge[0,:]) & (edge[0,:] <= accumulated_num_nodes + num_nodes-1)]
        edge_index_graph -= accumulated_num_nodes
        edge_weight_graph = edge_weight[(accumulated_num_nodes <= edge[0,:]) & (edge[0,:] <= accumulated_num_nodes + num_nodes-1)]

        accumulated_num_nodes += num_nodes
        adjacency_matrix = get_adjacency_matrix(edge_index_graph, edge_weight_graph, num_nodes)

        all_node_features.append(node_features)
        all_adjacency_matrices.append(adjacency_matrix)

    return all_node_features, all_adjacency_matrices

def get_compressed_data(data_set, device, ae_model):
    all_node_features = []
    all_adjacency_matrices = []
    all_labels = []
    start_time = time.time()
    for data in data_set:
        data = data.to(device)
        z_train, x_train, edge_train, edge_weight_train, batch_train = ae_model(data)
        
        node_features, adjacency_matrices = get_compressed_graphs(x_train, edge_train, edge_weight_train, batch_train)
        labels = data.y

        node_features = [node_features_ind.detach().cpu().numpy() for node_features_ind in node_features]
        adjacency_matrices = [adjacency_matrix.detach().cpu().numpy() for adjacency_matrix in adjacency_matrices]
        labels = [label.cpu().item() for label in labels]
        
        all_node_features.extend(node_features)
        all_adjacency_matrices.extend(adjacency_matrices)
        all_labels.extend(labels)
    num_nodes = [len(graph) for graph in all_node_features]
    max_num_nodes = max(num_nodes)

    data = [{
    'node_features': all_node_features[i],
    'A': all_adjacency_matrices[i],
    'num_nodes': num_nodes[i]
    } for i in range(len(all_node_features))]
    

    return data, all_labels, max_num_nodes

def pad_data(data, max_num_nodes):
    for i in range(len(data)):
        data[i]['node_features']= np.array(np.pad(data[i]['node_features'], ((0, max_num_nodes - data[i]['num_nodes']), (0, 0)), mode='constant'), requires_grad=False)
        #data[i]['node_features'] = np.pad(data[i]['node_features'], ((0, max_num_nodes - len(data[i]['node_features'])), (0, 0)))
        data[i]['A'] = np.array(np.pad(data[i]['A'], ((0, max_num_nodes - data[i]['num_nodes']), (0, max_num_nodes - data[i]['num_nodes'])), mode='constant'), requires_grad=False)
        #data[i]['A'] = np.pad(data[i]['A'], ((0, max_num_nodes - len(data[i]['A'])), (0, max_num_nodes - len(data[i]['A'])), (0, 0)))
    return data
    

class QuantumDataset(Dataset):
    def __init__(self, event_data, labels):
        self.event_data = event_data
        self.labels = labels

    def __len__(self):
        return len(self.event_data)

    def __getitem__(self, idx):
        
        if isinstance(idx, slice):
            # If idx is a slice, return a subset of the dataset
            event_slice = self.event_data[idx]
            label_slice = self.labels[idx]
            node_features_list = [np.array(sample['node_features'], requires_grad=False) for sample in event_slice]
            A_list = [np.array(sample['A'], requires_grad=False) for sample in event_slice]
            num_nodes_list = [sample['num_nodes'] for sample in event_slice]
            label_list = label_slice
            return node_features_list, A_list, num_nodes_list, label_list
        else:
            # If idx is an integer, return a single sample
            node_features = self.event_data[idx]['node_features']
            A = self.event_data[idx]['A']
            num_nodes = self.event_data[idx]['num_nodes']
            label = self.labels[idx]
            return node_features, A, num_nodes, label
    
