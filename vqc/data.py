
import torch
import torch_geometric
import numpy as np
import scipy.sparse as sp



class QuantumGraphsLoader:

    def __init__(self, original_dataLoader):
        self.original_dataLoader = original_dataLoader

    def __iter__(self):

        for data in self.original_dataLoader:

            num_graphs = data.y.size(0)
            batch = data.batch.numpy()

            batch_node_features = []
            batch_adj_matrices = []
            batch_labels = []

            for graph_idx in range(num_graphs):
                #Get nodes belongign to the current graph
                node_mask = (batch==graph_idx)
                node_indices = np.where(node_mask)[0]
                node_features = data.x[node_mask].numpy()
                num_nodes = node_features.shape[0]

                if num_nodes > 1:
                    # Create a mapping from global node indices to local node indices
                    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}

                    #Get edges belonging to the current graph
                    edge_mask = (batch[data.edge_index[0]]==graph_idx)
                    edge_index = data.edge_index[:,edge_mask].numpy()
                    edge_weight = data.edge_attr[edge_mask].numpy() if data.edge_attr is not None else np.ones(edge_index.shape[1])
                    edge_weight = edge_weight.flatten()

                    # Reindex edge indices to start from 0 for this graph's nodes
                    edge_index = np.vectorize(global_to_local.get)(edge_index)

                    num_nodes = node_features.shape[0]
                    adj_matrix = self.convert_to_adj_matrix(edge_index, edge_weight, num_nodes)
                
                else:
                    adj_matrix = np.zeros((1,1))
                
                label = data.y[graph_idx].item()

                batch_node_features.append(node_features)
                batch_adj_matrices.append(adj_matrix)
                batch_labels.append(label)

            yield batch_node_features, batch_adj_matrices, batch_labels

    def convert_to_adj_matrix(self, edge_index, edge_weight, num_nodes):
        """Convert edge indices and weights to a dense adjacency matrix."""
        adj_matrix = sp.coo_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        return adj_matrix.toarray()
            