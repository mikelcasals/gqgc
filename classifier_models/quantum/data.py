
import torch
import torch_geometric
import numpy as np
import scipy.sparse as sp
from torch_geometric.data.dataset import Dataset


class DataHelperClass(Dataset):
    def __init__(self, x, edge_index, edge_attr, batch, y):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.y = y
    def len(self):
        """Returns the number of data objects stored in the dataset."""
        raise len(self.y)
    
    def get(self, idx):
        """Returns the :obj:`i`-th data object."""
        return self.x[idx], self.edge_index[idx], self.edge_attr[idx], self.batch, self.y[idx]
class QuantumGraphsLoader:

    def __init__(self, original_dataLoader):
        self.original_dataLoader = original_dataLoader

    def __iter__(self):

        for data in self.original_dataLoader:
        
            x, edge_index, edge_attr, batch, y = data.x, data.edge_index, data.edge_attr, data.batch, data.y

            num_graphs = y.size(0)

            batch_x = []
            batch_edge_index = []
            batch_edge_weight = []

            for graph_idx in range(num_graphs):
                # Get nodes belonging to the current graph
                node_mask = (batch==graph_idx)
                node_indices = torch.where(node_mask)[0].tolist()
                node_features = x[node_mask]
                edge_mask = (batch[edge_index[0]]==graph_idx)
                edges = edge_index[:,edge_mask]
                print(edges)
                edge_weights = edge_attr[edge_mask]
                if edges.numel() > 0:
                    min_id = torch.min(edges)
                    edges -= min_id

                batch_x.append(node_features)
                batch_edge_index.append(edges)
                batch_edge_weight.append(edge_weights)

            yield DataHelperClass(batch_x, batch_edge_index, batch_edge_weight, None,y)
            