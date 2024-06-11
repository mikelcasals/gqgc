from pennylane import numpy as np
import scipy.sparse as sp
import torch


def convert_to_adj_matrix(edge_index, edge_weight, num_nodes):
        """Convert edge indices and weights to a dense adjacency matrix."""
        adj_matrix = sp.coo_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        return adj_matrix.toarray()

def transform_data_for_classifier_old_old(x, edge_index, batch, edge_attr, y):
    x, edge_index, edge_attr, batch, y = x.detach().cpu(), edge_index.detach().cpu(), edge_attr.detach().cpu(), batch.detach().cpu(), y.detach().cpu()
    num_graphs = batch[-1] + 1
    batch = batch.numpy()

    batch_node_features = []
    batch_adj_matrices = []
    batch_labels = []

    for graph_idx in range(num_graphs):
        #Get nodes belongign to the current graph
        node_mask = (batch==graph_idx)
        node_indices = np.where(node_mask)[0]
        node_features = x[node_mask].numpy()
        num_nodes = node_features.shape[0]

        if num_nodes > 1:
            # Create a mapping from global node indices to local node indices
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}

            #Get edges belonging to the current graph
            edge_mask = (batch[edge_index[0]]==graph_idx)
            edges = edge_index[:,edge_mask]
            edge_weight = edge_attr[edge_mask].numpy() if edge_attr is not None else np.ones(edges.shape[1])
            edge_weight = edge_weight.flatten()

            # Reindex edge indices to start from 0 for this graph's nodes
            edges = np.vectorize(global_to_local.get)(edges)

            num_nodes = node_features.shape[0]
            adj_matrix = convert_to_adj_matrix(edges, edge_weight, num_nodes)
        
        else:
            adj_matrix = np.zeros((1,1))
        
        label = y[graph_idx].item()


        batch_node_features.append(node_features)
        batch_adj_matrices.append(adj_matrix)
        batch_labels.append(label)

    return batch_node_features, batch_adj_matrices, batch_labels

def transform_data_for_classifier(x, edge_index, batch, edge_attr, y):
    num_graphs = batch.max().item() + 1
    batch_node_features = []
    batch_adj_matrices = []
    batch_labels = []
    batch_data = []

    for graph_idx in range(num_graphs):
        # Get nodes belonging to the current graph
        node_mask = (batch==graph_idx)
        node_indices = torch.where(node_mask)[0].tolist()
        node_features = x[node_mask]
        num_nodes = node_features.size(0)

        if num_nodes > 1:
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_indices)}
            #Get edges belonging to the current graph
            edge_mask = (batch[edge_index[0]]==graph_idx)
            edges = edge_index[:,edge_mask]
            edges = torch.tensor([global_to_local[edge.item()] for edge in edges.flatten()]).view(2, -1)
            edge_weight = edge_attr[edge_mask] if edge_attr is not None else torch.ones(edges.size(1))
            # Create a sparse adjacency matrix
            adj_matrix = torch.sparse_coo_tensor(edges, edge_weight, size=(num_nodes, num_nodes)).to_dense()
        else:
            adj_matrix = torch.zeros((1,1), dtype=torch.float)

        label = y[graph_idx]

        batch_node_features.append(node_features)
        batch_adj_matrices.append(adj_matrix)
        batch_labels.append(label)

        data = [num_nodes]
        data.extend(node_features.flatten().tolist())
        data.extend(adj_matrix.flatten().tolist())
        batch_data.append(torch.tensor(data, dtype=torch.float))
    #return batch_node_features, batch_adj_matrices, batch_labels
    return batch_data

def transform_data_for_classifier_definitive(x, edge_index, batch, edge_attr):

    num_graphs = batch.max().item()+1

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
        edge_weights = edge_attr[edge_mask]
        min_id = torch.min(edges)
        edges -= min_id

        batch_x.append(node_features)
        batch_edge_index.append(edges)
        batch_edge_weight.append(edge_weights)

    return batch_x, batch_edge_index, batch_edge_weight


def choose_ae_vqc_model(ae_vqc_type, qdevice, device, hyperparams) -> callable:
    from .miagae_vqc_hybrid import MIAGAE_VQC
    from .sag_model_vqc_hybrid import SAG_model_VQC
    """
    Picks and loads one of the implemented autoencoder model classes.
    @ae_type     :: String of the type of autoencoder that you want to load.
    @device      :: String of the device to load it on: 'cpu' or 'gpu'.
    @hyperparams :: Dictionary of the hyperparameters to load with.

    returns :: The loaded autoencoder model with the given hyperparams.
    """
    switcher = {
        "MIAGAE_vqc": lambda: MIAGAE_VQC(qdevice=qdevice, device=device, hpars=hyperparams).to(device),
        "SAG_model_vqc": lambda: SAG_model_VQC(qdevice=qdevice, device=device, hpars=hyperparams).to(device)
    }
    model = switcher.get(ae_vqc_type, lambda: None)()
    if model is None:
        raise TypeError("Specified AE type does not exist!")

    return model
