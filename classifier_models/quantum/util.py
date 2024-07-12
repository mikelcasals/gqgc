import torch

def transform_data_for_guided_quantum_classifier(x, edge_index, edge_attr, batch):

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
        if edges.numel() > 0:
            min_id = torch.min(edges)
            edges -= min_id
        batch_x.append(node_features)
        batch_edge_index.append(edges)
        batch_edge_weight.append(edge_weights)

    return batch_x, batch_edge_index, batch_edge_weight

def quantum_loss_function(class_output, y):
    y = y.float()
    class_output = class_output.float()

    
    return torch.nn.BCELoss(reduction='mean')(class_output, y)


class QuantumLossFunction(torch.nn.Module):
    def __init__(self):
        super(QuantumLossFunction, self).__init__()
        # Initialize any parameters or sub-modules here

    def forward(self, predictions, targets):
        # Implement the loss computation here
        return quantum_loss_function(predictions, targets)


