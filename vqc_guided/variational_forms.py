import pennylane as qml
from pennylane import numpy as np

def perm_equivariant_embedding(A,node_features, betas, gammas):
    """
    Ansatz to embedd a graph with node and edge weights into a quantum state.

    The adjacency matrix A contains the edge weights on the off-diagonal,
    as well as the node attributes on the diagonal.

    The embedding contains trainable weights 'betas' and 'gammas'.
    """
    
    n_nodes = len(A)
    n_layers = len(betas) # infer the number of layers from the parameters
    #A = A.numpy()
    # initialise in the plus state
    for i in range(n_nodes):
        qml.Hadamard(i)

    #node_features_mean = np.array(node_features.mean(axis=1),requires_grad =False)
    node_features_mean = node_features.mean(axis=1)
    #A = np.array(A, requires_grad=False)

    for l in range(n_layers):
        for i in range(n_nodes):
            for j in range(i):
                # factor of 2 due to definition of gate 
                qml.IsingZZ(2*gammas[l]*A[i,j], wires=[i,j])

        for i in range(n_nodes):
            #temp = np.array(node_features_mean[i]/10, requires_grad=False)
            qml.RX(betas[l]*node_features_mean[i], wires=i)


def perm_equivariant_embedding_definitive(x, edge_index, edge_weight, betas, gammas):

    n_nodes = x.shape[0]
    n_layers = len(betas) # infer the number of layers from the parameters
    # initialise in the plus state
    for i in range(n_nodes):
        qml.Hadamard(i)

    #node_features_mean = np.array(node_features.mean(axis=1),requires_grad =False)
    node_features_mean = x.mean(axis=1)
    #A = np.array(A, requires_grad=False)

    for l in range(n_layers):
        for i in range(edge_index.shape[1]):
            # assuming undirected graph
            if edge_index[0,i] < edge_index[1,i]:
                qml.IsingZZ(2*gammas[l]*edge_weight[i], wires=[edge_index[0,i].item(),edge_index[1,i].item()])

        for i in range(n_nodes):
            #temp = np.array(node_features_mean[i]/10, requires_grad=False)
            qml.RX(betas[l]*node_features_mean[i], wires=i)


def equivariant_ansatz(x, edge_index, edge_weight, alphas, betas):

    n_nodes = x.shape[0]
    n_features = x.shape[1]
    n_layers = alphas.shape[0]

    for layer in range(n_layers):
        for i in range(n_nodes):
            for j in range(n_features):
                qml.RX(x[i,j], wires=i)
                qml.RX(alphas[layer,j], wires=i)

        for i in range(edge_index.shape[1]):
            if edge_index[0,i] < edge_index[1,i] and edge_weight[i] != 0:
                qml.IsingZZ(edge_weight[i] + betas[layer], wires=[edge_index[0,i].item(),edge_index[1,i].item()])