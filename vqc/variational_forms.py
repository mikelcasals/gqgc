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

    node_features_mean = np.array(node_features.mean(axis=1),requires_grad =False)
    A = np.array(A, requires_grad=False)

    for l in range(n_layers):
        for i in range(n_nodes):
            for j in range(i):
                # factor of 2 due to definition of gate
                qml.IsingZZ(2*gammas[l]*A[i,j], wires=[i,j])

        for i in range(n_nodes):
            #temp = np.array(node_features_mean[i]/10, requires_grad=False)
            qml.RX(betas[l]*node_features_mean[i], wires=i)
