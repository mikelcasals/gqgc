import torch
import pennylane as qml
import matplotlib.pyplot as plt
import time

import numpy as np

start_time = time.time()
rng = np.random.default_rng(4324234)



class QGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()


node_features = torch.tensor([[ -5.2680,  -2.1468, -11.6577],
        [ -5.6969,  -2.3273, -12.2832],
        [ -5.6077,  -2.1682, -12.1081],
        [ -5.4387,  -1.6261, -12.3302],
        [ -4.7608,  -1.8691, -11.2132],
        [ -4.6511,  -1.9594, -10.9839],
        [ -7.7769,  -4.9261, -11.7698]])

node_features = node_features/(-12.3302)

edge_index = torch.tensor([[6, 6, 6, 6, 6, 6, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5],
                            [0, 2, 4, 3, 1, 5, 6, 2, 4, 3, 1, 5, 6, 0, 4, 3, 1, 5, 6, 0, 2, 3, 1, 5, 6, 0, 2, 4, 1, 5, 6, 0, 2, 4, 3, 5, 6, 0, 2, 4, 3, 1]])

edge_weight = torch.tensor([1.0356, 1.0816, 1.1874, 1.1914, 1.3556, 1.3585, 1.0356, 0.1473, 0.1657,
        0.1751, 0.3226, 0.3231, 1.0816, 0.1473, 0.1338, 0.1295, 0.3404, 0.3151,
        1.1874, 0.1657, 0.1338, 0.0147, 0.2066, 0.1833, 1.1914, 0.1751, 0.1295,
        0.0147, 0.2127, 0.1856, 1.3556, 0.3226, 0.3404, 0.2066, 0.2127, 0.0583,
        1.3585, 0.3231, 0.3151, 0.1833, 0.1856, 0.0583])

edge_weight = edge_weight/1.3585


def get_adjacency_matrix(edge_index, edge_weight):
    # Number of nodes (assuming nodes are 0-indexed and continuous)
    num_nodes = edge_index.max() + 1

    # Initialize the adjacency matrix
    adjacency_matrix = torch.zeros((num_nodes, num_nodes))

    # Populate the adjacency matrix
    for i, (src, dest) in enumerate(edge_index.T):
        adjacency_matrix[src, dest] = edge_weight[i]
    
    return adjacency_matrix

def perm_equivariant_embedding(A,node_features, betas, gammas):
    """
    Ansatz to embedd a graph with node and edge weights into a quantum state.

    The adjacency matrix A contains the edge weights on the off-diagonal,
    as well as the node attributes on the diagonal.

    The embedding contains trainable weights 'betas' and 'gammas'.
    """
    n_nodes = len(A)
    n_layers = len(betas) # infer the number of layers from the parameters

    # initialise in the plus state
    for i in range(n_nodes):
        qml.Hadamard(i)

    for l in range(n_layers):

        for i in range(n_nodes):
            for j in range(i):
                    # factor of 2 due to definition of gate
                qml.IsingZZ(2*gammas[l]*A[i,j], wires=[i,j])
    #node_features_mean = np.mean(node_features, axis=1)
    node_features_mean = node_features.mean(dim=1)
    for i in range(n_nodes):
        qml.RX(node_features_mean[i]*betas[l], wires=i)

def create_observable(num_qubits):
    """Create a tensor product of PauliZ observables across all qubits."""
    observable = qml.PauliX(0)
    for i in range(1, num_qubits):
        observable @= qml.PauliX(i)
    return observable


n_qubits = 7
n_layers = 2

dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev)
def eqc(adjacency_matrix,node_features, observable, trainable_betas, trainable_gammas):
    """Circuit that uses the permutation equivariant embedding"""

    perm_equivariant_embedding(adjacency_matrix, node_features, trainable_betas, trainable_gammas)
    return qml.expval(observable)

#qlayer = qml.qnn.TorchLayer(eqc, weight_shapes=2)

A = get_adjacency_matrix(edge_index, edge_weight)
#betas = rng.random(n_layers)
#gammas = rng.random(n_layers)
betas = [0.012, 0.10]
gammas = [0.10, 0.10]
observable = create_observable(n_qubits)

qml.draw_mpl(eqc, decimals=2)(A, node_features, observable, betas, gammas)
plt.savefig("quantum_circuit.png")
#plt.show()

result_A = eqc(A, node_features, observable, betas, gammas)

print(result_A)

end_time = time.time()

elapsed_time = end_time - start_time

print(elapsed_time)