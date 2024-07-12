import pennylane as qml

def perm_equivariant_embedding(x, edge_index, edge_weight, betas, gammas):

    n_nodes = x.shape[0]
    n_layers = len(betas) # infer the number of layers from the parameters
    # initialise in the plus state
    for i in range(n_nodes):
        qml.Hadamard(i)

    node_features_mean = x.mean(axis=1)

    for l in range(n_layers):
        for i in range(edge_index.shape[1]):
            # assuming undirected graph
            if edge_index[0,i] < edge_index[1,i]:
                qml.IsingZZ(2*gammas[l]*edge_weight[i], wires=[edge_index[0,i].item(),edge_index[1,i].item()])

        for i in range(n_nodes):
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

def equivariant_ansatz_v2(x, edge_index, edge_weight, alphas, betas):
    n_nodes = x.shape[0]
    n_features = x.shape[1]
    n_layers = alphas.shape[0]

    n_qubits = n_nodes*((n_features+1)//2)
    qubits_per_node = (n_features+1)//2

    node_to_first_qubit_mapping = {}
    qubit_index = 0
    for node_index in range(n_nodes):
        node_to_first_qubit_mapping[node_index] = qubit_index
        qubit_index += qubits_per_node

    
    for layer in range(n_layers):
        qubit_index = 0
        feature_count = 0
        for i in range(n_nodes):
            for j in range(n_features):
                qml.RX(x[i,j], wires=qubit_index)
                qml.RX(alphas[layer, j], wires=qubit_index)
                feature_count += 1
                if feature_count == 2:
                    feature_count = 0
                    qubit_index+=1


        for i in range(edge_index.shape[1]):
                if edge_index[0,i] < edge_index[1,i] and edge_weight[i] != 0:
                    qml.IsingZZ(edge_weight[i] + betas[layer,0], wires=[node_to_first_qubit_mapping[edge_index[0,i].item()],node_to_first_qubit_mapping[edge_index[1,i].item()]])


        for i in range(n_nodes):
            first_qubit_of_node = node_to_first_qubit_mapping[i]
            for j in range(first_qubit_of_node+1,first_qubit_of_node+qubits_per_node):
                qml.CRY(betas[layer,j-first_qubit_of_node], wires=[j, first_qubit_of_node])
            
    
def equivariant_ansatz_v3(x, edge_index, edge_weight, alphas, betas):

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

    last_qubit_index = len(x)
    for i in range(n_nodes):
        qml.IsingZZ(betas[n_layers], wires=[i, last_qubit_index])

def equivariant_ansatz_v4(x, edge_index, edge_weight, alphas, betas,gammas):

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

    last_qubit_index = len(x)
    for i in range(n_nodes):
        qml.IsingZZ(betas[n_layers], wires=[i, last_qubit_index])
    
    qml.RY(gammas[0], wires=last_qubit_index)