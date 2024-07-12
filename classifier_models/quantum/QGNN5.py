import torch
from torch_geometric.nn import GraphConv, global_mean_pool
from base_models.classifier_base_model import Classifier
from .terminal_colors import tcols
import pennylane as qml
from pennylane import numpy as np
from . import util as quantum_util
from . import variational_forms as vf

class QGNN5(Classifier):
    def __init__(self, device="cpu", hpars={}):
        Classifier.__init__(self, device, hpars)
        
        self.hp_QGNN2 = {
            "classifier_type": "QGNN",
            "num_layers": 2,
            "num_features": 2,
            "ideal_dev": "default.qubit",
            "n_qubits": 10,
            "diff_method": "best"
        }

        self.hp_QGNN2.update((k, hpars[k]) for k in self.hp_QGNN2.keys() & hpars.keys())
        self.hp_classifier.update((k, self.hp_QGNN2[k]) for k in self.hp_QGNN2.keys())

        self.classifier_type = self.hp_classifier["classifier_type"]
        self.num_layers = self.hp_classifier["num_layers"]
        self.num_features = self.hp_classifier["num_features"]
        self.diff_method = self.hp_classifier["diff_method"]

        self.alphas = torch.nn.Parameter(0.01*torch.randn(self.num_layers, self.num_features), requires_grad=True)
        self.betas = torch.nn.Parameter(0.01*torch.randn(self.num_layers+1), requires_grad=True)

        
        #self.qdevice = qml.device(self.hp_QGNN1["ideal_dev"], wires=self.hp_QGNN1["n_qubits"])
        self.circuit = qml.qnode(self.qdevice, diff_method = self.diff_method, interface="torch")(self.quantum_circuit)
        
        
        print(self.alphas)
        print(self.betas)

    
    def quantum_circuit(self, x, edge_index, edge_weight, alphas, betas):
        """
        Circuit that uses the permutation equivariant embedding
        """
        vf.equivariant_ansatz(x,edge_index, edge_weight, alphas, betas)
        
        observable = qml.PauliZ(len(x)-1)
        return qml.expval(observable)

    def classifier(self, x, edge_index, edge_weight, batch):
        """
        Forward pass through the classifier
        @latent_x :: torch tensor
        @latent_edge :: torch tensor
        @latent_edge_weight :: torch tensor
        @batch :: torch tensor
        """
        x, edge_index, edge_weight = quantum_util.transform_data_for_guided_quantum_classifier(x, edge_index, edge_weight, batch)
    
        class_output = []
        for i in range(len(x)):
            class_output.append(self.circuit(x[i], edge_index[i], edge_weight[i], self.alphas, self.betas))
        class_output = torch.stack(class_output)
        class_output = (class_output+1)/2
        return class_output
    
    def classifier_network_summary(self):
        #print(tcols.OKGREEN + "Classifier summary:" + tcols.ENDC)
        #self.print_summary(self.conv1)
        #self.print_summary(self.fc)
        print("QGNN2 classifier summary:")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {total_params}')
