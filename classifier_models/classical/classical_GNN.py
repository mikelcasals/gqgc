import torch
from torch_geometric.nn import GraphConv, global_mean_pool
from base_models.classifier_base_model import Classifier
from .terminal_colors import tcols

class ClassicalGNN(Classifier):
    def __init__(self, device="cpu", hpars={}):
        Classifier.__init__(self, device, hpars)
        
        self.hp_GNN = {
            "classifier_type": "ClassicalGNN",
            "num_features": 13,
            "hidden_size": 2
        }

        self.hp_GNN.update((k, hpars[k]) for k in self.hp_GNN.keys() & hpars.keys())
        self.hp_classifier.update((k, self.hp_GNN[k]) for k in self.hp_GNN.keys())

        self.classifier_type = self.hp_classifier["classifier_type"]

        self.conv1 = GraphConv(self.hp_classifier["num_features"], self.hp_classifier["hidden_size"], aggr='mean')
        self.fc = torch.nn.Linear(self.hp_classifier["hidden_size"], 2)

    def classifier(self, x, edge_index, edge_weight, batch):
        """
        Forward pass through the classifier
        @latent_x :: torch tensor
        @latent_edge :: torch tensor
        @latent_edge_weight :: torch tensor
        @batch :: torch tensor
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        class_output = torch.sigmoid(x)
        return class_output
    
    def classifier_network_summary(self):
        print(tcols.OKGREEN + "Classifier summary:" + tcols.ENDC)
        self.print_summary(self.conv1)
        self.print_summary(self.fc)