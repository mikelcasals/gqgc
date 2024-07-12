#Hybrid VQC attached to the MIAGAE autoencoder

from autoencoders.SAG_model_classifier import SAG_model_classifier
import pennylane as qml
import torch
import torch.nn as nn
from vqc_guided import variational_forms as vf
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F
from pennylane import numpy as np
from sklearn.metrics import roc_auc_score
from .util import convert_to_adj_matrix, transform_data_for_classifier, transform_data_for_classifier_definitive
from torch.autograd import Variable
import torch.optim as optim
# Diagnosis tools. Enable in case you need. Disabled to increase performance.
torch.autograd.set_detect_anomaly(True)
torch.autograd.profiler.profile(enabled=True)

class SAG_model_VQC(SAG_model_classifier):
    """
    Main skeleton for having a VQC classifier attached to
    latent space of the MIAGAE autoencoder
        @qdevice :: String containing what kind of device to run the
                    quantum circuit on: simulation, or actual computer?
        @device  ::
        @hpars   :: Dictionary of the hyperparameters to configure the vqc.
    """

    def __init__(self, qdevice, device, hpars):
        super().__init__(device, hpars)

        new_hp = {
            "hybrid": True,
            "nqubits": 7,
            "n_features": 1,
            "vform":"perm_equivariant_embedding",
            "n_layers": 2
        }

        self.hp.update(new_hp)
        self.hp.update((k, hpars[k]) for k in self.hp.keys() & hpars.keys())
        self.nlayers = self.hp["n_layers"]

        self.n_features = self.hp["n_features"]

        self.alphas = nn.Parameter(torch.randn(self.nlayers, self.n_features), requires_grad=True)
        self.betas = nn.Parameter(torch.randn(self.nlayers), requires_grad=True)
        
        print(self.alphas)
        print(self.betas)

        #self.gammas = Variable(0.1*torch.randn(self.nlayers), requires_grad=True)
        #self.betas = Variable(0.1*torch.randn(self.nlayers), requires_grad=True)
        #self.gammas = 0.1 * np.random.randn(self.nlayers, requires_grad=True)
        #self.betas = 0.1 * np.random.randn(self.nlayers, requires_grad=True)
        #self.gammas = torch.randn(self.nlayers)
        #self.betas = torch.randn(self.nlayers)

        self.qdevice = qdevice
        
        self.diff_method = self.select_diff_method(hpars)
        self.epochs_no_improve = 0
        self.circuit = qml.qnode(self.qdevice, diff_method = self.diff_method, interface="torch")(self.qcircuit)
        #self.circuit = qml.qnode(self.qdevice, diff_method = self.diff_method, interface="torch")(self.qcircuit_definitive)

        #self.classifier = qml.qnn.TorchLayer(self.circuit, weight_shapes = {"betas": (self.nlayers,), "gammas": (self.nlayers,)})
        
        del self.class_loss_function
        self.class_loss_function = self.shifted_bce

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {total_params}')

    def instantiate_adam_optimizer(self):
        """
        Instantiate the optimizer object, used in the training of the autoencoder.
        """
        #self.optimizer = optim.Adam([*self.parameters(), self.betas, self.gammas], lr=self.hp["lr"])
        self.optimizer = optim.Adam(self.parameters(), lr=self.hp["lr"])
        #self.optimizer = optim.Adam([self.betas, self.gammas], lr=self.hp["lr"])

    @staticmethod
    def select_diff_method(hpars):
        """Checks if a differentiation method for the quantum circuit is specified
        by the user. If not, 'best' is selected as the differentiation method.

        Args:
            args: Arguments given to the vqc by the user, specifiying various hps.

        Returns:
            String that specifies which differentiation method to use.
        """
        if "diff_method" in hpars:
            return hpars["diff_method"]

        return "best"


    
    def qcircuit(self, x, edge_index, edge_weight, alphas, betas):
        """
        Circuit that uses the permutation equivariant embedding
        """
        #vf.perm_equivariant_embedding_definitive(x,edge_index, edge_weight, alphas, betas)
        vf.equivariant_ansatz(x, edge_index, edge_weight, alphas, betas)
        observable = qml.PauliZ(0)
        #for i in range(1, len(x)):  #medir primer qubit
        #    if i%2 == 0:
        #        observable @= qml.PauliX(i)
        return qml.expval(observable)


    def shifted_bce(self, x, y):
        """
        Shift the input given to this method and calculate the binary cross entropy
        loss. This shift is required to have the output of the VQC model in [0,1].
        Args:
            x (torch.tensor): Data point/batch to evaluate the loss on.
            y (torch.tensor): Corresponding labels of the point/batch.
        Returns:
            The binary cross entropy loss computed on the given data.
        """
        #x = torch.tensor(x)
        x = (x+1)/2
        x = x.float()
        y = y.float()
        return nn.BCELoss(reduction="mean")(x.cpu(), y.cpu())
    
    def forward(self, data):
        """
        Forward pass through the autoencoder and classifier
        """

        x, edge_index, y, batch, edge_weight = data.x, data.edge_index, data.y, data.batch, data.edge_attr
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, num_nodes=x.shape[0])#,fill_value=0)
        edge_weight = edge_weight.squeeze()

        edge_list = []
        perm_list = []
        shape_list = []

        #Encoder
        f, e, b = x, edge_index, batch
        for i in range(self.depth):
            if i < self.depth:
                edge_list.append(e)
            f = self.down_list[i](f, e)
            shape_list.append(f.shape)
            f = F.leaky_relu(f)
            f, e, edge_weight, b, perm, _ = self.pool_list[i](f, e, edge_weight, batch=b)
            perm_list.append(perm)
        
        latent_x, latent_edge, latent_batch, latent_edge_weight = f, e, b, edge_weight

        #Classifier
        #quantum_data = transform_data_for_classifier(latent_x, latent_edge, latent_batch, latent_edge_weight, data.y)

        latent_x, latent_edge, latent_edge_weight = transform_data_for_classifier_definitive(latent_x,latent_edge, latent_batch, latent_edge_weight)

        #class_output = [self.circuit((quantum_data[0][i], quantum_data[1][i]), self.betas, self.gammas).item() for i in range(len(quantum_data[0]))]
        #class_output = [self.classifier(quantum_data[i]) for i in range(len(data.y))]
        #class_output = [self.classifier(latent_x[i]) for i in range(len(data.y))]
        #class_output = [self.circuit(latent_x[i], self.betas, self.gammas).item() for i in range(len(data.y))]
        class_output_list = []



        for i in range(len(data.y)):
            #class_output_list.append(self.classifier(quantum_data[i]))
            class_output_list.append(self.circuit(latent_x[i], latent_edge[i], latent_edge_weight[i], self.alphas, self.betas))
            #class_output_list.append(self.classifier(latent_x[i]))
        class_output = torch.stack(class_output_list)
        
        #Decoder
        z = f
        for i in range(self.depth):
            index = self.depth - i - 1
            shape = shape_list[index]
            up = torch.zeros(shape).to(self.device)
            p = perm_list[index]
            up[p] = z
            z = self.up_list[i](up, edge_list[index])
            if i < self.depth - 1:
                z = torch.relu(z)

        edge_list.clear()
        perm_list.clear()
        shape_list.clear()

        return z, latent_x, latent_edge, edge_weight, b, class_output
    
    def compute_loss(self, data, alphas=None, betas=None):
        """
        Objective function to be passed through the optimiser.
        Weights is taken as an argument here since the optimiser func needs it.
        We then use the class self variable inside the method.
        """
        if not alphas is None:
            self.alphas = alphas
        if not betas is None:
            self.betas = betas
        data = data.to(self.device)
        z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)
        
        #return self.class_loss_function(predictions, data[2])

        class_loss = self.class_loss_function(class_output, data.y.long())
        recon_loss = self.recon_loss_function(z, data.x)

        return self.recon_loss_weight * recon_loss + self.class_loss_weight*class_loss
    
    def compute_accuracy(self,preds, labels):
        """
        Compute the accuracy of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed accuracy value.
        """
        #preds = torch.argmax(class_output, dim=1)
        #correct = (preds==data.y).sum().item()
        #acc = correct/data.y.size(0)
        #labels = np.array(labels, requires_grad=False)
        shifted_preds = (preds + 1)/2
        rounded_preds = torch.round(shifted_preds).long()
        #shifted_preds = (np.array(preds, requires_grad=False) + 1) / 2
        #rounded_preds = np.round(shifted_preds).astype(int)

        acc = torch.mean((rounded_preds==labels).float())
        return acc
    
    def compute_roc_auc(self, preds, labels):
        """
        Compute the roc auc score of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed roc auc score.
        """
        #labels = np.array(labels, requires_grad=False)
        #shifted_preds = (np.array(preds, requires_grad=False) + 1) / 2
        shifted_preds = (preds + 1) / 2
        shifted_preds = shifted_preds.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        roc_auc = roc_auc_score(labels, shifted_preds)
        return roc_auc
    
    def compute_loss_acc_rocauc(self, data, alphas=None, betas=None):
        """
        Objective function to be passed through the optimiser.
        Weights is taken as an argument here since the optimiser func needs it.
        We then use the class self variable inside the method.
        """
        #if not weights is None:
        #    self._weights = weights
        #if not betas is None:
        #    self.betas = betas
        #if not gammas is None:
        #    self.gammas = gammas
        if not betas is None:
            self.betas = betas
        if not alphas is None:
            self.alphas = alphas
        #data = data.to(self.device)
        z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)
        #loss = self.class_loss_function(predictions, data[2])

        class_loss = self.class_loss_function(class_output, data.y.long())
        recon_loss = self.recon_loss_function(z, data.x)
        loss = self.recon_loss_weight * recon_loss + self.class_loss_weight*class_loss
        
        acc = self.compute_accuracy(class_output.cpu(), data.y.cpu())
        rocauc = self.compute_roc_auc(class_output.cpu(), data.y.cpu())
        #acc = 0
        #rocauc = 0

        return loss, recon_loss, class_loss, acc, rocauc
    
    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        """
        Evaluate the validation loss for the model and save the model if a
        new minimum is found.
        @valid_loader :: Pytorch data loader with the validation data.
        @outdir       :: Output folder where to save the model.

        returns :: Pytorch loss object of the validation loss.
        """
        data = next(iter(valid_loader))
        data = data.to(self.device)
        self.eval()
        valid_loss, recon_loss, class_loss, valid_acc, valid_roc_auc = self.compute_loss_acc_rocauc(data)

        self.save_best_loss_model(valid_loss, outdir)


        return (valid_loss, recon_loss, class_loss), valid_acc, valid_roc_auc
    

    def train_batch(self, data):
        """
        Train on one batch.
        """
        #x_batch = np.array(x_batch[:, :], requires_grad=False)
        #node_features_batch = [x['node_features'] for x in x_batch]
        #A_batch = [x['A'] for x in x_batch]
        #num_nodes_batch = [x['num_nodes'] for x in x_batch]
        #node_features_batch = np_normal.array(node_features[:])#, requires_grad=False)
        #A_batch = np_normal.array(A[:])#, requires_grad=False)
        #num_nodes_batch = np_normal.array(num_nodes[:])#, requires_grad=False)
        #x_batch = np.array(x_batch[:], requires_grad=False)
        #y_batch = np.array(y_batch[:], requires_grad=False)
        #_, betas, gammas = self.optimizer.step(
        #    self.compute_loss, data, self.betas, self.gammas
        #)
        #self.betas = betas
        #self.gammas = gammas

        #print(self.betas)
        #print(self.gammas)

        data = data.to(self.device)

        loss,recon_loss,class_loss, acc, rocauc = self.compute_loss_acc_rocauc(data, self.alphas, self.betas)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #for name, param in self.named_parameters():
        #    if param.requires_grad:
        #        print(f"Gradients for {name}: {param.grad}")

        #print(self.betas.grad)
        #print(self.gammas.grad)

        #print("Parameters being optimized:")
        #for param_group in self.optimizer.param_groups:
        #    for param in param_group['params']:
        #        print(param)

        #print("END OF PRINT")
        #print(self.betas.grad)
        #print(self.gammas.grad)
        #print(self.betas)

        #print(self.gammas)
        
        return loss, recon_loss, class_loss, acc, rocauc
    

    #def train_batch(self, data_batch) -> float:
        """
        Train the model on a batch and evaluate the different kinds of losses.
        Propagate this backwards for minimum train_loss.
        @data_batch :: Pytorch batch object with the data, including target values

        returns :: Pytorch loss object of the training loss.
        """

    #    loss, _, _, acc, roc_auc = self.compute_loss_acc_rocauc(data_batch)

    #    self.optimizer.zero_grad()
    #    loss.backward()
    #    self.optimizer.step()


    #    return loss, acc, roc_auc
    
    

    
        