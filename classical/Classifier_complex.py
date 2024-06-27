from torch_geometric.nn import global_mean_pool, GraphConv, SAGEConv
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import add_remaining_self_loops

from .terminal_colors import tcols


class GNN(torch.nn.Module):

    def __init__(self, device="cpu", hpars={}):
        super().__init__()

        self._hp = {
            "input_size": 1,
            "hidden": 64,
            "num_classes": 2,
            "dropout": 0.5,
            "lr": 0.002,
            "adam_betas": (0.9, 0.999),
            "early_stopping": 20
        }

        self._device = device
        self._class_loss_function = nn.BCELoss()
        #self._class_loss_function = nn.CrossEntropyLoss()
        self._hp.update((k, hpars[k]) for k in self._hp.keys() & hpars.keys())

        self.best_valid_loss = float("inf")
        self.all_train_loss = []
        self.all_valid_loss = []
        self.all_train_acc = []
        self.all_valid_acc = []
        self.all_train_roc_auc = []
        self.all_valid_roc_auc = []

        self.early_stopping_limit = self._hp["early_stopping"]
        self.epochs_no_improve = 0

        self.training = False
        self.dropout =self._hp["dropout"]
        input_size = self._hp["input_size"]
        hidden = self._hp["hidden"]
        num_classes = self._hp["num_classes"]
        self.conv1 = GraphConv(input_size, input_size,aggr='mean')
        self.conv2 = GraphConv(input_size, input_size,aggr='mean')
        self.conv3 = GraphConv(input_size, input_size,aggr='mean')

        self.lin1 = nn.Linear(input_size, input_size)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.lin2 = nn.Linear(input_size, input_size)
        self.bn2 = nn.BatchNorm1d(hidden // 2)
        self.lin3 = nn.Linear(hidden // 2, hidden // 4)
        self.bn3 = nn.BatchNorm1d(hidden // 4)
        self.lin4 = nn.Linear(hidden // 4, 1)

        self.conv1 = GraphConv(input_size, 2,aggr='mean')

        self.hidden_size = 1

        self.conv1 = GraphConv(input_size, self.hidden_size, aggr='mean')
        #self.conv2 = GraphConv(2, 4,aggr='mean')
        #self.conv3 = GraphConv(4, 8,aggr='mean')
        #self.fc = torch.nn.Linear(8, num_classes)
        #self.fc = torch.nn.Linear(input_size, num_classes)
        self.fc = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        #x1 = self.conv1(x, edge_index, edge_weight)
        #x1 = self.conv2(x1, edge_index, edge_weight)
        #x1 = self.conv3(x1, edge_index, edge_weight)
        #c = global_mean_pool(x1, batch)
        #c = F.dropout(c, p=self.dropout, training=self.training)
        #c = self.lin1(c)
        #b = self.bn1(c)
        #b = torch.tanh(b)
        #b = F.dropout(b, p=self.dropout, training=self.training)
        #c = self.lin2(b)
        #b = self.bn2(c)
        #b = torch.tanh(b)
        #b = F.dropout(b, p=self.dropout, training=self.training)
        #c = self.lin3(b)
        #b = self.bn3(c)
        #b = torch.tanh(b)
        #c = self.lin4(b)
        #c = torch.sigmoid(c)

        x = self.conv1(x, edge_index,edge_weight)
        x = F.relu(x)
        #x = self.conv2(x, edge_index,edge_weight)
        #x = F.relu(x)
        #x = self.conv3(x, edge_index,edge_weight)
        #x = F.relu(x)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        # Final classifier
        x = self.fc(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x = torch.sigmoid(x)
        c=x

        return c

    def instantiate_adam_optimizer(self):
        """Instantiate the optimizer object, used in the training of the model."""
        self = self.to(self._device)
        self.optimizer = optim.Adam(
            self.parameters(), lr=self._hp["lr"], betas=self._hp["adam_betas"]
        )

    def compute_loss(self, data):
        """
        Compute the loss of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed loss function value.
        """
        data = data.to(self._device)
        output = self.forward(data.x, data.edge_index, data.edge_attr, data.batch)
        #output = output.squeeze()
        loss = self._class_loss_function(output.flatten(), data.y.float())
        return loss
    
    def compute_accuracy(self,data):
        """
        Compute the accuracy of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed accuracy value.
        """
        data = data.to(self._device)
        output = self.forward(data.x, data.edge_index, data.edge_attr, data.batch)
        #preds = torch.argmax(output, dim=1)
        preds = torch.round(output).squeeze().int()
        #correct = (preds==data.y).sum().item()
        correct = torch.eq(preds, data.y).sum().item()
        acc = correct/data.y.size(0)
        return acc
    
    def compute_roc_auc(self, data):
        """
        Compute the roc auc score of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed roc auc score.
        """
        data = data.to(self._device)
        output = self.forward(data.x, data.edge_index, data.edge_attr, data.batch)
        output = output.detach().cpu().numpy()
        y_true = data.y.cpu().numpy()
        roc_auc = roc_auc_score(y_true, output)
        return roc_auc
    

    @staticmethod
    def print_metrics(epoch, epochs, train_loss, valid_loss, train_acc, valid_acc, train_roc_auc, valid_roc_auc):
        """
        Prints the training and validation losses in a nice format.

        Args:
            epoch: Current epoch.
            epochs: Total number of epochs.
            train_loss: The computed training loss pytorch object.
            valid_loss: The computed validation loss pytorch object.
            train_acc: The computed training accuracy.
            valid_acc: The computed validation accuracy.
            train_roc_auc: The computed training roc_auc score.
            valid_roc_auc: The computed validation roc_auc score.
        """
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Train loss (average) = {train_loss.item():.8f}, "
            f"Train accuracy = {train_acc:.4f}, "
            f"Train ROC AUC = {train_roc_auc:.4f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid loss = {valid_loss.item():.8f}, "
            f"Valid accuracy = {valid_acc:.4f}, "
            f"Valid ROC AUC = {valid_roc_auc:.4f}"
        )
    

    @staticmethod
    def print_losses(epoch, epochs: int, train_loss: torch.Tensor, valid_loss: torch.Tensor):
        """
        Prints the training and validation losses in a nice format.

        Args:
            epoch: Current epoch.
            epochs: Total number of epochs.
            train_loss: The computed training loss pytorch object.
            valid_loss: The computed validation loss pytorch object.
        """
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Train loss (average) = {train_loss.item():.8f}"
        )
        print(f"Epoch : {epoch + 1}/{epochs}, " f"Valid loss = {valid_loss.item():.8f}")

    @staticmethod
    def print_summary(torch_object):
        """
        Prints a neat summary the model with all the layers and activations.
        Can be the architecture of the network or the optimizer object.
        """
        try:
            summary(
                torch_object,
                show_input=True,
                show_hierarchical=False,
                print_summary=True,
                max_depth=1,
                show_parent_layers=False,
            )
        except Exception as e:
            print(e)
            print(tcols.WARNING + "Net summary failed!" + tcols.ENDC)

    def optimizer_summary(self):
        """
        Prints a summary of the optimizer that is used in the training.
        """
        print(tcols.OKGREEN + "Optimizer summary:" + tcols.ENDC)
        print(self.optimizer)
        print("\n\n")
    
    def save_best_loss_model(self, valid_loss: float, outdir: str) -> int:
        """
        Prints a message and saves the optimised model with the best loss.

        Args:
            valid_loss: Float of the validation loss.
            outdir: Directory where the best model is saved.
        """
        if self.best_valid_loss > valid_loss:
            self.epochs_no_improve = 0
            self.best_valid_loss = valid_loss

            print(tcols.OKGREEN + f"New min: {self.best_valid_loss:.2e}" + tcols.ENDC)
            if outdir is not None:
                torch.save(self.state_dict(), outdir + "best_model.pt")
        else:
            self.epochs_no_improve += 1
    
    def _early_stopping(self, early_stopping_limit) -> bool:
        """
        Stops the training if there has been no improvement in the loss
        function during the past, e.g. 10, number of epochs.

        Returns: True for when the early stopping limit was exceeded and
                 false otherwise.
        """
        if self.epochs_no_improve >= early_stopping_limit:
            return 1
        return 0

    @torch.no_grad()
    def valid(self, valid_loader, outdir) -> float:
        """
        Evaluate the validation loss for the model and save the model if a
        new minimum is found.
        @valid_loader :: Pytorch data loader with the validation data.
        @outdir       :: Output folder where to save the model.

        returns :: Pytorch loss object of the validation loss.
        """
        batch_loss_sum = 0
        batch_acc_sum = 0
        batch_roc_auc_sum = 0
        nb_of_batches = 0
        for data in valid_loader:
            data = data.to(self._device)
            self.eval()
            batch_loss = self.compute_loss(data)
            batch_acc = self.compute_accuracy(data)
            batch_roc_auc = self.compute_roc_auc(data)
            batch_loss_sum += batch_loss
            batch_acc_sum += batch_acc
            batch_roc_auc_sum += batch_roc_auc
            nb_of_batches += 1
        
        loss = batch_loss_sum / nb_of_batches
        acc = batch_acc_sum / nb_of_batches
        roc_auc = batch_roc_auc_sum / nb_of_batches
        self.save_best_loss_model(loss, outdir)

        return loss, acc, roc_auc

    def train_batch(self, data_batch) -> float:
        """
        Train the model on a batch and evaluate the different kinds of losses.
        Propagate this backwards for minimum train_loss.
        @data_batch :: Pytorch batch object with the data, including target values

        returns :: Pytorch loss object of the training loss.
        """

        loss = self.compute_loss(data_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = self.compute_accuracy(data_batch)
        roc_auc = self.compute_roc_auc(data_batch)

        return loss, acc, roc_auc

    def train_all_batches(self, train_loader) -> float:
        """
        Train the autoencoder on all the batches.
        @train_loader :: Pytorch loader object with the training data.

        returns :: The normalised training loss averaged over all the
            batches in an epoch.
        """
        batch_loss_sum = 0
        batch_acc_sum = 0
        batch_roc_auc_sum = 0
        nb_of_batches = 0
        for batch_data in train_loader:
            batch_loss, batch_acc, batch_roc_auc = self.train_batch(batch_data)
            batch_loss_sum += batch_loss
            batch_acc_sum += batch_acc
            batch_roc_auc_sum += batch_roc_auc
            nb_of_batches += 1

        return batch_loss_sum / nb_of_batches, batch_acc_sum / nb_of_batches, batch_roc_auc_sum / nb_of_batches
    
    def train_model(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        outdir: str,
    ):
        """
        Train the graph neural network.

        Args:
            train_loader: Pytorch data loader with the training data.
            valid_loader: Pytorch data loader with the validation data.
            epochs: The number of epochs to train for.
            outdir: The output dir where to save the training results.
        """
        self.instantiate_adam_optimizer()
        self.optimizer_summary()
        print(tcols.OKCYAN)
        print("Training the GNN...")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()
            train_loss, train_acc, train_roc_auc = self.train_all_batches(train_loader)
            valid_loss, valid_acc, valid_roc_auc = self.valid(valid_loader, outdir)
            if self._early_stopping(self.early_stopping_limit):
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_loss.item())
            self.all_train_acc.append(train_acc)
            self.all_valid_acc.append(valid_acc)
            self.all_train_roc_auc.append(train_roc_auc)
            self.all_valid_roc_auc.append(valid_roc_auc)
            #self.print_losses(epoch, epochs, train_loss, valid_loss)
            self.print_metrics(epoch, epochs, train_loss, valid_loss, train_acc, valid_acc, train_roc_auc, valid_roc_auc)
        
    def loss_plot(self, outdir: str):
        """
        Plots the loss for each epoch for the training and validation data.

        Args:
            outdir: Directory where to save the loss plot.
        """
        epochs = list(range(len(self.all_train_loss)))
        plt.plot(
            epochs,
            self.all_train_loss,
            color="gray",
            label="Training Loss (average)",
        )
        plt.plot(epochs, self.all_valid_loss, color="navy", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.text(
            np.min(epochs),
            np.max(self.all_train_loss),
            f"Min: {self.best_valid_loss:.2e}",
            verticalalignment="top",
            horizontalalignment="left",
            color="blue",
            fontsize=15,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )

        plt.legend()
        plt.savefig(outdir + "loss_epochs.pdf")
        plt.close()

        print(tcols.OKGREEN + f"Loss vs epochs plot saved to {outdir}." + tcols.ENDC)

    
    def export_architecture(self, outdir: str):
        """
        Saves the structure of the NN to a file.

        Args:
            outdir: Directory where to save the architecture of the network.
        """
        with open(outdir + "model_architecture.txt", "w") as model_arch:
            print(self, file=model_arch)

    def export_hyperparameters(self, outdir: str):
        """
        Saves the hyperparameters of the model to a json file.

        Args:
            outdir: Directory where to save the json file.
        """
        file_path = os.path.join(outdir, "hyperparameters.json")
        params_file = open(file_path, "w")
        json.dump(self._hp, params_file)
        params_file.close()
    
    def load_model(self, model_path):
        """
        Loads the weights of a trained model saved in a .pt file.

        Args:
            model_path: Directory where a trained model was saved.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("∄ path.")
        self.load_state_dict(
            torch.load(model_path, map_location=torch.device(self._device))
        )
    
    @torch.no_grad()
    def predict(self, data) -> np.ndarray:
        """
        Compute the prediction of the autoencoder.
        @x_data :: Input array to pass through the autoencoder.

        returns :: The latent space of the ae and the reco data.
        """
        data = data.to(self.device)
        self.eval()
        out = self.forward(data.x, data.edge_index, data.edge_attr, data.batch)

        out = out.cpu()

        return out