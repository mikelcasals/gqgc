import torch
from abc import ABC, abstractmethod
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from .terminal_colors import tcols
from sklearn.metrics import roc_auc_score
from torchinfo import summary
import pennylane as qml
from . import util

class Classifier(torch.nn.Module, ABC):
    def __init__(self, device="cpu", hpars={}):
        if not hpars["guided"]:
            super().__init__()
        
        self.hp_classifier = {
            "lr": 0.001,
            "early_stopping": 25,
            "quantum": False,
            "ideal_dev": "lightning.qubit",
            "n_qubits": 10,
            "guided": False
        } 

        self.device = device
        self.hp_classifier.update((k, hpars[k]) for k in self.hp_classifier.keys() & hpars.keys())

        
        self.lr_classifier = self.hp_classifier["lr"]
        self.classifier_type = "base_model"

        if self.hp_classifier["quantum"]:
            self.qdevice = qml.device(self.hp_classifier["ideal_dev"], wires=self.hp_classifier["n_qubits"])
            self.class_loss_function = util.QuantumLossFunction()

        else:
            self.class_loss_function = torch.nn.CrossEntropyLoss()

        self.all_train_class_loss = []
        self.all_valid_class_loss = []

        self.all_train_acc = []
        self.all_valid_acc = []
        self.all_train_roc_auc = []
        self.all_valid_roc_auc = []

        self.best_valid_class_loss = float("inf")
        self.early_stopping_limit = self.hp_classifier["early_stopping"]
        self.epochs_no_improve = 0
    @abstractmethod
    def classifier(self):
        pass

    @abstractmethod
    def classifier_network_summary(self):
        pass
    
    def network_summary(self):
        self.classifier_network_summary()
    
    def forward(self, data):
        """
        Forward pass through the model
        @data :: torch_geometric.data object
        """
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        class_output = self.classifier(x, edge_index, edge_weight, batch)
        return class_output

    def train_model(self, train_loader, valid_loader, epochs, outdir):
        """
        Train the model
        @train_loader :: Pytorch DataLoader object containing the training data
        @valid_loader :: Pytorch DataLoader object containing the validation data
        @epochs :: Integer number of epochs to train the model
        @outdir :: String path to the output directory
        """
        self.instantiate_adam_optimizer()
        self.network_summary()
        self.optimizer_summary()

        print(tcols.OKCYAN)
        print("Training the " + self.classifier_type + " classifier.")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()
            class_train_loss, train_acc, train_roc_auc = self.train_all_batches(train_loader)
            class_valid_loss, valid_acc, valid_roc_auc = self.valid(valid_loader, outdir)


            self.all_train_class_loss.append(class_train_loss.item())
            self.all_valid_class_loss.append(class_valid_loss.item())
            self.all_train_acc.append(train_acc)
            self.all_valid_acc.append(valid_acc)
            self.all_train_roc_auc.append(train_roc_auc)
            self.all_valid_roc_auc.append(valid_roc_auc)
            self.print_metrics(epoch, epochs, class_train_loss, class_valid_loss, train_acc, valid_acc, train_roc_auc, valid_roc_auc)
            if self.early_stopping():
                break

    def instantiate_adam_optimizer(self):
        """
        Instantiate the optimizer object, used in the training of the autoencoder.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr_classifier)
    
    @staticmethod
    def print_summary(model):
        """
        Prints a neat summary of a given classifier model, with all the layers.
        @model :: Pytorch object of the model to be printed.
        """
        try:
            summary(
                model,
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

    def train_all_batches(self, train_loader):
        """
        Train the model for all batches.
        @train_loader :: Pytorch data loader with the training data.

        returns:: The normalised training loss over all the batches
        """
        batch_class_loss_sum = 0
        batch_acc_sum = 0
        batch_roc_auc_sum = 0
        num_batches = 0
        for batch_data in train_loader:
            batch_class_loss, batch_acc, batch_roc_auc = self.train_batch(batch_data)
            batch_class_loss_sum += batch_class_loss
            batch_acc_sum += batch_acc
            batch_roc_auc_sum += batch_roc_auc
            num_batches += 1
        mean_batch_class_loss = batch_class_loss_sum/num_batches
        mean_batch_acc = batch_acc_sum/num_batches
        mean_batch_roc_auc = batch_roc_auc_sum/num_batches

        return mean_batch_class_loss, mean_batch_acc, mean_batch_roc_auc

    def train_batch(self, data_batch):
        """
        Train the model on a batch and evaluate the different kinds of losses.
        Propagate this backwards for minimum train_loss.
        @data_batch :: Pytorch batch object with the data, including target values

        returns :: Pytorch loss object of the training loss.
        """
        class_loss, class_output = self.compute_loss(data_batch)
        acc = self.compute_accuracy(data_batch, class_output)
        roc_auc = self.compute_roc_auc(data_batch, class_output)
        self.optimizer.zero_grad()
        class_loss.backward()
        self.optimizer.step()

        return class_loss, acc, roc_auc

    def compute_loss(self, data):
        """
        Compute the loss of a forward pass through the classifier. 
        @data :: Pytorch object with the data, including target values

        returns :: Float of the computed loss function value and the class output.
        """
        data = data.to(self.device)
        class_output = self.forward(data)
        class_output = class_output.to(self.device)
        class_loss = self.class_loss_function(class_output, data.y.long())

        return class_loss, class_output

    def compute_accuracy(self, data, class_output):
        """
        Compute the accuracy of the classifier.
        @data :: Pytorch object with the data, including target values
        @class_output :: Pytorch tensor with the output of the classifier.

        returns :: Float of the computed accuracy value.
        """
        if class_output.dim() == 1:
            preds = torch.round(class_output).long()
        else:
            preds = torch.argmax(class_output, dim=1)
        correct = (preds==data.y).sum().item()
        acc = correct/data.y.size(0)
        return acc
    
    def compute_roc_auc(self, data, class_output):
        """
        Compute the roc auc score of the classifier
        @data :: Pytorch object with the data, including target values

        returns :: Float of the computed roc auc score.
        """
        class_output = class_output.detach().cpu().numpy()
        y_true = data.y.cpu().numpy()
        if class_output.ndim == 1:
            roc_auc = roc_auc_score(y_true, class_output)
        else:
            roc_auc = roc_auc_score(y_true, class_output[:,1])
        return roc_auc
    
    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        """
        Evaluate the validation loss, accuracy and roc-auc for the model and save the model if a
        new minimum loss is found.
        @valid_loader :: Pytorch data loader with the validation data.
        @outdir       :: Output folder where to save the model.

        returns :: Float values of the validation losses, accuracy and roc-auc.
        """
        data = next(iter(valid_loader))
        self.eval()

        class_valid_loss, class_output = self.compute_loss(data)
        valid_acc = self.compute_accuracy(data, class_output)
        valid_roc_auc = self.compute_roc_auc(data, class_output)

        self.save_best_loss_model(class_valid_loss, outdir)

        return class_valid_loss, valid_acc, valid_roc_auc
    
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
            f"Train loss (avg) = {train_loss:.8f}, "
            f"Train accuracy (avg) = {train_acc:.4f}, "
            f"Train ROC AUC (avg) = {train_roc_auc:.4f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid loss = {valid_loss:.8f}, "
            f"Valid accuracy = {valid_acc:.4f}, "
            f"Valid ROC AUC = {valid_roc_auc:.4f}"
        )
        print("\n")

    @torch.no_grad()
    def predict(self, data):
        """
        Compute the prediction of the model.
        @data :: Input array to pass through the model.

        returns :: The classification output.
        """
        data = data.to(self.device)
        self.eval()
        class_output = self.forward(data)
        class_output = class_output.cpu()

        return class_output

    def save_best_loss_model(self, valid_loss, outdir):
        """
        Prints a message and saves the optimized model with the best loss.
        @valid_loss :: Float of the validation loss.
        @outdir     :: Directory where the best model is saved.
        """

        if self.best_valid_class_loss > valid_loss:
            self.epochs_no_improve = 0
            self.best_valid_class_loss = valid_loss
            print(tcols.OKGREEN + f"New min: {self.best_valid_class_loss:.2e}" + tcols.ENDC)
            if outdir is not None:
                torch.save(self.state_dict(), outdir + "best_model.pt")
        else:
            self.epochs_no_improve += 1

    def early_stopping(self):
        """
        Stops the training if there has been no improvement in the loss
        function during the past predefined limit number of epochs.

        returns :: True for when the early stopping limit was exceeded
            and false otherwise.
        """

        if self.epochs_no_improve >= self.early_stopping_limit:
            return 1
        return 0
    
    def export_architecture(self, outdir):
        """
        Saves the structure of the model to a file.
        @outdir :: Directory where to save the architecture of the network.
        """
        with open(outdir + "model_architecture.txt", "w") as model_arch:
            print(self, file=model_arch)

    def export_hyperparameters(self, outdir):
        """
        Saves the hyperparameters of the model to a json file.
        @outdir :: Directory where to save the json file.
        """
        file_path = os.path.join(outdir, "hyperparameters.json")
        params_file = open(file_path, "w")
        json.dump(self.hp_classifier, params_file)
        params_file.close()

    
    def load_model(self, model_path):
        """
        Loads the weights of a trained model saved in a .pt file.
        @model_path :: Directory where a trained model was saved.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("∄ path.")
        self.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )

    def loss_plot(self, outdir):
        """
        Plots the loss for each epoch for the training and validation data.
        @outdir :: Directory where to save the loss plot.
        """

        epochs = list(range(len(self.all_train_class_loss)))
        plt.plot(
            epochs,
            self.all_train_class_loss,
            color="gray",
            label="Training Loss (average)",
        )
        plt.plot(epochs, self.all_valid_class_loss, color="navy", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.text(
            np.min(epochs),
            np.max(self.all_train_class_loss),
            f"Min: {self.best_valid_class_loss:.2e}",
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