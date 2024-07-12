import torch
from abc import ABC, abstractmethod
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from .terminal_colors import tcols
from sklearn.metrics import roc_auc_score
from torchinfo import summary

class GuidedClassifier(torch.nn.Module, ABC):
    def __init__(self,device="cpu", hpars={}):
        #super().__init__(device, hpars)
        torch.nn.Module.__init__(self)
        ABC.__init__(self)

        self.hp_guided_classifier = {
            "lr": 0.001,
            "early_stopping": 25,
            "num_node_features": 13,
            "class_weight":0.7
        }

        self.hp_guided_classifier.update((k, hpars[k]) for k in self.hp_guided_classifier.keys() & hpars.keys())

        self.device = device
        self.lr_guided_classifier = self.hp_guided_classifier["lr"]

        self.num_node_features_guided_classifier = self.hp_guided_classifier["num_node_features"]

        self.recon_loss_weight = 1 - self.hp_guided_classifier["class_weight"]
        self.class_loss_weight = self.hp_guided_classifier["class_weight"]

        self.recon_loss_function = torch.nn.MSELoss()
        self.class_loss_function = torch.nn.CrossEntropyLoss()

        self.all_train_loss = []
        self.all_train_recon_loss = []
        self.all_train_class_loss = []
        self.all_valid_loss = []
        self.all_valid_recon_loss= []
        self.all_valid_class_loss = []

        self.all_train_acc = []
        self.all_valid_acc = []
        self.all_train_roc_auc = []
        self.all_valid_roc_auc = []

        self.best_valid_loss = float("inf")
        self.early_stopping_limit = self.hp_guided_classifier["early_stopping"]
        self.epochs_no_improve = 0

    @abstractmethod
    def instantiate_encoder(self):
        pass

    @abstractmethod
    def instantiate_decoder(self):
        pass
    
    @abstractmethod
    def encoder_decoder(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def gae_network_summary(self):
        pass

    @abstractmethod
    def classifier(self, *args, **kwargs):
        pass

    @abstractmethod
    def classifier_network_summary(self):
        pass

    def network_summary(self):
        self.gae_network_summary()
        self.classifier_network_summary()

    def forward(self, data):
        """
        Forward pass through the model
        @data :: torch_geometric.data object
        """
        z, latent_x, latent_edge, latent_edge_weight, batch = self.encoder_decoder(data)
        class_output = self.classifier(latent_x, latent_edge, latent_edge_weight, batch)
        return z, latent_x, latent_edge, latent_edge_weight, batch, class_output

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
        print("Training the " + self.hp_gae["gae_type"] + " GAE model with the " + self.hp_classifier["classifier_type"] + " classifier.")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()
            train_losses, train_acc, train_roc_auc = self.train_all_batches(train_loader)
            valid_losses, valid_acc, valid_roc_auc = self.valid(valid_loader, outdir)
            self.all_train_loss.append(train_losses[0].item())
            self.all_train_recon_loss.append(train_losses[1].item())
            self.all_train_class_loss.append(train_losses[2].item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_valid_recon_loss.append(valid_losses[1].item())
            self.all_valid_class_loss.append(valid_losses[2].item())
            self.all_train_acc.append(train_acc)
            self.all_valid_acc.append(valid_acc)
            self.all_train_roc_auc.append(train_roc_auc)
            self.all_valid_roc_auc.append(valid_roc_auc)
            self.print_metrics(epoch, epochs, train_losses, valid_losses, train_acc, valid_acc, train_roc_auc, valid_roc_auc)
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
        batch_loss_sum = 0
        batch_recon_loss_sum = 0
        batch_class_loss_sum = 0
        batch_acc_sum = 0
        batch_roc_auc_sum = 0
        num_batches = 0
        for batch_data in train_loader:
            batch_loss, batch_recon_loss, batch_class_loss, batch_acc, batch_roc_auc = self.train_batch(batch_data)
            batch_loss_sum += batch_loss
            batch_acc_sum += batch_acc
            batch_roc_auc_sum += batch_roc_auc
            batch_recon_loss_sum += batch_recon_loss
            batch_class_loss_sum += batch_class_loss
            num_batches += 1
        mean_batch_loss = batch_loss_sum/num_batches
        mean_batch_recon_loss = batch_recon_loss_sum/num_batches
        mean_batch_class_loss = batch_class_loss_sum/num_batches
        mean_batch_acc = batch_acc_sum/num_batches
        mean_batch_roc_auc = batch_roc_auc_sum/num_batches

        return (mean_batch_loss, mean_batch_recon_loss, mean_batch_class_loss), mean_batch_acc, mean_batch_roc_auc

    def train_batch(self, data_batch):
        """
        Train the model on a batch and evaluate the different kinds of losses.
        Propagate this backwards for minimum train_loss.
        @data_batch :: Pytorch batch object with the data, including target values

        returns :: Pytorch loss object of the training loss.
        """
        loss, recon_loss, class_loss, class_output = self.compute_loss(data_batch)
        acc = self.compute_accuracy(data_batch, class_output)
        roc_auc = self.compute_roc_auc(data_batch, class_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, recon_loss, class_loss, acc, roc_auc

    def compute_loss(self, data):
        """
        Compute the loss of a forward pass through the ae and
        classifier. Combine the two losses and return the one loss.
        @data :: Pytorch object with the data, including target values

        returns :: Float of the computed combined loss function value.
        """
        data = data.to(self.device)
        z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)

        recon_loss = self.recon_loss_function(z, data.x)
        class_loss = self.class_loss_function(class_output, data.y.long())
        

        return (self.recon_loss_weight * recon_loss + self.class_loss_weight*class_loss, recon_loss, class_loss, class_output)

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
        data = data.to(self.device)
        self.eval()
        z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)

        class_loss = self.class_loss_function(class_output, data.y.long())
        recon_loss = self.recon_loss_function(z, data.x)

        valid_loss = (
            self.recon_loss_weight * recon_loss + self.class_loss_weight * class_loss
        )

        valid_acc = self.compute_accuracy(data, class_output)
        valid_roc_auc = self.compute_roc_auc(data, class_output)

        self.save_best_loss_model(valid_loss, outdir)

        return (valid_loss, recon_loss, class_loss), valid_acc, valid_roc_auc
    
    @staticmethod
    def print_metrics(epoch, epochs, train_losses, valid_losses, train_acc, valid_acc, train_roc_auc, valid_roc_auc):
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
            f"Train loss (avg) = {train_losses[0].item():.8f}, "
            f"Train accuracy (avg) = {train_acc:.4f}, "
            f"Train ROC AUC (avg) = {train_roc_auc:.4f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Train recon loss (no weight) (avg) = {train_losses[1].item():.8f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Train class loss (no weight) (avg) = {train_losses[2].item():.8f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid loss = {valid_losses[0].item():.8f}, "
            f"Valid accuracy = {valid_acc:.4f}, "
            f"Valid ROC AUC = {valid_roc_auc:.4f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid recon loss (no weight) = {valid_losses[1].item():.8f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid class loss (no weight) = {valid_losses[2].item():.8f}"
        )
        print("\n")

    @torch.no_grad()
    def predict(self, data):
        """
        Compute the prediction of the model.
        @data :: Input array to pass through the model.

        returns :: The latent space of the ae and the recon data, and the classification output.
        """
        data = data.to(self.device)
        self.eval()
        reconstructed_z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)

        reconstructed_z = reconstructed_z.cpu()
        latent_x = latent_x.cpu()
        latent_edge_index = latent_edge.cpu()
        latent_edge_weight = edge_weight.cpu()
        b = b.cpu()
        class_output = class_output.cpu()

        return reconstructed_z, latent_x, latent_edge_index, latent_edge_weight, b, class_output

    def save_best_loss_model(self, valid_loss, outdir):
        """
        Prints a message and saves the optimized model with the best loss.
        @valid_loss :: Float of the validation loss.
        @outdir     :: Directory where the best model is saved.
        """

        if self.best_valid_loss > valid_loss:
            self.epochs_no_improve = 0
            self.best_valid_loss = valid_loss
            print(tcols.OKGREEN + f"New min: {self.best_valid_loss:.2e}" + tcols.ENDC)
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
        file_path = os.path.join(outdir, "hyperparameters_gae.json")
        params_file = open(file_path, "w")
        json.dump(self.hp_gae, params_file)
        params_file.close()

        file_path = os.path.join(outdir, "hyperparameters_classifier.json")
        params_file = open(file_path, "w")
        json.dump(self.hp_classifier, params_file)
        params_file.close()

        file_path = os.path.join(outdir, "hyperparameters.json")
        params_file = open(file_path, "w")
        json.dump(self.hp_guided_classifier, params_file)
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

