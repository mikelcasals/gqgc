import torch
from abc import ABC, abstractmethod
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from .terminal_colors import tcols
from torchinfo import summary


class GAE(torch.nn.Module, ABC):
    
    def __init__(self, device="cpu", hpars={}):
        if not hpars["guided"]:
            super().__init__()

        self.hp_gae = {
            "lr": 0.001,
            "early_stopping": 25,
            "guided": False
        }
        self.hp_gae.update((k, hpars[k]) for k in self.hp_gae.keys() & hpars.keys())


        self.device = device

        self.lr_gae = self.hp_gae["lr"]
        self.gae_type = "base_model"

        self.recon_loss_function = torch.nn.MSELoss(reduction="mean")

        self.all_train_recon_loss = []
        self.all_valid_recon_loss = []

        self.best_valid_recon_loss = float("inf")
        self.early_stopping_limit = self.hp_gae["early_stopping"]
        self.epochs_no_improve = 0

    @abstractmethod
    def instantiate_encoder(self):
        pass

    @abstractmethod
    def instantiate_decoder(self):
        pass
    
    @abstractmethod
    def encoder_decoder(self):
        pass
    
    @abstractmethod
    def gae_network_summary(self):
        pass
        
    def network_summary(self):
        self.gae_network_summary()

    def forward(self, data):
        """
        Forward pass through the autoencoder
        @data :: torch_geometric.data object
        """
        z, latent_x, latent_edge, latent_edge_weight, batch = self.encoder_decoder(data)
        return z, latent_x, latent_edge, latent_edge_weight, batch

   
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
        print("Training the " + self.gae_type + " GAE model...")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()
            recon_train_loss = self.train_all_batches(train_loader)
            recon_valid_loss = self.valid(valid_loader, outdir)
            self.all_train_recon_loss.append(recon_train_loss.item())
            self.all_valid_recon_loss.append(recon_valid_loss.item())
            self.print_metrics(epoch, epochs, recon_train_loss, recon_valid_loss)
            if self.early_stopping():
                break
    
    def instantiate_adam_optimizer(self):
        """
        Instantiate the optimizer object, used in the training of the autoencoder.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr_gae)
    
    @staticmethod
    def print_summary(model):
        """
        Prints a neat summary of a given ae model, with all the layers.
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
        Train the model on all the batches.
        @train_loader :: Pytorch DataLoader object with the training data.

        returns :: The normalized training loss averaged over all the 
            batches in an epoch.
        """
        batch_loss_sum = 0
        num_batches = 0
        for batch_data in train_loader:
            batch_loss = self.train_batch(batch_data)
            batch_loss_sum += batch_loss
            num_batches += 1
        mean_batch_loss = batch_loss_sum / num_batches

        return mean_batch_loss

    def train_batch(self, data_batch):
        """
        Train the model on a batch and evaluate the loss.
        Propagate this backwards for minimum train_loss.
        @data_batch :: Pytorch batch object with the data, including target values

        returns :: Pytorch loss object of the training loss.
        """

        loss = self.compute_loss(data_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def compute_loss(self, data):
        """
        Compute the loss of a forward pass through the autoencoder
        
        @data :: Pytorch object with the data, including target values

        returns :: Float of the computed loss function value
        """

        data = data.to(self.device)
        z, _, _, _, _ =self.forward(data)
        loss = self.recon_loss_function(z, data.x)

        return loss
    
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
        loss = self.compute_loss(data)
        self.save_best_loss_model(loss, outdir)

        return loss
    
    def save_best_loss_model(self, valid_loss, outdir):
        """
        Prints a message and saves the optimized model with the best loss.
        @valid_loss :: Float of the validation loss.
        @outdir     :: Directory where the best model is saved.
        """

        if self.best_valid_recon_loss > valid_loss:
            self.epochs_no_improve = 0
            self.best_valid_recon_loss = valid_loss
            print(tcols.OKGREEN + f"New min: {self.best_valid_recon_loss:.2e}" + tcols.ENDC)
            if outdir is not None:
                torch.save(self.state_dict(), outdir + "best_model.pt")
        else:
            self.epochs_no_improve += 1

    def print_metrics(self, epoch, epochs, train_loss, valid_loss):
        """
        Prints the training and validation metrics in a nice format.
        @epoch      :: Int of the current epoch.
        @epochs     :: Int of the total number of epochs.
        @train_loss :: The computed training loss pytorch object.
        @valid_loss :: The computed validation loss pytorch object.
        """
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Train loss (average) = {train_loss.item():.8f}"
        )
        print(f"Epoch : {epoch + 1}/{epochs}, " f"Valid loss = {valid_loss.item():.8f}")
        print("\n")

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
        json.dump(self.hp_gae, params_file)
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

    @torch.no_grad()
    def predict(self, data):
        """
        Compute the prediction of the model.
        @data :: Input array to pass through the model.

        returns :: The latent space of the ae and the recon data.
        """
        data = data.to(self.device)
        self.eval()
        reconstructed_z, latent_x, latent_edge_index, latent_edge_weight, b = self.forward(data)

        reconstructed_z = reconstructed_z.cpu()
        latent_x = latent_x.cpu()
        latent_edge_index = latent_edge_index.cpu()
        latent_edge_weight = latent_edge_weight.cpu()
        b = b.cpu()

        return reconstructed_z, latent_x, latent_edge_index, latent_edge_weight, b
    
    def loss_plot(self, outdir):
        """
        Plots the loss for each epoch for the training and validation data.
        @outdir :: Directory where to save the loss plot.
        """
        epochs = list(range(len(self.all_train_recon_loss)))
        plt.plot(
            epochs,
            self.all_train_recon_loss,
            color="gray",
            label="Training Loss (average)",
        )
        plt.plot(epochs, self.all_valid_recon_loss, color="navy", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.text(
            np.min(epochs),
            np.max(self.all_train_recon_loss),
            f"Min: {self.best_valid_recon_loss:.2e}",
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