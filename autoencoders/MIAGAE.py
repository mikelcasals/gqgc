# Vanilla MIAGAE autoencoder

import torch
import torch.nn as nn
from graphAE.utils.Layer import SGAT
from torch_geometric.nn import TopKPooling
from graphAE.utils.SAGEConv import SAGEConv
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from .terminal_colors import tcols
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from torch_geometric.utils import sort_edge_index
import shutil


class MIAGAE(nn.Module):

    def __init__(self, device="cpu", hpars={}):

        super().__init__()
        self.hp = {
            "ae_type": "vanilla MIAGAE",
            "input_size": 13,
            "kernels": 1,
            "depth":3,
            "rate":0.35,
            "shapes": "13,5,1",
            "lr": 0.001,
            "early_stopping": 100,
        }
        self.hp.update((k, hpars[k]) for k in self.hp.keys() & hpars.keys())

        self.device = device

        self.input_size = self.hp["input_size"]
        size = self.hp["kernels"]
        self.depth = self.hp["depth"]
        self.shapes = list(map(int, self.hp["shapes"].split(",")))[0:self.depth]
        self.rate = [self.hp["rate"]] * self.depth

        self.direction = 1

        self.down_list = torch.nn.ModuleList()
        self.up_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()

        #Encoder
        conv = SGAT(size, self.hp["input_size"], self.shapes[0])
        self.down_list.append(conv)
        for i in range(self.depth - 1):
            pool = TopKPooling(self.shapes[i], self.rate[i])
            self.pool_list.append(pool)
            conv = SGAT(size, self.shapes[i], self.shapes[i + 1])
            self.down_list.append(conv)
        pool = TopKPooling(self.shapes[-1], self.rate[-1])
        self.pool_list.append(pool)

        #Decoder
        for i in range(self.depth - 1):
            conv = SAGEConv(self.shapes[self.depth - i - 1], self.shapes[self.depth - i - 2])
            self.up_list.append(conv)
        conv = SAGEConv(self.shapes[0], self.input_size)
        self.up_list.append(conv)


        self.recon_loss_function = nn.MSELoss(reduction="mean")

        self.best_valid_loss = float('inf')
        self.all_train_loss = []
        self.all_valid_loss = []

        self.early_stopping_limit = self.hp["early_stopping"]
        self.epochs_no_improve = 0

    def forward(self, data):
        """
        Forward pass through the autoencoder
        """

        x, edge_index, y, batch, edge_weight = data.x, data.edge_index, data.y, data.batch, data.edge_attr
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, num_nodes=x.shape[0],fill_value=0)
        edge_weight = edge_weight.squeeze()

        edge_list = []
        perm_list = []
        shape_list = []

        #Encoder
        f, e, b = x, edge_index, batch
        for i in range(self.depth):
            if i < self.depth:
                edge_list.append(e)
            f, attn = self.down_list[i](f, e, self.direction)
            shape_list.append(f.shape)
            f = F.leaky_relu(f)

            f, e, edge_weight, b, perm, _ = self.pool_list[i](f, e, edge_weight, b, attn)

            perm_list.append(perm)
        latent_x, latent_edge = f, e

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

        return z, latent_x, latent_edge, edge_weight, b

    def instantiate_adam_optimizer(self):
        """
        Instantiate the optimizer object, used in the training of the autoencoder.
        """
        self.optimizer = optim.Adam(self.parameters(), lr=self.hp["lr"])

    def compute_loss(self, data):
        """
        Compute the loss of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed loss function value.
        """
        data = data.to(self.device)
        z, _, _, _, _ = self.forward(data)
        loss = self.recon_loss_function(z, data.x)
        return loss

    def print_losses(self, epoch, epochs, train_loss, valid_loss):
        """
        Prints the training and validation losses in a nice format.
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

    #def network_summary(self) -> NOT APPLICABLE

    def optimizer_summary(self):
        """
        Prints a summary of the optimizer that is used in the training.
        """
        print(tcols.OKGREEN + "Optimizer summary:" + tcols.ENDC)
        print(self.optimizer)
        print("\n\n")

    def save_best_loss_model(self, valid_loss, outdir):
        """
        Prints a message and saves the optimised model with the best loss.
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

    def early_stopping(self) -> bool:
        """
        Stops the training if there has been no improvement in the loss
        function during the past, e.g. 10, number of epochs.

        returns :: True for when the early stopping limit was exceeded
            and false otherwise.
        """
        if self.epochs_no_improve >= self.early_stopping_limit:
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
        nb_of_batches = 0
        for data in valid_loader:
            data = data.to(self.device)
            self.eval()
            batch_loss = self.compute_loss(data)
            batch_loss_sum += batch_loss
            nb_of_batches += 1
        
        loss = batch_loss_sum / nb_of_batches
        self.save_best_loss_model(loss, outdir)

        return loss

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

        return loss
    
    def train_all_batches(self, train_loader) -> float:
        """
        Train the autoencoder on all the batches.
        @train_loader :: Pytorch loader object with the training data.

        returns :: The normalised training loss averaged over all the
            batches in an epoch.
        """
        batch_loss_sum = 0
        nb_of_batches = 0
        for batch_data in train_loader:
            batch_loss = self.train_batch(batch_data)
            batch_loss_sum += batch_loss
            nb_of_batches += 1

        return batch_loss_sum / nb_of_batches
    
    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):
        """
        Train the vanilla autoencoder.
        @train_loader :: Pytorch data loader with the training data.
        @valid_loader :: Pytorch data loader with the validation data.
        @epochs       :: The number of epochs to train for.
        @outdir       :: The output dir where to save the training results.
        """
        self.instantiate_adam_optimizer()
        self.optimizer_summary()
        print(tcols.OKCYAN)
        print("Training the " + self.hp["ae_type"] + " AE model...")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()

            train_loss = self.train_all_batches(train_loader)
            valid_loss = self.valid(valid_loader, outdir)
            if self.early_stopping():
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_loss.item())
            
            self.print_losses(epoch, epochs, train_loss, valid_loss)

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
    
    def export_architecture(self, outdir):
        """
        Saves the structure of the nn to a file.
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
        json.dump(self.hp, params_file)
        params_file.close()
    
    def load_model(self, model_path):
        """
        Loads the weights of a trained model saved in a .pt file.
        @model_path :: Directory where a trained model was saved.
        """
        print(model_path)
        print(os.getcwd())
        if not os.path.exists(model_path):
            raise FileNotFoundError("∄ path.")
        self.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
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
        reconstructed_z, latent_x, latent_edge_index, latent_edge_weight, b = self.forward(data)

        reconstructed_z = reconstructed_z.cpu()
        latent_x = latent_x.cpu()
        latent_edge_index = latent_edge_index.cpu()
        latent_edge_weight = latent_edge_weight.cpu()
        b = b.cpu()

        return reconstructed_z, latent_x, latent_edge_index, latent_edge_weight, b
    
    def save_latent_space(self, data_loader, outdir, prefix = "train"):
        """
        Save the latent space of the autoencoder.
        @data_loader :: Pytorch data loader with the data.
        @outdir      :: Directory where to save the latent space.
        @prefix      :: Prefix for the saved folder
        """

        #latent_data/outdir/train
        full_data_path = outdir + "/" + prefix + "/" + prefix + "/raw/"  #format for graphAE

        if os.path.exists(outdir+ "/" + prefix + "/"):
            shutil.rmtree(outdir+ "/" + prefix + "/")

        if not os.path.exists(full_data_path):
            os.makedirs(full_data_path, exist_ok=True)


        latent_space = []

        node_attributes = []
        A = []
        edge_attributes = []
        graph_indicator = []
        graph_labels = []

        latent_edge_index_all = torch.empty(2,0, dtype=int)
        latent_edge_weight_all = torch.empty(0)

        global_node_index = 1
        global_graph_index = 1

        for data in data_loader:
            _, latent_x, latent_edge_index, latent_edge_weight, b = self.predict(data)

            latent_x_list = [','.join(map(str, row)) for row in latent_x.numpy()]
            node_attributes.extend(latent_x_list)
            #edge_attributes.extend(latent_edge_weight)
            latent_edge_index += global_node_index
            #edges = latent_edge_index.T
            #A.extend([tuple(pair) for pair in edges])
            latent_edge_index_all = torch.cat((latent_edge_index_all,latent_edge_index), dim=1)
            latent_edge_weight_all = torch.cat((latent_edge_weight_all,latent_edge_weight))

            graph_indicator.extend(b+global_graph_index)
            graph_labels.extend(data.y)

            global_graph_index += len(data.y)
            global_node_index += len(latent_x_list)

        A, edge_attributes = sort_edge_index(latent_edge_index_all,latent_edge_weight_all)
        
        A = [tuple(pair) for pair in A.T]
        edge_attributes = list(edge_attributes)

        # Step 1: Sort the list by the first element of each tuple
        #sorted_pairs = sorted(A, key=lambda x: (x[0], x[1]))

        # Step 2: Create a new list where each tuple is followed by its inverse
        #ordered_pairs = []
        #used_pairs = set()

        #for pair in sorted_pairs:
        #    if pair not in used_pairs:
        #        ordered_pairs.append(pair)
        #        ordered_pairs.append((pair[1], pair[0]))
        #        used_pairs.add(pair)
        #        used_pairs.add((pair[1], pair[0]))
        
        #A = ordered_pairs

        #DS_node_attributes.append(','.join(map(str, features)))
        # Save files
        with open(full_data_path + prefix + '_A.txt', 'w') as f:
            for entry in A:
                f.write(f'{entry[0]},{entry[1]}\n')
        with open(full_data_path + prefix + '_graph_indicator.txt', 'w') as f:
            for entry in graph_indicator:
                f.write(f'{entry}\n')
        with open(full_data_path + prefix + '_graph_labels.txt', 'w') as f:
            for label in graph_labels:
                f.write(f'{label}\n')
        with open(full_data_path + prefix + '_node_attributes.txt', 'w') as f:
            for attributes in node_attributes:
                f.write(f'{attributes}\n')
        with open(full_data_path + prefix + '_edge_attributes.txt', 'w') as f:
            for attribute in edge_attributes:
                f.write(f'{attribute}\n')


            #latent_space.append(latent_x)
        
        #latent_space = torch.cat(latent_space, dim=0)
        #torch.save(latent_space, outdir + "latent_space.pt")