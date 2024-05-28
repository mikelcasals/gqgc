# Classifier MIAGAE autencoder. Different from the vanilla one since it has a
# classifier attached to the latent space, that does the classification
# for each batch latent space and outputs the binary cross-entropy loss
# that is then used to optimize the autoencoder as a whole.

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GraphConv, SAGEConv
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F

from .SAG_model import SAG_model
from .terminal_colors import tcols

class SAG_model_classifier(SAG_model):

    def __init__(self, device="cpu", hpars = {}):
        super().__init__(device, hpars)
        new_hp  = {
            "ae_type":"SAG_model_classifier",
            "adam_betas":(0.9, 0.999),
            "class_weight":0.7,
            "input_size_class":1
        }
        self.hp.update(new_hp)
        self.hp.update((k, hpars[k]) for k in self.hp.keys() & hpars.keys())

        self.class_loss_function = nn.CrossEntropyLoss()

        self.recon_loss_weight = 1 - self.hp["class_weight"]
        self.class_loss_weight = self.hp["class_weight"]
        self.all_recon_loss_valid = []
        self.all_class_loss_valid = []

        #Classifier
        self.conv1 = GraphConv(self.hp["input_size_class"], 2,aggr='mean')
        self.conv2 = GraphConv(2, 4,aggr='mean')
        self.conv3 = GraphConv(4, 8,aggr='mean')
        self.fc = torch.nn.Linear(8, 2)


    def forward(self, data):
        """
        Forward pass through the autoencoder and classifier
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
            f = self.down_list[i](f, e)
            shape_list.append(f.shape)
            f = F.leaky_relu(f)
            f, e, edge_weight, b, perm, _ = self.pool_list[i](f, e, edge_weight, batch=b)
            perm_list.append(perm)
        
        latent_x, latent_edge, latent_batch = f, e, b

        #Classifier
        c = self.conv1(latent_x, latent_edge,edge_weight)
        c = F.relu(c)
        c = self.conv2(c, latent_edge,edge_weight)
        c = F.relu(c)
        c = self.conv3(c, latent_edge,edge_weight)
        c = F.relu(c)
        c = global_mean_pool(c, latent_batch)
        c = self.fc(c)
        #x = F.dropout(x, p=0.1, training=self.training)
        class_output = torch.sigmoid(c)

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
    
    def compute_loss(self, data):
        """
        Compute the loss of a forward pass through the ae and
        classifier. Combine the two losses and return the one loss.
        @data: :: Data object containing the graph data.

        returns :: Float of the computed combined loss function value.
        """
        data = data.to(self.device)
        z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)

        class_loss = self.class_loss_function(class_output, data.y.long())
        recon_loss = self.recon_loss_function(z, data.x)

        return self.recon_loss_weight * recon_loss + self.class_loss_weight*class_loss
    
    @staticmethod
    def print_losses(epoch, epochs, train_loss, valid_losses):
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
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid loss = {valid_losses[0].item():.8f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid recon loss (no weight) = {valid_losses[1].item():.8f}"
        )
        print(
            f"Epoch : {epoch + 1}/{epochs}, "
            f"Valid class loss (no weight) = {valid_losses[2].item():.8f}"
        )
    @torch.no_grad()
    def valid(self, valid_loader, outdir):
        """
        Evaluate the validation loss for the model and save the model if a
        new minimum is found.
        @valid_loader :: Pytorch data loader with the validation data.
        @outdir       :: Output folder where to save the model.

        returns :: Pytorch loss object of the validation loss.
        """
        batch_loss_sum = 0
        nb_of_batches = 0
        data = next(iter(valid_loader))
        data = data.to(self.device)
        self.eval()
        z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)

        class_loss = self.class_loss_function(class_output, data.y.long())
        recon_loss = self.recon_loss_function(z, data.x)

        valid_loss = (
            self.recon_loss_weight * recon_loss + self.class_loss_weight * class_loss
        )

        self.save_best_loss_model(valid_loss, outdir)

        return valid_loss, recon_loss, class_loss
    
    def train_all_batches(self, train_loader):
        """
        Train the autoencoder and classifier for all batches.
        @train_loader :: Pytorch data loader with the training data.

        returns:: The normalised training loss over all the batches
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
        Train the classifier autoencoder.
        @train_loader :: Pytorch data loader with the training data.
        @valid_loader :: Pytorch data loader with the validation data.
        @epochs       :: The number of epochs to train for.
        @outdir       :: The output dir where to save the train results.
        """
        self.instantiate_adam_optimizer()
        self.optimizer_summary()
        print(tcols.OKCYAN)
        print("Training the " + self.hp["ae_type"] + " AE model...")
        print(tcols.ENDC)

        for epoch in range(epochs):
            self.train()

            train_loss = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader, outdir)

            if self.early_stopping():
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_recon_loss_valid.append(valid_losses[1].item())
            self.all_class_loss_valid.append(valid_losses[2].item())

            self.print_losses(epoch, epochs, train_loss, valid_losses)
    
    @torch.no_grad()
    def predict(self, data):
        """
        Compute the prediction of the autoencoder.
        @data :: Input array to pass through the autoencoder.

        returns :: The latent space of the ae and the reco data.
        """
        data = data.to(self.device)
        self.eval()
        reconstructed_z, latent_x, latent_edge, edge_weight, b, class_output = self.forward(data)

        reconstructed_z = reconstructed_z.cpu().numpy()
        latent_x = latent_x.cpu().numpy()
        latent_edge_index = latent_edge.cpu().numpy()
        latent_edge_weight = edge_weight.cpu().numpy()
        b = b.cpu().numpy()
        class_output = class_output.cpu().numpy()

        return reconstructed_z, latent_x, latent_edge_index, latent_edge_weight, b, class_output