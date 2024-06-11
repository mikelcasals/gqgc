# Classifier MIAGAE autencoder. Different from the vanilla one since it has a
# classifier attached to the latent space, that does the classification
# for each batch latent space and outputs the binary cross-entropy loss
# that is then used to optimize the autoencoder as a whole.

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GraphConv, SAGEConv
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F

from .MIAGAE import MIAGAE
from .terminal_colors import tcols
from sklearn.metrics import roc_auc_score

class MIAGAE_classifier(MIAGAE):

    def __init__(self, device="cpu", hpars = {}):
        super().__init__(device, hpars)
        new_hp  = {
            "ae_type":"MIAGAE_classifier",
            "adam_betas":(0.9, 0.999),
            "class_weight":0.7,
            "input_size_class":1
        }
        self.hp.update(new_hp)
        self.hp.update((k, hpars[k]) for k in self.hp.keys() & hpars.keys())

        #self.class_loss_function = nn.CrossEntropyLoss()
        self.class_loss_function = nn.BCELoss()

        self.recon_loss_weight = 1 - self.hp["class_weight"]
        self.class_loss_weight = self.hp["class_weight"]
        self.all_recon_loss_valid = []
        self.all_class_loss_valid = []

        self.all_train_acc = []
        self.all_valid_acc = []
        self.all_train_roc_auc = []
        self.all_valid_roc_auc = []

        #Classifier
        self.conv1 = GraphConv(self.hp["input_size_class"], 2,aggr='mean')
        self.conv2 = GraphConv(2, 4,aggr='mean')
        self.conv3 = GraphConv(4, 8,aggr='mean')
        self.fc = torch.nn.Linear(8, 1)


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
            f, attn = self.down_list[i](f, e, self.direction)
            shape_list.append(f.shape)
            f = F.leaky_relu(f)

            f, e, edge_weight, b, perm, _ = self.pool_list[i](f, e, edge_weight, b, attn)

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

        class_loss = self.class_loss_function(class_output.flatten(), data.y.float())
        recon_loss = self.recon_loss_function(z, data.x)

        return self.recon_loss_weight * recon_loss + self.class_loss_weight*class_loss
    
    def compute_accuracy(self,data):
        """
        Compute the accuracy of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed accuracy value.
        """
        data = data.to(self.device)
        _, _, _, _, _,class_output = self.forward(data)
        #preds = torch.argmax(class_output, dim=1)
        preds = torch.round(class_output).squeeze().int()
        #correct = (preds==data.y).sum().item()
        labels = data.y
        correct = torch.eq(preds, data.y).sum().item()
        acc = correct/data.y.size(0)
        return acc
    
    def compute_roc_auc(self, data):
        """
        Compute the roc auc score of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed roc auc score.
        """
        data = data.to(self.device)
        _, _, _, _, _, class_output = self.forward(data)
        class_output = class_output.detach().cpu().numpy()
        y_true = data.y.cpu().numpy()
        roc_auc = roc_auc_score(y_true, class_output)
        return roc_auc

    @staticmethod
    def print_metrics(epoch, epochs, train_loss, valid_losses, train_acc, valid_acc, train_roc_auc, valid_roc_auc):
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

        class_loss = self.class_loss_function(class_output.flatten(), data.y.float())
        recon_loss = self.recon_loss_function(z, data.x)

        valid_loss = (
            self.recon_loss_weight * recon_loss + self.class_loss_weight * class_loss
        )

        valid_acc = self.compute_accuracy(data)
        valid_roc_auc = self.compute_roc_auc(data)

        self.save_best_loss_model(valid_loss, outdir)


        return (valid_loss, recon_loss, class_loss), valid_acc, valid_roc_auc
    
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

    def train_all_batches(self, train_loader):
        """
        Train the autoencoder and classifier for all batches.
        @train_loader :: Pytorch data loader with the training data.

        returns:: The normalised training loss over all the batches
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

            train_loss, train_acc, train_roc_auc = self.train_all_batches(train_loader)
            valid_losses, valid_acc, valid_roc_auc = self.valid(valid_loader, outdir)

            if self.early_stopping():
                break

            self.all_train_loss.append(train_loss.item())
            self.all_valid_loss.append(valid_losses[0].item())
            self.all_recon_loss_valid.append(valid_losses[1].item())
            self.all_class_loss_valid.append(valid_losses[2].item())
            self.all_train_acc.append(train_acc)
            self.all_valid_acc.append(valid_acc)
            self.all_train_roc_auc.append(train_roc_auc)
            self.all_valid_roc_auc.append(valid_roc_auc)

            #self.print_losses(epoch, epochs, train_loss, valid_losses)
            self.print_metrics(epoch, epochs, train_loss, valid_losses, train_acc, valid_acc, train_roc_auc, valid_roc_auc)
    
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