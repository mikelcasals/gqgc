
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import autograd.numpy as anp
import variational_forms as vf
import os
import json
from sklearn.metrics import roc_auc_score

from .terminal_colors import tcols

class VQC:

    def __init__(self, qdevice: qml.device, hpars: dict):
        
        self.hp = {
            'n_qubits': 7,
            'n_features': 1,
            'optimizer': 'adam',
            'lr':0.1,
            'batch':128,
            'n_layers':2,
        }

        self.hp.update((k, hpars[k]) for k in self.hp.keys() & hpars.keys())
        self.qdevice = qdevice
        self.nlayers = self.hp['n_layers']

        np.random.seed(hpars['seed'])

        self.gammas = 0.1*np.random.randn(self.nlayers, requires_grad=True)
        self.betas = 0.1*np.random.randn(self.nlayers, requires_grad=True)

        self.diff_method = self.select_diff_method(hpars)
        self.optimizer = self.choose_optimizer(self.hp['optimizer'], self.hp['lr'])

        self.class_loss_function = self.shift_bce
        self.best_valid_loss = float("inf")
        self.epochs_no_improve = 0
        self.all_train_loss = []
        self.all_valid_loss = []
        self.all_train_acc = []
        self.all_valid_acc = []
        self.all_train_rocauc = []
        self.all_valid_rocauc = []

        self.circuit = qml.qnode(self.qdevice, diff_method = self.diff_method)(self.qcircuit)


    @staticmethod
    def select_diff_method(hpars: dict) -> str:
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
    
    @staticmethod
    def choose_optimizer(choice, lr):
        """
        Choose an optimizer to use in the training of the vqc.
        @choice :: String of the optimizer name you want to use to train vqc.
        @lr     :: Learning rate for the optimizer.
        """
        if choice is None:
            return None

        switcher = {
            "adam": lambda: AdamOptimizer(stepsize=lr),
            "none": lambda: "No Optimizer",
        }
        optimizer = switcher.get(choice, lambda: None)()
        if optimizer is None:
            raise TypeError("Specified optimizer is not an option atm!")

        print("Optimizer used in this run: ")
        print(optimizer, "\n")

        return optimizer

    def shift_bce(self, y_preds, y_batch):
        """Shift the input given to this method and then calculate the binary cross
        entropy loss.

        Args:
            y_preds: The predictions made by the vqc on the data.
            y_batch: Batch of the target array.

        Returns:
            The binary cross entropy loss computed on the given data.
        """
        y_preds = (np.array(y_preds, requires_grad=False) + 1) / 2
        return self.binary_cross_entropy(y_preds, y_batch)
    
    @staticmethod
    def binary_cross_entropy(y_preds, y_batch):
        """
        Binary cross entropy loss calculation.
        """
        eps = anp.finfo(np.float32).eps
        y_preds = anp.clip(y_preds, eps, 1 - eps)
        y_batch = anp.array(y_batch)
        bce_one = anp.array(
            [y * anp.log(pred + eps) for pred, y in zip(y_preds, y_batch)]
        )
        bce_two = anp.array(
            [(1 - y) * anp.log(1 - pred + eps) for pred, y in zip(y_preds, y_batch)]
        )

        bce = anp.array(bce_one + bce_two)

        return -anp.mean(bce)
    

    def qcircuit(self, inputs, betas, gammas):
        #betas = weights["betas"]
        #gammas = weights["gammas"]
        #print(weights)

        #print(inputs)
        
        node_features = inputs[0]
        A = inputs[1]
        #A = inputs[1]
        #node_features = inputs[0]
        #num_nodes = inputs[2]

        """Circuit that uses the permutation equivariant embedding"""
        vf.perm_equivariant_embedding(A, node_features,betas, gammas)
        
        observable = qml.PauliX(0)# @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5) @ qml.PauliZ(6)
        #for i in range(1, len(A)):  #medir primer qubit
            #if i%2 == 0:
            #observable @= qml.PauliX(i)
        return qml.expval(observable)
    
    def forward(self, data):
        return [self.circuit((data[0][i], data[1][i]), self.betas, self.gammas) for i in range(len(data[0]))]

    def compute_loss(self, data, betas=None, gammas=None):
        """
        Objective function to be passed through the optimiser.
        Weights is taken as an argument here since the optimiser func needs it.
        We then use the class self variable inside the method.
        """
        #if not weights is None:
        #    self._weights = weights
        if not betas is None:
            self.betas = betas
        if not gammas is None:
            self.gammas = gammas
        predictions = self.forward(data)
        return self.class_loss_function(predictions, data[2])
    
    def compute_accuracy(self,preds, labels):
        """
        Compute the accuracy of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed accuracy value.
        """
        #preds = torch.argmax(class_output, dim=1)
        #correct = (preds==data.y).sum().item()
        #acc = correct/data.y.size(0)
        labels = np.array(labels, requires_grad=False)
        shifted_preds = (np.array(preds, requires_grad=False) + 1) / 2
        rounded_preds = np.round(shifted_preds).astype(int)

        acc = np.mean(rounded_preds==labels)
        return acc
    
    def compute_roc_auc(self, preds, labels):
        """
        Compute the roc auc score of a forward pass through the ae.
        @data  :: Numpy array of the original input data. Contains target values

        returns :: Float of the computed roc auc score.
        """
        labels = np.array(labels, requires_grad=False)
        shifted_preds = (np.array(preds, requires_grad=False) + 1) / 2
        roc_auc = roc_auc_score(labels, shifted_preds)
        return roc_auc

    def compute_loss_acc_rocauc(self, data, betas=None, gammas=None):
        """
        Objective function to be passed through the optimiser.
        Weights is taken as an argument here since the optimiser func needs it.
        We then use the class self variable inside the method.
        """
        #if not weights is None:
        #    self._weights = weights
        if not betas is None:
            self.betas = betas
        if not gammas is None:
            self.gammas = gammas
        predictions = self.forward(data)
        loss = self.class_loss_function(predictions, data[2])
        
        acc = self.compute_accuracy(predictions, data[2])
        rocauc = self.compute_roc_auc(predictions, data[2])

        return loss, acc, rocauc
    
    def validate(self, valid_loader, outdir):
        """
        Calculate the loss on a validation data set.
        """
        data = next(iter(valid_loader))
        loss, acc, rocauc = self.compute_loss_acc_rocauc(data, self.betas, self.gammas)
        self.save_best_loss_model(loss, outdir)

        return loss, acc, rocauc

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
        _, betas, gammas = self.optimizer.step(
            self.compute_loss, data, self.betas, self.gammas
        )
        self.betas = betas
        self.gammas = gammas
        loss, acc, rocauc = self.compute_loss_acc_rocauc(data, self.betas, self.gammas)
        return loss, acc, rocauc

    def train_all_batches(self, train_loader, batch_size=32):
        """
        Train on the full data set. Add randomness.
        """
        batch_loss_sum = 0
        batch_acc_sum = 0
        batch_rocauc_sum = 0
        nb_of_batches = 0
        #x_train, y_train = train_loader

        
        for data in train_loader:
            batch_loss, batch_acc, batch_rocauc = self.train_batch(data)
            batch_loss_sum += batch_loss
            batch_acc_sum += batch_acc
            batch_rocauc_sum += batch_rocauc
            nb_of_batches += 1

        return batch_loss_sum / nb_of_batches, batch_acc_sum / nb_of_batches, batch_rocauc_sum / nb_of_batches

        idx = 0

        while idx < len(train_loader):

            if idx+batch_size >= len(train_loader):
                node_features_batch, A_batch, num_nodes_batch, y_batch = train_loader[idx:]
            else:
                node_features_batch, A_batch, num_nodes_batch, y_batch = train_loader[idx:idx+batch_size]
            
            idx += batch_size


            batch_loss = self.train_batch(node_features_batch, A_batch, num_nodes_batch, y_batch)
            batch_loss_sum += batch_loss
            nb_of_batches += 1

        #for x_batch, y_batch in train_loader:
            #np.random.seed(batch_seed[nb_of_batches])
            #perm = np.random.permutation(len(y_batch))
            #x_batch = x_batch[perm]
            #y_batch = y_batch[perm]
            #batch_loss = self._train_batch(x_batch, y_batch)
            #batch_loss_sum += batch_loss
            #nb_of_batches += 1

        return batch_loss_sum / nb_of_batches
    

    def train_model(self, train_loader, valid_loader, epochs, estopping_limit, outdir):
        """Train an instantiated vqc algorithm."""
        #self._print_total_weights()
        print(tcols.OKCYAN + "Training the vqc..." + tcols.ENDC)
        rng = np.random.default_rng(12345)
        #batch_seeds = rng.integers(low=0, high=100, size=(epochs, len(train_loader[1])))

        print(self.betas)
        print(self.gammas)
        for epoch in range(epochs):
            train_loss, train_acc, train_rocauc = self.train_all_batches(train_loader)
            valid_loss, valid_acc, valid_rocauc = self.validate(valid_loader, outdir)
            if self.early_stopping(estopping_limit):
                break
            self.all_train_loss.append(train_loss)
            self.all_valid_loss.append(valid_loss)
            self.all_train_acc.append(train_acc)
            self.all_valid_acc.append(valid_acc)
            self.all_train_rocauc.append(train_rocauc)
            self.all_valid_rocauc.append(valid_rocauc)
            self.print_metrics(epoch, epochs, train_loss, valid_loss, train_acc, valid_acc, train_rocauc, valid_rocauc)
            print(self.betas)
            print(self.gammas)

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
    def print_losses(epoch, epochs, train_loss, valid_loss):
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

    def predict(self, x_data) -> np.ndarray:
        """
        Compute the prediction of the vqc on a data array. The output is casted to
        a list because the test.py is desinged to be model agnostic between VQC and
        HybridVQC. The predict function of the latter outputs 3 values, from which we
        want the last one, i.e., the classification branch output (see ae_classifier.py)

        @x_data :: Input array to pass through the vqc.

        returns :: The latent space of the ae and the reco data.
        """
        x_data = np.array(x_data[:, :], requires_grad=False)
        classification_output = self.forward(x_data)

        return [classification_output]
    
    def export_hyperparameters(self, outdir):
        """
        Saves the hyperparameters of the model to a json file.
        @outdir :: Directory where to save the json file.
        """
        file_path = os.path.join(outdir, "hyperparameters.json")
        params_file = open(file_path, "w")
        json.dump(self.hp, params_file)
        params_file.close()

        print(tcols.OKGREEN + f"Hyperparameters exported to {file_path}!" + tcols.ENDC)
    
    def export_architecture(self, outdir):
        """Saves a drawing of the circuit to a text file.

        Args:
            outdir: The output folder where the circuit file will be saved.
        """
        outfile = os.path.join(outdir, "circuit_architecture.pdf")
        #fig, ax = self.draw()
        #fig.savefig(outfile)

        #print(tcols.OKGREEN + f"Architecture exported to {outfile}!" + tcols.ENDC)

    def early_stopping(self, early_stopping_limit) -> bool:
        """
        Stops the training if there has been no improvement in the loss
        function during the past, e.g. 10, number of epochs.

        returns :: True for when the early stopping limit was exceeded
            and false otherwise.
        """
        if self.epochs_no_improve >= early_stopping_limit:
            return 1
        return 0
    
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
            #if outdir is not None:
            #    np.save(outdir + "best_model.npy", self.gammas, self.betas)
        else:
            self.epochs_no_improve += 1

    def draw(self):
        """
        Draws the circuit using dummy parameters.
        Parameterless implementation is not yet available in pennylane,
        and it seems not feasible either by the way pennylane is constructed.
        """
        drawing = qml.draw_mpl(self.circuit)
        fig, ax = drawing([0] * int(self.hp["n_features"]), self.gammas, self.betas)

        return fig, ax