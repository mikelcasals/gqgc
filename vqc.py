import pennylane as qml

from pennylane import numpy as np
import autograd.numpy as anp
from pennylane.optimize import AdamOptimizer
import variational_forms as vf
#import numpy as np_normal



class VQC:
    def __init__(self, qdevice: qml.device, hpars: dict):
        
        self._hp = {
            'n_qubits': 7,
            'n_features': 3,
            "optimizer": "adam",
            "lr": 0.1,
            "batch_size": 512,
            "num_layers": 2,
        }

        #self._hp.update((k, hpars[k]) for k in self._hp.keys() & hpars.keys())
        self._qdevice = qdevice
        self._nlayers = self._hp["num_layers"]
        
        self._gammas = 0.01 * np.random.randn(self._nlayers, requires_grad=True)
        self._betas = 0.01 * np.random.randn(self._nlayers, requires_grad=True)
        #self._weights = {"betas": self._betas, "gammas": self._gammas}

        self._diff_method = self._select_diff_method(self._hp)
        self._optimizer = self._choose_optimiser(self._hp["optimizer"], self._hp["lr"])
        self._class_loss_function = self._shift_bce
        self._best_valid_loss = 999
        self.all_train_loss = []
        self.all_valid_loss = []

        self._circuit = qml.qnode(self._qdevice, diff_method=self._diff_method)(self._qcircuit)

    def _qcircuit(self, node_features,A, num_nodes, betas, gammas):
        #betas = weights["betas"]
        #gammas = weights["gammas"]
        #print(weights)

        #print(inputs)
        
        #A = inputs[1]
        #node_features = inputs[0]
        #num_nodes = inputs[2]


        """Circuit that uses the permutation equivariant embedding"""
        vf.perm_equivariant_embedding(A, node_features,betas, gammas)
        
        observable = qml.PauliX(0)# @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3) @ qml.PauliZ(4) @ qml.PauliZ(5) @ qml.PauliZ(6)
        for i in range(1, len(A)):
            observable @= qml.PauliX(i)
        return qml.expval(observable)

    @staticmethod
    def _select_diff_method(hpars: dict) -> str:
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
    def _choose_optimiser(choice, lr):
        """
        Choose an optimiser to use in the training of the vqc.
        @choice :: String of the optimiser name you want to use to train vqc.
        @lr     :: Learning rate for the optimiser.
        """
        if choice is None:
            return None

        switcher = {
            "adam": lambda: AdamOptimizer(stepsize=lr),
            "none": lambda: "No Optimiser",
        }
        optimiser = switcher.get(choice, lambda: None)()
        if optimiser is None:
            raise TypeError("Specified optimiser is not an option atm!")

        print("Optimiser used in this run: ")
        print(optimiser, "\n")

        return optimiser
    
    def _shift_bce(self, y_preds, y_batch):
        """Shift the input given to this method and then calculate the binary cross
        entropy loss.

        Args:
            y_preds: The predictions made by the vqc on the data.
            y_batch: Batch of the target array.

        Returns:
            The binary cross entropy loss computed on the given data.
        """
        y_preds = (np.array(y_preds, requires_grad=False) + 1) / 2
        return self._binary_cross_entropy(y_preds, y_batch)
    
    @staticmethod
    def _binary_cross_entropy(y_preds, y_batch):
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
    
    def forward(self, node_features, A, num_nodes):
        return [self._circuit(node_features[i], A[i], num_nodes[i], self._betas, self._gammas) for i in range(len(num_nodes))]
    
    def compute_loss(self, node_features_batch, A_batch, num_nodes_batch, y_batch, betas=None, gammas=None):
        """
        Objective function to be passed through the optimiser.
        Weights is taken as an argument here since the optimiser func needs it.
        We then use the class self variable inside the method.
        """
        #if not weights is None:
        #    self._weights = weights
        if not betas is None:
            self._betas = betas
        if not gammas is None:
            self._gammas = gammas
        predictions = self.forward(node_features_batch, A_batch, num_nodes_batch)
        return self._class_loss_function(predictions, y_batch)
    
    def _validate(self, valid_loader, outdir):
        """
        Calculate the loss on a validation data set.
        """
        x_valid, y_valid = valid_loader
        loss = self.compute_loss(x_valid, y_valid, self._weights)
        self._save_best_loss_model(loss, outdir)

        return loss

    def _train_batch(self, node_features_batch, A_batch, num_nodes_batch, y_batch):
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
        y_batch = np.array(y_batch[:], requires_grad=False)
        _, _, _, _, betas, gammas = self._optimizer.step(
            self.compute_loss, node_features_batch, A_batch, num_nodes_batch, y_batch, self._betas, self._gammas
        )
        self._betas = betas
        self._gammas = gammas
        loss = self.compute_loss(node_features_batch,A_batch, num_nodes_batch, y_batch, self._betas, self._gammas)
        return loss
    
    def _train_all_batches(self, train_loader, batch_size=32):
        """
        Train on the full data set. Add randomness.
        """
        batch_loss_sum = 0
        nb_of_batches = 0
        #x_train, y_train = train_loader
        idx = 0

        while idx < len(train_loader):

            if idx+batch_size >= len(train_loader):
                node_features_batch, A_batch, num_nodes_batch, y_batch = train_loader[idx:]
            else:
                node_features_batch, A_batch, num_nodes_batch, y_batch = train_loader[idx:idx+batch_size]
            
            idx += batch_size


            batch_loss = self._train_batch(node_features_batch, A_batch, num_nodes_batch, y_batch)
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
    
    
    def train_model(self, train_loader, valid_loader, epochs, outdir):
        """Train an instantiated vqc algorithm."""
        #self._print_total_weights()
        print("Training the vqc...")
        rng = np.random.default_rng(12345)
        #batch_seeds = rng.integers(low=0, high=100, size=(epochs, len(train_loader[1])))

        print(self._betas)
        print(self._gammas)
        for epoch in range(epochs):
            train_loss = self._train_all_batches(train_loader)
            #valid_loss = self._validate(valid_loader, outdir)
            #if self._early_stopping(estopping_limit):
            #    break
            valid_loss = 0
            self.all_train_loss.append(train_loss)
            self.all_valid_loss.append(valid_loss)
            self._print_losses(epoch, epochs, train_loss, valid_loss)
            print(self._betas)
            print(self._gammas)

    @staticmethod
    def _print_losses(epoch, epochs, train_loss, valid_loss):
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
        #print(f"Epoch : {epoch + 1}/{epochs}, " f"Valid loss = {valid_loss.item():.8f}")

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

