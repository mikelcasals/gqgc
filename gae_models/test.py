# Runs the autoencoder. The normalized or standardized data is imported,
# and the autoencoder model is defined, given the specified options.
# The model is then trained and a loss plot is saved, along with the
# architecture of the model, its hyperparameters, and the best model weights.

import time
import os

from . import util
from .terminal_colors import tcols
from . import data
from torch_geometric.loader import DataLoader
import numpy as np

def main(args):
    device='cpu'
    model_folder = os.path.dirname(args["model_path"])
    hp_file = os.path.join(model_folder, "hyperparameters.json")
    hp = util.import_hyperparams(hp_file)
    
    print(hp)
    # Load the data
    test_graphs = data.SelectGraph(args['data_folder']+"/test")

    test_loader = DataLoader(test_graphs, batch_size=len(test_graphs)//args["num_kfolds"], shuffle=False)

    #Autoencoder model definition
    model = util.choose_gae_model(hp["gae_type"], device, hp)

    model.load_model(args["model_path"])
    
    start_time = time.time()


    test_kfold_gae(model,test_loader, args["num_kfolds"])
    



    end_time = time.time()

    train_time = (end_time - start_time) / 60 

    print(tcols.OKCYAN + f"Testing time: {train_time:.2e} mins." + tcols.ENDC)


def test_kfold_gae(model,test_loader, num_folds):

    all_losses = []
    for test_data in test_loader:
        loss = model.compute_loss(test_data).item()
        all_losses.append(loss)
    
    all_losses = np.array(all_losses)

    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)

    print(tcols.OKCYAN + f"Test loss: {mean_loss:.4f} +/- {std_loss:.4f}" + tcols.ENDC)

