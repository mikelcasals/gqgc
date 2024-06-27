# Description: Utility functions for classical methods.
import os
from autoencoders import util as ae_util
from autoencoders import data as ae_data
from torch_geometric.loader import DataLoader
from .terminal_colors import tcols
import time

def save_compressed_data(device, args):

    print(os.getcwd())

    print(args["ae_model_path"])
    ae_model_folder = os.path.dirname(args['ae_model_path'])
    print(ae_model_folder)
    hp_file = os.path.join(ae_model_folder, "hyperparameters.json")
    hp = ae_util.import_hyperparams(hp_file)
    ae_model = ae_util.choose_ae_model(args['aetype'], device, hp)
    ae_model.load_model(os.path.join(ae_model_folder, "best_model.pt"))

    # Load the data
    train_graphs = ae_data.SelectGraph(args['data_folder']+"train")
    valid_graphs = ae_data.SelectGraph(args['data_folder']+"valid")
    #test_graphs = ae_data.SelectGraph(args['data_folder']+"test")

    train_loader = DataLoader(train_graphs, batch_size=args["batch"], shuffle=False)
    valid_loader = DataLoader(valid_graphs, batch_size=args["batch"], shuffle=False)
    #test_loader = DataLoader(test_graphs, batch_size=args["batch"], shuffle=False)


    ae_model.save_latent_space(train_loader, args["compressed_data_path"], "train")
    ae_model.save_latent_space(valid_loader, args["compressed_data_path"], "valid")
    #ae_model.save_latent_space(test_loader, args["comporessed_data_path"], "test")

    return

def time_the_training(train: callable, *args):
    """Times the training of the neural network.

    Args:
        train (callable): The training method of the NeuralNetwork class.
        *args: Arguments for the train_model method.
    """
    train_time_start = time.perf_counter()
    train(*args)
    train_time_end = time.perf_counter()
    print(
        tcols.OKCYAN
        + f"Training completed in: {train_time_end-train_time_start:.2e} s or "
        f"{(train_time_end-train_time_start)/3600:.2e} h." + tcols.ENDC
    )