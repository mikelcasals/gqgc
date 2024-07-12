# Description: Utility functions for classical methods.
import os
from gae_models import util as gae_util
from autoencoders import data as ae_data
from torch_geometric.loader import DataLoader
from .terminal_colors import tcols
import time
import shutil
import torch
from torch_geometric.utils import sort_edge_index

def save_compressed_data(device, args, prefix):

    print(os.getcwd())

    print(args["gae_model_path"])
    gae_model_folder = os.path.dirname(args['gae_model_path'])
    print(gae_model_folder)
    hp_file = os.path.join(gae_model_folder, "hyperparameters.json")
    hp = gae_util.import_hyperparams(hp_file)
    gae_model = gae_util.choose_ae_model(args['gae_type'], device, hp)
    gae_model.load_model(os.path.join(gae_model_folder, "best_model.pt"))

    # Load the data
    graphs = ae_data.SelectGraph(args['data_folder']+ "/" + prefix)
    #valid_graphs = ae_data.SelectGraph(args['data_folder']+"valid")
    #test_graphs = ae_data.SelectGraph(args['data_folder']+"test")

    loader = DataLoader(graphs, batch_size=args["batch"], shuffle=False)
    #valid_loader = DataLoader(valid_graphs, batch_size=args["batch"], shuffle=False)
    #test_loader = DataLoader(test_graphs, batch_size=args["batch"], shuffle=False)


    save_latent_space(gae_model, loader, args["compressed_data_path"], prefix)
    #save_latent_space(gae_model, valid_loader, args["compressed_data_path"], "valid")
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

def choose_classifier_model(classifier_type, device, hyperparams) -> callable:
    """
    Picks and loads one of the implemented autoencoder model classes.
    @ae_type     :: String of the type of autoencoder that you want to load.
    @device      :: String of the device to load it on: 'cpu' or 'gpu'.
    @hyperparams :: Dictionary of the hyperparameters to load with.

    returns :: The loaded autoencoder model with the given hyperparams.
    """
    from classifier_models.classical.classical_GNN import ClassicalGNN
    from classifier_models.quantum.QGNN1 import QGNN1
    from classifier_models.quantum.QGNN2 import QGNN2
    from classifier_models.quantum.QGNN3 import QGNN3
    switcher = {
        "ClassicalGNN": lambda : ClassicalGNN(device=device, hpars=hyperparams).to(device),
        "QGNN1": lambda : QGNN1(device=device, hpars=hyperparams).to(device),
        "QGNN2": lambda : QGNN2(device=device, hpars=hyperparams).to(device),
        "QGNN3": lambda : QGNN3(device=device, hpars=hyperparams).to(device)
    }
    model = switcher.get(classifier_type, lambda: None)()
    if model is None:
        raise TypeError("Specified classifier type does not exist!")

    return model


def save_latent_space(gae_model, data_loader, outdir, prefix = "train"):
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
            _, latent_x, latent_edge_index, latent_edge_weight, b = gae_model.predict(data)

            latent_x_list = [','.join(map(str, row)) for row in latent_x.numpy()]
            node_attributes.extend(latent_x_list)
            latent_edge_index += global_node_index
            latent_edge_index_all = torch.cat((latent_edge_index_all,latent_edge_index), dim=1)
            latent_edge_weight_all = torch.cat((latent_edge_weight_all,latent_edge_weight))

            graph_indicator.extend(b+global_graph_index)
            graph_labels.extend(data.y)

            global_graph_index += len(data.y)
            global_node_index += len(latent_x_list)

        A, edge_attributes = sort_edge_index(latent_edge_index_all,latent_edge_weight_all)
        
        A = [tuple(pair) for pair in A.T]
        edge_attributes = list(edge_attributes)

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