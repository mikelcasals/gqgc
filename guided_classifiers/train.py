# Runs the autoencoder. The normalized or standardized data is imported,
# and the autoencoder model is defined, given the specified options.
# The model is then trained and a loss plot is saved, along with the
# architecture of the model, its hyperparameters, and the best model weights.

import time
import os

from . import util
from gae_models import util as gae_util
from .terminal_colors import tcols
from gae_models import data as gae_data
from torch_geometric.loader import DataLoader

def main(args):
    gae_util.set_seeds(args["seed"])
    device = gae_util.define_torch_device(args["device"])
    outdir = "./trained_guided_classifiers/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Load the data
    train_graphs = gae_data.SelectGraph(args['data_folder']+"/train")
    valid_graphs = gae_data.SelectGraph(args['data_folder']+"/valid")

    if args["train_dataloader_type"] == "fixed_sampling":
        sampler = gae_data.BalancedFixedSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "random_sampling":
        sampler = gae_data.BalancedRandomSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "fixed_full":
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], shuffle=True)
    else:
        raise TypeError("Specified train dataloader type not recognized")

    valid_loader = DataLoader(valid_graphs, batch_size=len(valid_graphs), shuffle=False)
    
    #Model definition
    model = util.choose_guided_classifier_model(args["gae_type"], args["classifier_type"], device, args)
    print(type(model).__mro__)
    start_time = time.time()

    model.export_architecture(outdir)
    model.export_hyperparameters(outdir)
    model.train_model(train_loader, valid_loader, args["epochs"], outdir)

    end_time = time.time()

    train_time = (end_time - start_time) / 60 

    print(tcols.OKCYAN + f"Training time: {train_time:.2e} mins." + tcols.ENDC)

    model.loss_plot(outdir)
