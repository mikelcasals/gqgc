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

def main(args):
    util.set_seeds(args["seed"])
    device = util.define_torch_device()
    outdir = "./trained_aes/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    # Load the data
    train_graphs = data.SelectGraph(args['data_folder']+"/train")
    valid_graphs = data.SelectGraph(args['data_folder']+"/valid")

    if args["train_dataloader_type"] == "fixed_sampling":
        sampler = data.BalancedFixedSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "random_sampling":
        sampler = data.BalancedRandomSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "fixed_full":
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], shuffle=True)
    else:
        raise TypeError("Specified train dataloader type not recognized")

    valid_loader = DataLoader(valid_graphs, batch_size=len(valid_graphs), shuffle=False)
    
    #Autoencoder model definition
    model = util.choose_ae_model(args["aetype"], device, args)

    start_time = time.time()

    model.export_architecture(outdir)
    model.export_hyperparameters(outdir)
    model.train_autoencoder(train_loader, valid_loader, args["epochs"], outdir)

    end_time = time.time()

    train_time = (end_time - start_time) / 60 

    print(tcols.OKCYAN + f"Training time: {train_time:.2e} mins." + tcols.ENDC)

    model.loss_plot(outdir)
