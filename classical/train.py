import time
import os
import argparse

#import sys
#sys.path.append('..')
import random
import numpy as np
import torch
from .Classifier import GNN
from . import util as classical_util

from autoencoders import util as ae_util
from autoencoders import data as ae_data
from .terminal_colors import tcols
from torch_geometric.loader import DataLoader

def main(args):
    ae_util.set_seeds(args["seed"])

    device = ae_util.define_torch_device()

    outdir = "./trained_nns/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Load the data
    classical_util.save_compressed_data(device, args)

    train_graphs = ae_data.SelectGraph(args['compressed_data_path']+"/train")
    valid_graphs = ae_data.SelectGraph(args['compressed_data_path']+"/valid")

    train_loader = DataLoader(train_graphs, batch_size=args["batch"], shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=args["batch"], shuffle=False)


    model = GNN(device, args)
    model.export_architecture(outdir)
    model.export_hyperparameters(outdir)
    classical_util.time_the_training(model.train_model, train_loader, valid_loader, args["epochs"],outdir)
    model.loss_plot(outdir)





    

