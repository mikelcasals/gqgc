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

    print(args["compressed"])

    if args["compressed"]==True:
        print("Compressing data...")
        classical_util.save_compressed_data(device, args)
        train_graphs = ae_data.SelectGraph(args['compressed_data_path']+"/train")
        valid_graphs = ae_data.SelectGraph(args['compressed_data_path']+"/valid")
    else:
        print("Using uncompressed data")
        train_graphs = ae_data.SelectGraph(args['data_folder'] + "/train")
        valid_graphs = ae_data.SelectGraph(args['data_folder'] + "/valid")

    if args["train_dataloader_type"] == "fixed_sampling":
        sampler = ae_data.BalancedFixedSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "random_sampling":
        sampler = ae_data.BalancedRandomSubsetSampler(train_graphs, args["num_samples_train"])
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], sampler=sampler)
    elif args["train_dataloader_type"] == "fixed_full":
        train_loader = DataLoader(train_graphs, batch_size=args["batch"], shuffle=True)
    else:
        raise TypeError("Specified train dataloader type not recognized")

    valid_loader = DataLoader(valid_graphs, batch_size=args["batch"], shuffle=False)


    model = GNN(device, args)
    model.export_architecture(outdir)
    model.export_hyperparameters(outdir)
    classical_util.time_the_training(model.train_model, train_loader, valid_loader, args["epochs"],outdir)
    model.loss_plot(outdir)





    

