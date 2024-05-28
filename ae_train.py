import argparse

from autoencoders.train import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--data_folder", type=str, default="data/graphdata_10000_part_dist/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--norm", type=str, default="std", help="Normalization technique to be used")
parser.add_argument("--num_samples", type=int, default=10000, help="Total number of data samples to use")
parser.add_argument("--train_samples", type=int, default=8000, help="The exact number of training events used < num_samples")
parser.add_argument("--valid_samples", type=int, default=1000, help="The exact number of validation events used < num_samples")

parser.add_argument("--aetype", type=str, default="SAG_model_classifier", help="Type of autoencoder to be used (MIAGAE_vanilla or SAG_model_vanilla)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--batch", type=int, default=512, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

parser.add_argument("--class_weight", type=float, default=1, help="The weight of the classifier BCE loss")
parser.add_argument("--outdir", type=str, default = "trained_model_sag_classifier", help="Output directory for the trained model")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--early_stopping", type=int, default=100, help="Number of epochs to wait before early stopping")

parser.add_argument("--input_size", type=int, default=13, help="Number of features per node")
parser.add_argument("--kernels", type=int, default=1, help="Number of kernels")
parser.add_argument("--depth", type=int, default=3, help="Depth of encoder and decoder")
parser.add_argument("--c_rate", type=float, default=0.35, help="Compression ratio for each layer of the encoder")
parser.add_argument("--shapes", type=str, default="13,5,1", help="Shape of each layer in the encoder")
parser.add_argument("--input_size_class", type=int, default=1, help="Number of features per node for the classifier")

args = parser.parse_args()

args = {
    "data_folder": args.data_folder,
    "norm": args.norm,
    "num_samples": args.num_samples,
    "train_samples": args.train_samples,
    "valid_samples": args.valid_samples,
    "aetype": args.aetype,
    "lr": args.lr,
    "batch": args.batch,
    "epochs": args.epochs,
    "class_weight": args.class_weight,
    "outdir": args.outdir,
    "seed": args.seed,
    "early_stopping": args.early_stopping,
    "input_size": args.input_size,
    "kernels": args.kernels,
    "depth": args.depth,
    "c_rate": args.c_rate,
    "shapes": args.shapes
}

main(args)
