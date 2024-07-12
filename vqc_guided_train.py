import argparse

from vqc_guided.train import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_folder", type=str, default="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--norm", type=str, default="std", help="Normalization technique to be used")
parser.add_argument("--num_samples", type=int, default=11000, help="Total number of data samples to use")
parser.add_argument("--train_samples", type=int, default=10000, help="The exact number of training events used < num_samples")
parser.add_argument("--valid_samples", type=int, default=1000, help="The exact number of validation events used < num_samples")

parser.add_argument("--aetype", type=str, default="MIAGAE_classifier", help="Type of autoencoder to be used (MIAGAE_vanilla or SAG_model_vanilla)")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--batch", type=int, default=128, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

parser.add_argument("--class_weight", type=float, default=0.7, help="The weight of the classifier BCE loss")
parser.add_argument("--outdir", type=str, default = "SAG_model_new", help="Output directory for the trained model")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--early_stopping", type=int, default=100, help="Number of epochs to wait before early stopping")

parser.add_argument("--input_size", type=int, default=13, help="Number of features per node")
parser.add_argument("--kernels", type=int, default=1, help="Number of kernels")
parser.add_argument("--depth", type=int, default=3, help="Depth of encoder and decoder")
parser.add_argument("--c_rate", type=float, default=0.40, help="Compression ratio for each layer of the encoder")
parser.add_argument("--shapes", type=str, default="13,13,2", help="Shape of each layer in the encoder")
parser.add_argument("--input_size_class", type=int, default=2, help="Number of features per node for the classifier")

parser.add_argument("--n_qubits", type=int, default=10, help="Number of qubits")
parser.add_argument("--n_features", type=int, default=2, help="Number of features per node")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to be used")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--ideal_dev", type=str, default="lightning.qubit", help="Ideal device to be used")
parser.add_argument("--ae_vqc_type", type=str, default="SAG_model_vqc_new", help="Type of autoencoder-vqc to be used (MIAGAE_vqc)")
parser.add_argument("--device", type=str, default="cpu", help="Device to be used")

parser.add_argument("--train_dataloader_type", type=str, default="fixed_sampling", help="Options: fixed_sampling, random_sampling, fixed_full")
parser.add_argument("--num_samples_train", type=int, default=10000, help="Number of subsamples to be used for training")

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
    "shapes": args.shapes,
    "input_size_class": args.input_size_class,
    "n_qubits": args.n_qubits,
    "n_features": args.n_features,
    "optimizer": args.optimizer,
    "n_layers": args.n_layers,
    "ideal_dev": args.ideal_dev,
    "ae_vqc_type": args.ae_vqc_type,
    "device": args.device,
    "train_dataloader_type": args.train_dataloader_type,
    "num_samples_train": args.num_samples_train
}


main(args)


