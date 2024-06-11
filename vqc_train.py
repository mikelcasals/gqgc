import argparse

from vqc.train import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_folder", type=str, default="data/graphdata_10000_part_dist/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--norm", type=str, default="std", help="Normalization technique to be used")
parser.add_argument("--ae_model_path", type=str, default="trained_aes/trained_model_sag/", help="Path to the autoencoder model")
parser.add_argument("--compressed_data_path", type=str, default="compressed_data/", help="Path to save the compressed data")
parser.add_argument("--batch", type=int, default=512, help="Batch size")
parser.add_argument("--outdir", type=str, default="trained_model_sag", help="Output directory for the trained model")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--aetype", type=str, default="SAG_model_vanilla", help="Type of autoencoder to be used (MIAGAE_vanilla or SAG_model_vanilla)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--early_stopping", type=int, default=20, help="Number of epochs to wait before early stopping")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--input_size", type=int, default=1, help="Number of features per node")

parser.add_argument("--n_qubits", type=int, default=7, help="Number of qubits")
parser.add_argument("--n_features", type=int, default=1, help="Number of features per node")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to be used")
parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
parser.add_argument("--ideal_dev", type=str, default="lightning.qubit", help="Ideal device to be used")


args  = parser.parse_args()

args = {
    "data_folder": args.data_folder,
    "norm": args.norm,
    "ae_model_path": args.ae_model_path,
    "compressed_data_path": args.compressed_data_path,
    "batch": args.batch,
    "outdir": args.outdir,
    "epochs": args.epochs,
    "adam_betas":(0.9,0.99),
    "input_size": args.input_size,
    "hidden": 64,
    "num_classes": 2,
    "dropout": 0.5,
    "lr": args.lr,
    "aetype": args.aetype,
    "seed": args.seed,
    "early_stopping": args.early_stopping,
    "n_qubits": args.n_qubits,
    "n_features": args.n_features,
    "optimizer": args.optimizer,
    "n_layers": args.n_layers,
    "ideal_dev": args.ideal_dev
}

main(args)
