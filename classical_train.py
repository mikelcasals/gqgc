import argparse

from classical.train import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_folder", type=str, default="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--norm", type=str, default="maxabs", help="Normalization technique to be used")
parser.add_argument("--ae_model_path", type=str, default="trained_aes/trained_model_miagae/", help="Path to the autoencoder model")
parser.add_argument("--compressed_data_path", type=str, default="compressed_data/", help="Path to save the compressed data")
parser.add_argument("--batch", type=int, default=128, help="Batch size")
parser.add_argument("--outdir", type=str, default="classical_random_sampling", help="Output directory for the trained model")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--aetype", type=str, default="MIAGAE_vanilla", help="Type of autoencoder to be used (MIAGAE_vanilla or SAG_model_vanilla)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--early_stopping", type=int, default=25, help="Number of epochs to wait before early stopping")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--input_size", type=int, default=13, help="Number of features per node")
parser.add_argument("--train_dataloader_type", type=str, default="random_sampling", help="Options: fixed_sampling, random_sampling, fixed_full")
parser.add_argument("--num_samples_train", type=int, default=10000, help="Number of subsamples to be used for training")
parser.add_argument("--compressed", action='store_true', default="False", help="Whether to use compressed data or not")
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
    "train_dataloader_type": args.train_dataloader_type,
    "num_samples_train": args.num_samples_train,
    "compressed": args.compressed
}

main(args)
