import argparse

from gae_models.train import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default="gpu", help="Device to run the model on")
parser.add_argument("--outdir", type=str, default = "SAG_model_for_test", help="Output directory for the trained model")
parser.add_argument("--data_folder", type=str, default="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--train_dataloader_type", type=str, default="fixed_full", help="Options: fixed_sampling, random_sampling, fixed_full")
parser.add_argument("--num_samples_train", type=int, default=10000, help="Number of subsamples to be used for training")

parser.add_argument("--gae_type", type=str, default="SAG_model", help="Type of autoencoder to be used (MIAGAE or SAG_model)")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--early_stopping", type=int, default=25, help="Number of epochs to wait before early stopping")
parser.add_argument("--num_node_features", type=int, default=13, help="Number of features per node")
parser.add_argument("--depth", type=int, default=3, help="Depth of encoder and decoder")
parser.add_argument("--shapes", type=str, default="13,13,2", help="Shape of each layer in the encoder")
parser.add_argument("--c_rate", type=float, default=0.40, help="Compression ratio for each layer of the encoder")

parser.add_argument("--kernels", type=int, default=1, help="Number of kernels (only used in MIAGAE)")

parser.add_argument("--batch", type=int, default=128, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

args = parser.parse_args()

args = {
    "seed": args.seed,
    "device": args.device,
    "outdir": args.outdir,
    "data_folder": args.data_folder,
    "train_dataloader_type": args.train_dataloader_type,
    "num_samples_train": args.num_samples_train,
    "gae_type": args.gae_type,
    "lr": args.lr,
    "early_stopping": args.early_stopping,
    "num_node_features": args.num_node_features,
    "depth": args.depth,
    "shapes": args.shapes,
    "c_rate": args.c_rate,
    "kernels": args.kernels,
    "batch": args.batch,
    "epochs": args.epochs,
    "guided": False
}

main(args)
