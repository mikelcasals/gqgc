import argparse

from classifier_models.train import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default="gpu", help="Device to run the model on")
parser.add_argument("--outdir", type=str, default = "GNN", help="Output directory for the trained model")
parser.add_argument("--data_folder", type=str, default="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--compressed", action='store_true', help="Whether to use compressed data or not")
parser.add_argument("--gae_type", type=str, default="SAG_model", help="Type of autoencoder to use to compress data")
parser.add_argument("--gae_model_path", type=str, default="trained_gaes/SAG_model_for_QGNN2_fixed_full/", help="Path to the autoencoder model")
parser.add_argument("--compressed_data_path", type=str, default="compressed_data/", help="Path to save the compressed data")
parser.add_argument("--train_dataloader_type", type=str, default="fixed_sampling", help="Options: fixed_sampling, random_sampling, fixed_full")
parser.add_argument("--num_samples_train", type=int, default=10000, help="Number of subsamples to be used for training")

parser.add_argument("--classifier_type", type=str, default="ClassicalGNN", help="Type of classifier to be used")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--early_stopping", type=int, default=25, help="Number of epochs to wait before early stopping")

parser.add_argument("--hidden_size", type=int, default=2, help="Hidden size of the GNN")

parser.add_argument("--batch", type=int, default=128, help="Batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

parser.add_argument("--quantum", action='store_true',help='Whether a quantum classifier is used or not')
parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the quantum classifier")
parser.add_argument("--num_features", type=int, default=2, help="Number of features per node")
parser.add_argument("--ideal_dev", type=str, default="lightning.qubit", help="Ideal device to run the quantum classifier")
parser.add_argument("--n_qubits", type=int, default=10, help="Number of qubits in quantum device")

args  = parser.parse_args()

args = {
    "seed": args.seed,
    "device": args.device,
    "outdir": args.outdir,
    "data_folder": args.data_folder,
    "compressed": args.compressed,
    "gae_type": args.gae_type,
    "gae_model_path": args.gae_model_path,
    "compressed_data_path": args.compressed_data_path,
    "train_dataloader_type": args.train_dataloader_type,
    "num_samples_train": args.num_samples_train,
    "classifier_type": args.classifier_type,
    "lr": args.lr,
    "early_stopping": args.early_stopping,
    "batch": args.batch,
    "epochs": args.epochs,
    "quantum":args.quantum,
    "num_layers": args.num_layers,
    "num_features": args.num_features,
    "ideal_dev": args.ideal_dev,
    "n_qubits": args.n_qubits,
    "guided": False
}

main(args)
