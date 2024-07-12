import argparse

from classifier_models.test import main

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_folder", type=str, default="data/graphdata_105000_train50000_valid5000_test50000_part_dist_maxabs/", help="Folder containing the graph data to bed fed to the autoencoder")
parser.add_argument("--model_path", type=str, default = "trained_classifiers/GNN/best_model.pt")
parser.add_argument("--num_kfolds", type=int, default=5, help="Number of k-folds to be used for testing")
parser.add_argument("--compressed", action='store_true', help="Whether the data is compressed or not")


args = parser.parse_args()

args = {
    "data_folder": args.data_folder,
    "model_path": args.model_path,
    "num_kfolds": args.num_kfolds,
    "compressed": args.compressed,
}

main(args)
