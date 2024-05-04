import argparse
import numpy as np
from load_data import *
from preprocessing import *


def main(args):
    num_samples = args.num_samples
    part_dist = args.part_dist
    filter_outliers = args.filter_outliers
    subsampled_data_path = args.subsampled_data_path
    augmented_data_path = args.augmented_data_path
    train_size = args.train_size
    valid_size = args.valid_size
    test_size = args.test_size
    random_seed = args.random_seed
    graph_data_path = args.graph_data_path

    np.random.seed(random_seed)

    X, y = load_data(num_samples, part_dist, filter_outliers, subsampled_data_path)

    #plot_hist(X)

    X, y = augment_features(X,y,num_samples, part_dist, filter_outliers, augmented_data_path)

    #train valid test split

    train_X, train_y, valid_X, valid_y, test_X, test_y = split_data(X, y, train_size, valid_size, test_size, random_seed)

    #fit standardizer to train data
    train_X, valid_X, test_X = standardize_features(train_X, valid_X, test_X)

    #transform to graph format
    format_to_graph(train_X, train_y, valid_X, valid_y, test_X, test_y, graph_data_path)

    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_samples', type=int, default=10000, help="Total number of data samples to use")
    parser.add_argument('--part_dist', type=bool, default=True, help="Sample according to the distribution of number of particles in the jets")
    parser.add_argument('--filter_outliers', type=bool, default=False, help="Whether or not to filter out the outliers in terms of number of particles")
    parser.add_argument('--subsampled_data_path', type=str, default="subsampled_data", help="Path in which the subsampled data is saved")
    parser.add_argument('--augmented_data_path', type=str, default="augmented_data", help="Path in which the augmented data is saved")
    parser.add_argument('--train_size', type=float, default=0.8, help="Proportion of data samples for training")
    parser.add_argument('--valid_size', type=float, default=0.1, help="Proportion of data samples for validation")
    parser.add_argument('--test_size', type=float, default=0.1, help="Proportion of data samples for test")
    parser.add_argument('--random_seed', type=int, default=42, help="Seed for reproducibility")
    parser.add_argument('--graph_data_path', type=str, default="graph_data", help="Path in which the graph formatted data is saved")



    args = parser.parse_args()
    main(args)


