import argparse
import numpy as np
from load_data import *
from preprocessing import *
from graphAE.utils.CustomDataSet import SelectGraph
from torch_geometric.loader import DataLoader
from graphAE.utils.train_utils import train_cp
import torch
import random


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
    num_epoch = args.e
    batch_size = args.batch
    device = torch.device(args.device)

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


    X, y = load_data(num_samples, part_dist, filter_outliers, subsampled_data_path)

    #plot_hist(X)

    X, y = augment_features(X,y,num_samples, part_dist, filter_outliers, augmented_data_path)

    #train valid test split

    train_X, train_y, valid_X, valid_y, test_X, test_y = split_data(X, y, train_size, valid_size, test_size, random_seed)

    #fit standardizer to train data
    train_X, valid_X, test_X = standardize_features(train_X, valid_X, test_X)

    #transform to graph format
    full_graph_data_path = graph_data_path + "/data_" + str(num_samples)
    if part_dist:
        full_graph_data_path += "_part_dist"
    if filter_outliers:
        full_graph_data_path += "_filtered"

    format_to_graph(train_X, train_y, valid_X, valid_y, test_X, test_y, full_graph_data_path)

    SelectGraph.data_name = full_graph_data_path + "/train"
    train_graphs = SelectGraph(SelectGraph.data_name)
    SelectGraph.data_name = full_graph_data_path + "/valid"
    valid_graphs = SelectGraph(SelectGraph.data_name)
    SelectGraph.data_name = full_graph_data_path + "/test"
    test_graphs = SelectGraph(SelectGraph.data_name)

    input_size = train_graphs.num_features
    shapes = list(map(int, args.shapes.split(",")))

    train_set = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    valid_set = DataLoader(valid_graphs, batch_size=batch_size, shuffle=False)
    test_set = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    if args.m == "MIAGAE":
        from graphAE.models.MIAGAE import Net
        model = Net(input_size, args.k, args.depth, [args.c_rate] * args.depth, shapes, device).to(device)
    elif args.m == "SAGpool":
        from graphAE.models.SAG_model import Net
        model = Net(input_size, args.depth, [args.c_rate] * args.depth, shapes, device).to(device)
    else:
        print("model not found")
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_cp(model, optimizer, device, train_set, valid_set, num_epoch, args.model_dir, args.m)

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
    parser.add_argument('--e', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch', type=int, default=512, help="batch size")
    parser.add_argument('--device', type=str, default='cuda', help="cuda / cpu")
    parser.add_argument('--shapes', type=str, default="32,16,3", help="shape of each layer in encoder")
    parser.add_argument('--m', type=str, default='SAGpool', help="model name, MIAGAE or SAGpool")
    parser.add_argument('--k', type=int, default=2, help="number of kernels of MIAGAE")
    parser.add_argument('--depth', type=int, default=3, help="depth of encoder and decoder")
    parser.add_argument('--c_rate', type=float, default=0.35, help="compression ratio for each layer of encoder")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--model_dir', type=str, default="saved_model/", help="path to save model")

    args = parser.parse_args()
    main(args)


