#Data input of the autoencoder

from .terminal_colors import tcols
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset
import os
import os.path as osp


class SelectGraph(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        self.data_name = root
        #root = osp.join(osp.dirname(osp.realpath(__file__)), '../', root)
        #root = osp.join(os.getcwd(), "../", root)
        
        super(SelectGraph, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../', self.data_name)
        #path = osp.join(os.getcwd(), "../", self.data_name)
        print(path)
        
        self.data_name = self.data_name.split('/')[-1]    #to obtain just train, valid, test

        data_set = TUDataset(path, name=self.data_name, use_node_attr=True, use_edge_attr=True)

        data, slices = self.collate(data_set)
        torch.save((data, slices), self.processed_paths[0])