#Data input of the autoencoder

from .terminal_colors import tcols
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset
import os
import os.path as osp

from torch.utils.data import Sampler

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


class BalancedRandomSubsetSampler(Sampler):
    def __init__(self, dataset, num_to_sample):
        self.dataset = dataset
        self.num_to_sample = num_to_sample
        self.labels = torch.tensor([data.y.item() for data in dataset])
        self.class0_indices = torch.where(self.labels == 0)[0]
        self.class1_indices = torch.where(self.labels == 1)[0]
        #self.class0_indices = [i for i, data in enumerate(dataset) if data.y == 0]
        #self.class1_indices = [i for i, data in enumerate(dataset) if data.y == 1]

    def __iter__(self):
        class0_samples = self.class0_indices[torch.randperm(len(self.class0_indices))[:self.num_to_sample // 2]]
        class1_samples = self.class1_indices[torch.randperm(len(self.class1_indices))[:self.num_to_sample // 2]]
        indices = torch.cat([class0_samples, class1_samples])
        indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_to_sample
    
class BalancedFixedSubsetSampler(Sampler):
    def __init__(self, dataset, num_to_sample):
        self.dataset = dataset
        self.num_to_sample = num_to_sample
        self.labels = torch.tensor([data.y.item() for data in dataset])
        self.class0_indices = torch.where(self.labels == 0)[0]
        self.class1_indices = torch.where(self.labels == 1)[0]
        self.class0_samples = self.class0_indices[torch.randperm(len(self.class0_indices))[:self.num_to_sample // 2]]
        self.class1_samples = self.class1_indices[torch.randperm(len(self.class1_indices))[:self.num_to_sample // 2]]
        self.indices = torch.cat([self.class0_samples, self.class1_samples])
    
    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())