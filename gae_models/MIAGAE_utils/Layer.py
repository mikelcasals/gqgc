from abc import ABC

#from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.pool.connect.filter_edges import filter_adj
from torch_geometric.nn import global_add_pool as g_pooling
from torch_geometric.utils import sort_edge_index
# from graph_ae.GATConv import GATConv
from .SAGEAttn import SAGEAttn
from graphAE.utils.SAGEAttn import SAGEAttn
from torch_sparse import spspmm
import torch.nn.functional as f
from torch.nn import Parameter
import torch

from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SGAT(torch.nn.Module, ABC):

    def __init__(self, device, size, in_channel, out_channel, heads: int = 1):
        super(SGAT, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.heads = heads
        # self.pm = Parameter(torch.ones([self.size]))
        self.gat_list = torch.nn.ModuleList()

        for i in range(size):
            self.gat_list.append(SAGEAttn(in_channel, out_channel).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        # self.pm.data.fill_(1)
        for conv in self.gat_list:
            conv.reset_parameters()

    def forward(self, x, edge_index, direction=1):
        feature_list = None
        attention_list = []
        # pm = torch.softmax(self.pm, dim=-1)
        idx = 0
        for conv in self.gat_list:
            feature, attn = conv(x, edge_index)
            if feature_list is None:
                feature_list = f.leaky_relu(feature)
            else:
                feature_list += f.leaky_relu(feature)
            attention_list.append(attn)
            idx += 1

        attention_list = torch.stack(attention_list, dim=1)
        if attention_list.shape[1] > 1:
            attention_list = torch.sum(attention_list, dim=1)
        e_batch = edge_index[0]
        node_scores = direction * g_pooling(attention_list, e_batch).view(-1)
        return feature_list, node_scores
