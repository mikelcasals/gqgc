
from base_models.GAE_base_model import GAE
import torch
from .MIAGAE_utils.Layer import SGAT
from torch_geometric.nn import TopKPooling
from .MIAGAE_utils.SAGEConv import SAGEConv
from torch_geometric.utils import add_remaining_self_loops
from .terminal_colors import tcols

class MIAGAE(GAE):

    def __init__(self, device="cpu", hpars={}):
        GAE.__init__(self, device, hpars)

        self.hp_MIAGAE = {
            "gae_type": "MIAGAE",
            "num_node_features": 13,
            "depth": 3,
            "shapes": "13,13,1",
            "c_rate": 0.40,
            "kernels": 2
        }
        self.hp_MIAGAE.update((k, hpars[k]) for k in self.hp_MIAGAE.keys() & hpars.keys())
        self.hp_gae.update((k, self.hp_MIAGAE[k]) for k in self.hp_MIAGAE.keys())

        self.gae_type = self.hp_gae["gae_type"]
        self.num_node_features = self.hp_gae["num_node_features"]
        self.depth = self.hp_gae["depth"]
        self.shapes = list(map(int, self.hp_gae["shapes"].split(",")))[0:self.depth]
        self.c_rate = [self.hp_gae["c_rate"]] * self.depth
        self.kernels = self.hp_gae["kernels"]
        
        self.instantiate_encoder()
        self.instantiate_decoder()

    def instantiate_encoder(self):
        """
        This function instantiates the encoder part of the model
        """
        
        self.down_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()
        conv = SGAT(self.device, self.kernels, self.num_node_features, self.shapes[0])
        self.down_list.append(conv)
        for i in range(self.depth - 1):
            pool = TopKPooling(self.shapes[i], self.c_rate[i])
            self.pool_list.append(pool)
            conv = SGAT(self.device, self.kernels, self.shapes[i], self.shapes[i + 1])
            self.down_list.append(conv)
        pool = TopKPooling(self.shapes[-1], self.c_rate[-1])
        self.pool_list.append(pool)
    
    def instantiate_decoder(self):
        """
        This function instantiates the decoder part of the model
        """

        self.up_list = torch.nn.ModuleList()
        for i in range(self.depth - 1):
            conv = SAGEConv(self.shapes[self.depth - i - 1], self.shapes[self.depth - i - 2])
            self.up_list.append(conv)
        conv = SAGEConv(self.shapes[0], self.num_node_features)
        self.up_list.append(conv)

    def encoder_decoder(self, data):
        """
        Forward pass through the encoder and decoder
        @data :: torch_geometric.data object
        """
        x, edge_index, y, batch, edge_weight = data.x, data.edge_index, data.y, data.batch, data.edge_attr
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, num_nodes=x.shape[0])
        edge_weight = edge_weight.squeeze()

        edge_list = []
        perm_list = []
        shape_list = []

        #Encoder
        f, e, b = x, edge_index, batch
        for i in range(self.depth):
            if i < self.depth:
                edge_list.append(e)
            f, attn = self.down_list[i](f, e)
            shape_list.append(f.shape)
            f = torch.nn.functional.leaky_relu(f)

            f, e, edge_weight, b, perm, _ = self.pool_list[i](f, e, edge_weight, b, attn)

            perm_list.append(perm)
        latent_x, latent_edge, latent_edge_weight = f, e, edge_weight

        #Decoder
        z = f
        for i in range(self.depth):
            index = self.depth - i - 1
            shape = shape_list[index]
            up = torch.zeros(shape).to(self.device)
            p = perm_list[index]
            up[p] = z
            z = self.up_list[i](up, edge_list[index])
            if i < self.depth - 1:
                z = torch.relu(z)

        edge_list.clear()
        perm_list.clear()
        shape_list.clear()

        return z, latent_x, latent_edge, latent_edge_weight, b
    
    def gae_network_summary(self):
        """
        Prints a summary of the entire ae network.
        """
        print(tcols.OKGREEN + "Encoder summary:" + tcols.ENDC)
        self.print_summary(self.down_list)
        self.print_summary(self.pool_list)
        print("\n")
        print(tcols.OKGREEN + "Decoder summary:" + tcols.ENDC)
        self.print_summary(self.up_list)
        print("\n\n")




