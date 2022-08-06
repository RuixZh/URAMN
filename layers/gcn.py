import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian
from torch_scatter import scatter_add

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, drop_prob, isBias=False):
        super(GCN, self).__init__()

        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)

        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        elif act == 'selu':
            self.act = nn.SELU()
        elif act == 'celu':
            self.act = nn.CELU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.drop_prob = drop_prob
        self.isBias = isBias


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def normalize_adj(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return edge_index, edge_weight

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq = self.fc_1(seq)
        
        node_size = seq.shape[0]
        edge_index, edge_weight, size = adj._indices(),adj._values(), adj.size()
        edge_index1, norm1 = add_self_loops(edge_index, edge_weight, fill_value=3., num_nodes=node_size)
        edge_index2, norm2 = self.normalize_adj(edge_index1, norm1, node_size)

        adj = torch.sparse_coo_tensor(edge_index2, norm2, size)   
        
        
        if sparse:
            seq = torch.sparse.mm(adj, seq)
        else:
            seq = torch.mm(adj, seq)
            
        if self.isBias:
            seq += self.bias_1

        return self.act(seq)
