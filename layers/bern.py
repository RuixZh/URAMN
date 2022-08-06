import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from scipy.special import comb
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian
import numpy as np

class Bernprop(nn.Module):
    def __init__(self, K):
        super(Bernprop, self).__init__()
        
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K+1))
        self.temp.data.fill_(1.0)
        

    def forward(self, x, shuf, adj, neighbor, sparse=True):
        TEMP=F.relu(self.temp)
        edge_index, edge_weight, size = adj._indices(),adj._values(), adj.size()
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype, num_nodes=size[0])
        #2I-L
        edge_index2, norm2=add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=size[0])
        norm1 = torch.sparse_coo_tensor(edge_index1, norm1, size)
        norm2 = torch.sparse_coo_tensor(edge_index2, norm2, size)
        
        tmp = []
        tmp.append(x)
        for i in range(self.K):
            if sparse:
                x = torch.sparse.mm(norm2, x)
            else:
                x = torch.mm(norm2, x)
                
            tmp.append(x)
            
        out = (comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]
        for i in range(self.K):
            x=tmp[self.K-i-1]
            
            if sparse:
                x = torch.sparse.mm(norm1, x)
                
            else:
                x = torch.mm(norm1, x)
                
            for j in range(i):
                if sparse:
                    x = torch.sparse.mm(norm1, x)
                    
                else:
                    x = torch.mm(norm1, x)
                    
            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        
        if sparse:
            z_pos = torch.sparse.mm(neighbor, out)
            z_pos = torch.sparse.mm(neighbor, z_pos)

        else:
            z_pos = torch.mm(neighbor, out)
            z_pos = torch.mm(neighbor, z_pos)
            
        if sparse:
            z_neg = torch.sparse.mm(neighbor, out[shuf,:])
            z_neg = torch.sparse.mm(neighbor,z_neg)
            
        else:
            z_neg = torch.mm(neighbor, out[shuf,:])
            z_neg = torch.mm(neighbor, z_neg)
            
        
        return out, z_pos, z_neg
    
    
class Bernprop2(nn.Module):
    def __init__(self, K):
        super(Bernprop2, self).__init__()
        
        self.K = K
        self.temp = nn.Parameter(torch.Tensor(self.K+1))
        self.temp.data.fill_(1.0)
        
    def forward(self, x, shuf, adj, neighbor, sparse=True):
        TEMP=F.relu(self.temp)
        edge_index, edge_weight, size = adj._indices(),adj._values(), adj.size()
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype, num_nodes=size[0])
        norm1 = torch.sparse_coo_tensor(edge_index1, norm1, size)
        Lx = torch.sparse.mm(norm1, x)
        out = TEMP[0] * x + (TEMP[1] - TEMP[0]) * Lx + (TEMP[0] + TEMP[2] - 2 * TEMP[1]) / 4 * torch.sparse.mm(norm1, Lx)
        if sparse:
            z_pos = torch.sparse.mm(neighbor, out)
            z_pos = torch.sparse.mm(neighbor, z_pos)
        else:
            z_pos = torch.mm(neighbor, out)
            z_pos = torch.mm(neighbor, z_pos)       
        if sparse:
            z_neg = torch.sparse.mm(neighbor, out[shuf,:])
            z_neg = torch.sparse.mm(neighbor,z_neg)            
        else:
            z_neg = torch.mm(neighbor, out[shuf,:])
            z_neg = torch.mm(neighbor, z_neg)
            
        
        return out, z_pos, z_neg