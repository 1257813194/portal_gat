import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
import torch.nn as nn
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn import GCNConv,GATConv
from .gat import get_p,get_q

class GAT(torch.nn.Module):
    def __init__(self,add, hidden_dims):
        super(GAT, self).__init__() 
        in_dim, num_hidden, out_dim=hidden_dims
        self.conv1 = GATConv(in_dim+add, num_hidden)
        self.prelu1 = nn.PReLU(num_hidden)
        self.conv2 = GATConv(num_hidden,out_dim)
        self.prelu2 = nn.PReLU(out_dim)
    def forward(self, x, edge_index,id):
        id_tensor =torch.tensor(np.tile(np.array([id]), (x.shape[0], 1)), dtype=torch.float32).cuda()
        x = torch.cat((x,id_tensor),dim=1)
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self,add, hidden_dims):
        super(GCN, self).__init__() 
        in_dim, num_hidden, out_dim=hidden_dims
        self.conv1 = GCNConv(in_dim+add, num_hidden)
        self.prelu1 = nn.PReLU(num_hidden)
        self.conv2 = GCNConv(num_hidden,out_dim)
        self.prelu2 = nn.PReLU(out_dim)
    def forward(self, x, edge_index,id):
        id_tensor =torch.tensor(np.tile(np.array([id]), (x.shape[0], 1)), dtype=torch.float32).cuda()
        x = torch.cat((x,id_tensor),dim=1)
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        return x
    
class Generator(nn.Module):
    def __init__(self, hidden_dims):
        super(Generator, self).__init__()
        in_dim, num_hidden, out_dim=hidden_dims
        self.l1 = nn.Linear(out_dim, num_hidden)
        self.prelu1 = nn.PReLU(num_hidden)
        self.l2 = nn.Linear(num_hidden, in_dim)
    def forward(self, z):
        h1 = self.prelu1(self.l1(z))
        h2 = self.l2(h1)
        return h2

class Generator1(nn.Module):
    def __init__(self, hidden_dims):
        super(Generator1, self).__init__()
        in_dim, num_hidden, out_dim=hidden_dims
        self.conv1 = GCNConv(out_dim, num_hidden)
        self.prelu1 = nn.PReLU(num_hidden)
        self.conv2 = GATConv(num_hidden, in_dim)
    def forward(self, z,edge_index):
        h1 = self.prelu1(self.conv1(z,edge_index))
        h2 = self.conv2(h1,edge_index)
        return h2


class discriminator(nn.Module):
    def __init__(self, hidden_dims):
        super(discriminator, self).__init__()
        in_dim, num_hidden, out_dim=hidden_dims
        self.l1 = nn.Linear(in_dim, num_hidden)
        self.prelu1 = nn.PReLU(num_hidden)
        self.l2 = nn.Linear( num_hidden, num_hidden)
        self.prelu2 = nn.PReLU(num_hidden)
        self.l3 = nn.Linear(num_hidden, 1)
        #self.l2 = nn.Linear(num_hidden, 1024)

    def forward(self, z):
        h1 = self.prelu1(self.l1(z))
        h2 = self.prelu2(self.l2(h1))
        h3 = self.l3(h2)
        return torch.clamp(h3, min=-50.0, max=50.0)

class PROST_NN_sparse(nn.Module):
    def __init__(self, hidden_dims,mu):
        super(PROST_NN_sparse, self).__init__()
        self.embedding_size = hidden_dims[-1]
        self.beta=0.5
        self.gal=GCN(hidden_dims)
        self.mu=mu

    def get_q(self, z):
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1)-self.mu)**2, dim=2) / self.beta) + 1e-8)
        q = q**(self.beta+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def KL_div(self, p, q):
        loss = torch.mean(torch.sum(p*torch.log(p/(q+1e-6)), dim=1))
        return loss
  
    def forward(self, x, adj):
        z = self.gal(x, adj)
        q = self.get_q(z)
        return z, q

