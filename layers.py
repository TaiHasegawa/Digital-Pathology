import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter



class EGNNCLayer(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, channel_dim=None, bias=True, device='cpu'):
        super(EGNNCLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_dim = channel_dim

        self.weight0 = Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight1 = Parameter(torch.FloatTensor(input_dim, output_dim))

        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim * channel_dim))
        self._reset_parameters()
    

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight0)
        torch.nn.init.xavier_uniform_(self.weight1)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
    

    def forward(self, features, edge_features):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n x input_dim) tensor of input node features.
        edge_features : torch.Tensor
            An (p x n x n) tensor of edge features.
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """
        support0 = torch.matmul(features, self.weight0)
        support1 = torch.matmul(features, self.weight1)
        x = torch.matmul(edge_features, support1) + support0
        output = torch.cat([xi for xi in x], dim=1)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.input_dim) + ' -> ' \
                + str(self.output_dim * self.channel_dim) + ')'
    