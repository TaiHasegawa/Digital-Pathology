import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import EGNNCLayer

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)


class EGNNC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, channel_dim, dropout=0.5, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layer. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(EGNNC, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.channel_dim = channel_dim
        self.dropout = dropout
        self.device = device

        self.elu = nn.ELU()
        self.egnn1 = EGNNCLayer(input_dim, hidden_dim, channel_dim, device=device)
        self.egnn2 = EGNNCLayer(hidden_dim*channel_dim, output_dim, channel_dim, device=device)

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
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """
        x = self.egnn1(features, edge_features)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.egnn2(x, edge_features)
        return x


class MLPTwoLayers(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.5, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dim : int
            Dimension of hidden layer. Must be non empty.
        output_dim : int
            Dimension of output node features.
        dropout : float
            Probability of setting an element to 0 in dropout layer. Default: 0.5.
        device : string
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(MLPTwoLayers, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device

        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True).to(device)
        self.linear2 = nn.Linear(hidden_dim, output_dim, bias=True).to(device)

    def forward(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        Returns
        -------
        out: torch.Tensor
            Output of two layer MLPs
        """
        x = F.relu(self.linear1(features))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear2(x)
        out = x.reshape(-1)
        return out