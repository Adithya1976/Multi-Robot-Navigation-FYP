import torch.nn as nn
from typing import List
import numpy as np
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True, device=device)

    def forward(self, packed_seqs):
        output, (hn, cn) = self.rnn_net(packed_seqs)
        hnv = torch.squeeze(hn)
        return hnv

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.rnn_net = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, device=device)

    def forward(self, packed_seqs):
        output, (hn, cn) = self.rnn_net(packed_seqs)
        hnv = torch.sum(hn, 0)
        return hnv

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, device=device)

    def forward(self, packed_seqs):
        output, hn = self.rnn_net(packed_seqs)
        hnv = torch.squeeze(hn)
        return hnv

class BiGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(BiGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.rnn_net = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True, device=device)

    def forward(self, packed_seqs):
        output, hn = self.rnn_net(packed_seqs)
        hnv = torch.sum(hn, 0)
        return hnv
