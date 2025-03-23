import torch
import torch.nn as nn
from dynamic_input_models.set_transformer import SetTransformer
from dynamic_input_models.deep_set import DeepSet
from dynamic_input_models.rnn import LSTM, BiLSTM, GRU, BiGRU
from typing import List
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


model_dict = {
    "LSTM": LSTM,
    "BiLSTM": BiLSTM,
    "GRU": GRU,
    "BiGRU": BiGRU,
    "SetTransformer": SetTransformer,
    "DeepSet": DeepSet
}

# create an abstract class for dynamic input models which inherit from nn.Module
class DynamicInputModel(nn.Module):
    def __init__(self, input_dim = 8, hidden_dim = 256, device = "mps", mode = "BiGRU"):
        super(DynamicInputModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.mode = mode
        
        if mode not in model_dict:
            raise ValueError("Invalid mode")
        self.model = model_dict[mode](input_dim, hidden_dim, device)


    def forward(self, exteroceptive_observations: List[np.ndarray] | np.ndarray):
        if isinstance(exteroceptive_observations, np.ndarray):
            exteroceptive_observations = [exteroceptive_observations]
        # Convert each numpy array to a torch tensor.
        # Each tensor is assumed to have shape (seq_len, input_dim).
        seqs = [torch.tensor(obs, dtype=torch.float32) for obs in exteroceptive_observations]
        # Get the lengths of each sequence.
        lengths = [seq.size(0) for seq in seqs]
        # Pad the sequences to create a batch tensor of shape (batch, max_seq_len, input_dim).
        padded_seqs = pad_sequence(seqs, batch_first=True)
        padded_seqs = padded_seqs.to(self.device)
        packed_input = pack_padded_sequence(padded_seqs, lengths, batch_first=True, enforce_sorted=False)
        
        return self.model(packed_input)