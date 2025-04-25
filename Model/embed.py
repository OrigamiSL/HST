import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, start_pos=0):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.start_pos = start_pos

    def forward(self, x):
        pos = self.pe[:,  self.start_pos: self.start_pos + x.size(1)].unsqueeze(2)
        return pos  # B, P, V, D


class TokenEmbedding(nn.Module):
    def __init__(self, patch, d_model):
        super(TokenEmbedding, self).__init__()
        self.embed = nn.Linear(patch, d_model)

    def forward(self, x):
        x = self.embed(x.permute(0, 1, 3, 2))
        return x


class DataEmbedding(nn.Module):
    def __init__(self, patch, d_model, start_pos=0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(patch=patch, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, start_pos=start_pos)

    def forward(self, x):  # B, P, D, V
        pos = self.position_embedding(x)
        x = self.value_embedding(x) + pos
        return x.transpose(1, 2)
