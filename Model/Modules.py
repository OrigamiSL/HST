import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Attn_PatchLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_PatchLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys):
        B, V, P, D = queries.shape
        _, _, S, D = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)

        scores = torch.einsum("bvpd,bvsd->bvps", queries, keys)  # [B V P P]
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B V P P]
        out = torch.einsum("bvps,bvsd->bvpd", attn, keys)  # [B V P D]
        return self.out_projection(out)  # [B V P D]


class Attn_VarLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Attn_VarLevel, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, var_ccc):
        B, P, V, D = queries.shape
        _, _, R, _ = keys.shape
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries).unsqueeze(-2)  # B, P, V, 1, D
        keys = self.kv_projection(keys)

        B, V, N = var_ccc.shape
        var_ccc = var_ccc.contiguous().view(B, -1).unsqueeze(1).unsqueeze(-1).expand(B, P, V * N, D)
        keys_ccc = torch.gather(keys[:, -P:], dim=2, index=var_ccc)  # B, P, VN, D
        values_ccc = keys_ccc = keys_ccc.contiguous().view(B, P, V, N, D)

        scores = torch.einsum("bpvmd,bpvnd->bpvmn", queries, keys_ccc)  # [B P V 1 N]

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bpvmn,bpvnd->bpvmd", attn, values_ccc).squeeze(-2)  # [B P V D]

        return self.out_projection(torch.cat([keys[:, :-P], out], dim=1))  # [B P V D]


class Encoder_Cross(nn.Module):
    def __init__(self, input_len, period, partial_temporal, d_model, dropout=0.1, device=None, cross=False):
        super(Encoder_Cross, self).__init__()
        self.cross = cross
        self.patch_dim = d_model
        self.input_len = input_len
        self.period = period
        self.device = device
        self.attn1 = Attn_PatchLevel(self.patch_dim, dropout)
        self.poollinear = nn.Sequential(nn.Linear(input_len // period, 1))
        if self.cross:
            self.attn2 = Attn_VarLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm0 = nn.LayerNorm(self.patch_dim)
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        if self.cross:
            self.norm3 = nn.LayerNorm(self.patch_dim)
            self.norm4 = nn.LayerNorm(self.patch_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 2 * self.patch_dim)
        self.linear2 = nn.Linear(2 * self.patch_dim, self.patch_dim)
        if self.cross:
            self.linear3 = nn.Linear(self.patch_dim, 2 * self.patch_dim)
            self.linear4 = nn.Linear(2 * self.patch_dim, self.patch_dim)
        self.partial_temporal = partial_temporal

    def forward(self, x, var_ccc):
        B, V, P, D = x.shape
        if self.period == self.input_len:
            attn1_x = self.attn1(x, x)
        else:
            x_period = x.contiguous().view(B, V, -1, self.period, D).transpose(2, 4)
            x_period = self.poollinear(x_period).squeeze(-1)
            x_period = self.norm0(x_period.transpose(-1, -2))
            attn1_x = self.attn1(x, x_period)

        y = x = self.norm1(x + self.dropout(attn1_x))
        y = self.activation(self.linear1(y))
        y = self.linear2(y)
        x = x + self.dropout(y)  # B, V, P, D

        if self.cross:
            x = self.norm2(x).permute(0, 2, 1, 3)  # B, P, V, D
            attn2_x = self.dropout(self.attn2(x[:, -P // self.partial_temporal:], x, var_ccc))
            z = x = self.norm3(x + attn2_x)
            z = self.activation(self.linear3(z))
            x = x + self.dropout(self.linear4(z))

            x_next = self.norm4(x).permute(0, 2, 1, 3)  # B, V, P, D
        else:
            x_next = self.norm2(x)

        return x_next
