# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from Model.Modules import *
from Model.embed import DataEmbedding
from utils.RevIN import RevIN


class Encoder_process(nn.Module):
    def __init__(self, patch_size, d_model, encoder_num, layer_stack, encoders1, encoders2):
        super(Encoder_process, self).__init__()
        self.patch_size = patch_size
        self.layer_stack = layer_stack - 1
        self.encoder_num = encoder_num
        self.encoders1 = encoders1
        self.encoders2 = encoders2
        self.linear1 = nn.ModuleList([nn.Linear(d_model, patch_size * (2 ** i)) for i in range(encoder_num)])
        self.linear2 = nn.ModuleList([nn.Linear(patch_size * (2 ** i), d_model) for i in range(encoder_num - 1)])
        self.linear3 = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(encoder_num - 1)])
        self.F = nn.Flatten(start_dim=2)

    def forward(self, x_enc, var_ccc, revin):
        encoder_out_list = []
        for i in range(self.encoder_num):
            B, V, P, D = x_enc.shape
            if self.encoders1 is not None:
                for j in range(self.layer_stack):
                    x_enc = self.encoders1[i * self.layer_stack + j](x_enc, var_ccc[:, i])
            x_enc = self.encoders2[i](x_enc, var_ccc[:, i])

            x_current = self.F(self.linear1[i](x_enc))
            encoder_out_list.append(x_current)
            if i != self.encoder_num - 1:
                x_current = revin(x_current.transpose(1, 2), 'norm').transpose(1, 2)
                x_enc = self.linear2[i](x_current.contiguous().view(B, V, P, -1))
                x_enc = x_enc.contiguous().view(B, V, P // 2, 2, D) \
                    .view(B, V, P // 2, 2 * D)
                x_enc = self.linear3[i](x_enc)

        return encoder_out_list


class Encoder_map(nn.Module):
    def __init__(self, input_len, period, partial_temporal, encoder_layer=3, layer_stack=4, patch_size=12,
                 d_model=4, dropout=0.05, device=None):
        super(Encoder_map, self).__init__()
        self.input_len = input_len
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model

        self.Embed = DataEmbedding(patch_size, d_model)
        if layer_stack:
            self.encoders1 = []
            for i in range(encoder_layer):
                self.encoders1 += [Encoder_Cross(input_len // (patch_size * 2 ** i), period[i] // (patch_size * 2 ** i),
                                                 partial_temporal, d_model, dropout, device, cross=False)
                                   for _ in range(layer_stack - 1)]
            self.encoders1 = nn.ModuleList(self.encoders1)
        else:
            self.encoders1 = None
        self.encoders2 = [Encoder_Cross(input_len // (patch_size * 2 ** i), period[i] // (patch_size * 2 ** i),
                                        partial_temporal, d_model, dropout, device, cross=True)
                          for i in range(encoder_layer)]
        self.encoders2 = nn.ModuleList(self.encoders2)
        self.encoder_process = Encoder_process(patch_size, d_model, self.encoder_num, layer_stack,
                                               self.encoders1, self.encoders2)

    def forward(self, x, var_ccc, revin):
        B, L, V = x.shape
        x = x.contiguous().view(B, -1, self.patch_size, V)
        x_enc = self.Embed(x)  # B, V, P, D

        encoder_out_list = self.encoder_process(x_enc, var_ccc, revin)
        return encoder_out_list


class HST(nn.Module):
    def __init__(self, input_len, pred_len, period, encoder_layer, layer_stack, patch_size, d_model, dropout, device):
        super(HST, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.encoder_num = encoder_layer
        self.d_model = d_model
        self.layer_stack = layer_stack
        self.partial_temporal = input_len // period[1]

        self.revin = RevIN()
        self.Encoder_process = (
            Encoder_map(input_len, period, self.partial_temporal, encoder_layer,
                        layer_stack, patch_size, d_model, dropout, device))

        self.linear = nn.Sequential(
            nn.Linear(self.input_len * encoder_layer, 2 * self.pred_len),
            nn.Linear(2 * self.pred_len, self.pred_len)
        )

    def forward(self, x, var_ccc):
        B, _, V = x.shape
        self.revin(x, 'stats')
        x_enc = self.revin(x, 'norm')  # [B L V]
        enc_list = self.Encoder_process(x_enc, var_ccc, self.revin)
        enc_map = torch.cat(enc_list, dim=-1)  # B, V, Nenc * Lin
        x_out = self.linear(enc_map).transpose(1, 2)
        x_out = self.revin(x_out, 'denorm')
        return x_out
