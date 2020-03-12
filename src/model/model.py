import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import Dataset
from torchvision.models.video import r3d_18


class ClipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = r3d_18(pretrained=False)
        modules = list(self.model.children())[:-2]
        self.model = nn.Sequential(*modules)

    # x : (B, N, C, T, H, W)
    # out : (B, N, 512, H', W')
    def forward(self, x):
        B, N, *_ = x.size()
        x = x.reshape(B * N, *_)
        out = self.model(x)
        BN, *out_sizes = out.size()
        return self.model(x).reshape(B, N, *out_sizes).mean(3)


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = int(kernel_size / 2)

        self.reset_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.update_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )
        self.out_gate = nn.Conv2d(
            input_size + hidden_size, hidden_size, kernel_size, padding=padding
        )

    # x : (B, input_size, H', W')
    # h : (B, hidden_size, H', W')
    def forward(self, x, h):
        if h is None:
            B, C, *spatial_dim = x.size()
            h = torch.zeros([B, self.hidden_size, *spatial_dim]).to(x.device)
        input = torch.cat([x, h], dim=1)
        update = torch.sigmoid(self.update_gate(input))
        reset = torch.sigmoid(self.reset_gate(input))
        out = torch.tanh(self.out_gate(torch.cat([x, h * reset], dim=1)))
        new_state = h * (1 - update) + out * update
        return new_state


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            input_dim = self.input_size if i == 0 else self.hidden_size
            cell = ConvGRUCell(input_dim, self.hidden_size, self.kernel_size)
            name = "ConvGRUCell_{:02d}".format(i)
            setattr(self, name, cell)
            cell_list.append(getattr(self, name))

        self.cell_list = nn.ModuleList(cell_list)
        self.dropout = nn.Dropout(p=dropout)

    # x : (B, T, input_size, H, W)
    # layer_output : (B, T, hidden_size, H, W)
    # last_state_list : (B, num_layers, hidden_size, H, W)
    def forward(self, x, h=None):
        B, T, *_ = x.size()
        if h is None:
            h = [None] * self.num_layers
        current_layer_input = x
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = h[idx]
            output_inner = []
            for t in range(T):
                cell_hidden = self.cell_list[idx](
                    current_layer_input[:, t, ...], cell_hidden
                )
                cell_hidden = self.dropout(cell_hidden)
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output
            last_state_list.append(cell_hidden)
        last_state_list = torch.stack(last_state_list, dim=1)
        return layer_output, last_state_list


class DPC(nn.Module):
    def __init__(
        self, input_size, hidden_size, kernel_size, num_layers, pred_step=3, dropout=0.1
    ):
        super().__init__()
        self.cnn = ClipEncoder()
        self.rnn = ConvGRU(input_size, hidden_size, kernel_size, num_layers, dropout)

    # x: (B, num_clips, C, clip_len, H, W)
    # out : (B, N, hidden_size, H, W)
    def forward(self, x):
        # out : (B, N, hidden_size, H', W')
        out = self.cnn(x)
        out, last = self.rnn(out)
        return out, last
