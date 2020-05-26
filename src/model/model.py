import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torchvision.models.video import r3d_18


class ClipEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = r3d_18(pretrained=False)
        modules = list(self.model.children())[:-2]
        self.model = nn.Sequential(*modules)

    # x : (B, C, T, H, W)
    # out : (B, 512, H', W')
    def forward(self, x):
        out = self.model(x)
        return out.mean(2)


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
            h = torch.zeros([B, self.hidden_size, *spatial_dim], device=x.device)
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
                cell_hidden = self.cell_list[idx](current_layer_input[:, t, ...], cell_hidden)
                cell_hidden = self.dropout(cell_hidden)
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output
            last_state_list.append(cell_hidden)
        last_state_list = torch.stack(last_state_list, dim=1)
        return layer_output, last_state_list


class DPC(nn.Module):
    def __init__(
        self, input_size, hidden_size, kernel_size, num_layers, n_clip, pred_step, dropout,
    ):
        super().__init__()
        self.n_clip = n_clip
        self.pred_step = pred_step
        self.cnn = ClipEncoder()
        self.rnn = ConvGRU(input_size, hidden_size, kernel_size, num_layers, dropout)
        self.network_pred = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, padding=0),
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, flag="full"):
        if flag == "full":
            return self._full_pass(x)
        elif flag == "extract":
            return self._extract_features(x)

    # x: (B, num_clips, C, clip_len, H, W)
    # pred, out : (B, N, hidden_size, H, W)
    def _full_pass(self, x):
        B, N, *sizes = x.size()
        x = x.view(B * N, *sizes)
        # out : (B * N, hidden_size, H', W')
        out = self.cnn(x)
        _, D, H, W = out.size()
        # out : (B, N, hidden_size, H', W')
        out = out.view(B, N, D, H, W)

        # hidden: (B, hidden_size, H', W')
        _, hidden = self.rnn(out[:, : self.n_clip - self.pred_step, ...])
        hidden = hidden[:, -1, ...]
        pred = []
        for step in range(self.pred_step):
            # predicted: (B, hidden_size, H', W')
            predicted = self.network_pred(hidden)
            pred.append(predicted)
            _, hidden = self.rnn(self.relu(predicted).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, ...]
        # pred: (B, pred_step, hidden_size, H', W')
        pred = torch.stack(pred, 1)
        return pred, out[:, self.n_clip - self.pred_step :, ...]

    # x: (B, num_clips, C, clip_len, H, W)
    # hidden : (B, hidden_size, H, W)
    def _extract_feature(self, x):
        B, N, *sizes = x.size()
        x = x.view(B * N, *sizes)
        # out : (B * N, hidden_size, H', W')
        out = self.cnn(x)
        _, D, H, W = out.size()
        # out : (B, N, hidden_size, H', W')
        out = out.view(B, N, D, H, W)
        # hidden: (B, hidden_size, H', W')
        _, hidden = self.rnn(out[:, : self.n_clip - self.pred_step, ...])
        hidden = hidden[:, -1, ...]
        return hidden


class DPCClassification(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        num_layers,
        n_clip,
        pred_step,
        dropout,
        num_classes,
    ):
        super().__init__()
        self.dpc = DPC(input_size, hidden_size, kernel_size, num_layers, n_clip, pred_step, dropout)
        self.classification = nn.Linear(hidden_size, num_classes)

    # x : (B, num_clips, C, clip_len, H, W)
    # out : (B, num_classes)
    def forward(self, x, flag="extract"):
        out = self.dpc(x, flag)
        out = self.classification(out.mean(-1).mean(-1))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class BERTCPC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, n_clip, mask_p=0.15):
        super().__init__()
        self.mask_p = mask_p
        self.hidden_size = hidden_size
        self.cnn = ClipEncoder()
        self.fc = nn.Linear(512, hidden_size)
        self.pe = PositionalEncoding(hidden_size, max_len=n_clip)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # number of tokens to mask
        self.dropnum = int(n_clip * mask_p)

    def forward(self, x, flag="full"):
        if flag == "full":
            return self._full_pass(x)
        elif flag == "extract":
            return self._extract_feature(x)

    # x: (B, num_clips, C, clip_len, H, W)
    # pred, out : (B, N, hidden_size)
    # drop_indices : (B, self.dropnum)
    def _full_pass(self, x):
        B, N, *sizes = x.size()
        x = x.view(B * N, *sizes)
        # out : (B * N, 512, H', W')
        out = self.cnn(x)
        BxN, D, *_ = out.size()
        # out : (BxN, 512)
        out = F.adaptive_avg_pool2d(out, 1).view(BxN, D)
        # out : (B, N, hidden_size)
        out = self.fc(out).view(B, N, self.hidden_size)

        # masked_out : (B, N, hidden_size)
        masked_out = torch.empty_like(out)
        # masked_out : (B, self.dropnum)
        drop_indices = torch.empty(B, self.dropnum, dtype=torch.long, device=out.device)
        for i, seq in enumerate(out):
            drop = torch.randperm(N, device=out.device)[: self.dropnum]
            drop_indices[i] = drop
            # seq: (N, hidden_size)
            masked_out[i] = seq.index_fill(0, drop, 0)

        # trans_out : (B, N, hidden_size)
        trans_out = self.transformer_encoder(self.pe(masked_out.permute(1, 0, 2))).permute(1, 0, 2)
        pred = self.predictor(self.pe(trans_out.permute(1, 0, 2))).permute(1, 0, 2)
        return out, pred, drop_indices

    # x: (B, num_clips, C, clip_len, H, W)
    def _extract_feature(self, x):
        B, N, *sizes = x.size()
        x = x.view(B * N, *sizes)
        # out : (B * N, hidden_size, H', W')
        out = self.cnn(x)
        BxN, D, *_ = out.size()
        # out : (BxN, 512)
        out = F.adaptive_avg_pool2d(out, 1).view(BxN, D)
        # out : (B, N, hidden_size)
        out = self.fc(out).view(B, N, self.hidden_size)

        # trans_out : (B, N, hidden_size)
        trans_out = self.transformer_encoder(self.pe(out.permute(1, 0, 2))).permute(1, 0, 2)
        return trans_out


class BERTCPCClassification(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_heads, n_clip, mask_p, num_classes,
    ):
        super().__init__()
        self.dpc = BERTCPC(input_size, hidden_size, num_layers, num_heads, n_clip, mask_p)
        self.classification = nn.Linear(hidden_size, num_classes)

    # x : (B, num_clips, C, clip_len, H, W)
    # out : (B, num_classes)
    def forward(self, x, flag="extract"):
        out = self.dpc(x, flag)
        # average over temporal dimension
        out = self.classification(out.mean(1))
        return out


class SpatialPositionalEncoding(nn.Module):
    def __init__(
        self, d_model,
    ):
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    # B N C T H W
    a = torch.randn(2, 8, 3, 5, 112, 112)
    model = BERTCPC(512, 128, 1, 8, 8, 0.3)
    out, trans_out, mask = model(a)
    # B N C'
    print(out.size())
    print(trans_out.size())
    print(mask.size())
    print(mask)
