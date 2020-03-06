import sys, os
import time
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset
import torchvision

from utils import sec2str, weight_init
from dataset.msrvtt import MSR_VTT_Resnet

# L2 normalize a batched tensor (bs, ft)
def l2normalize(ten, dim=1):
    norm = torch.norm(ten, dim, keepdim=True)
    return ten / norm


class MotionEncoder(nn.Module):
    def __init__(self, ft_size=2048, out_size=256):
        super(MotionEncoder, self).__init__()
        self.out_size = out_size
        # 1x1x1 conv for feature affine transformation
        self.layer = nn.Conv3d(ft_size, out_size, (1, 1, 1))
        self.layer2 = nn.Conv3d(out_size, out_size, (1, 1, 1))
        #self.layer2 = nn.Conv3d(out_size, out_size, (10, 1, 1))
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.layer.apply(weight_init)
        self.layer2.apply(weight_init)

    # x : (bs, C_fast, T_fast, H, W)
    def forward(self, x):
        bs = x.size(0)
        middle = self.layer(x)
        # ft_map : (bs, out_size, T, H, W)
        ft_map = l2normalize(middle)
        # out : (bs, out_size)
        out = l2normalize(self.pool(self.layer2(middle)).view(bs, self.out_size))
        return ft_map, out

class SemanticEncoder(nn.Module):
    def __init__(self, ft_size=2048, out_size=256):
        super(SemanticEncoder, self).__init__()
        self.out_size = out_size
        # 1x1x1 conv for feature affine transformation
        self.layer = nn.Conv3d(ft_size, out_size, (1, 1, 1))
        self.layer2 = nn.Conv3d(out_size, out_size, (1, 1, 1))
        #self.layer2 = nn.Conv3d(out_size, out_size, (1, 3, 3))
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.layer.apply(weight_init)

    # x : (bs, C_slow, T_slow, H, W)
    def forward(self, x):
        bs = x.size(0)
        middle = self.layer(x)
        # ft_map : (bs, out_size, T, H, W)
        ft_map = l2normalize(middle)
        # out : (bs, out_size)
        out = l2normalize(self.pool(self.layer2(middle)).view(bs, self.out_size))
        return ft_map, out

class MSE(nn.Module):
    def __init__(self, ft_size_slow=2048, ft_size_fast=256, out_size=256):
        super(MSE, self).__init__()
        self.out_size = out_size
        # 1x1x1 conv for feature affine transformation
        self.enc_sem = SemanticEncoder(ft_size_slow, out_size)
        self.enc_mot = MotionEncoder(ft_size_fast, out_size)
        self.layer = nn.Linear(out_size*2, out_size)
        self.shortcut = nn.Linear(ft_size_slow+ft_size_fast, out_size)
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.layer.apply(weight_init)

    # slow : (bs, ft_size, T_slow, H, W)
    # fast : (bs, ft_size, T_fast, H, W)
    def forward(self, slow, fast):
        bs = slow.size(0)
        # slowmap : (bs, out_size, T_slow, H, W)
        # fastmap : (bs, out_size, T_fast, H, W)
        # slowft, fastft : (bs, out_size)
        slowmap, slowft = self.enc_sem(slow)
        fastmap, fastft = self.enc_mot(fast)
        # jointft : (bs, out_size)
        jointft = self.layer(torch.cat([slowft, fastft], dim=1))
        # short_out : (bs, out_size)
        short_out = self.shortcut(
                        torch.cat(
                            [self.pool(slow).squeeze(-1).squeeze(-1).squeeze(-1), self.pool(fast).squeeze(-1).squeeze(-1).squeeze(-1)],
                            dim=1
                        )
                    )
        # out : (bs, out_size)
        out = l2normalize(jointft+short_out)
        return (slowmap, fastmap), out

class CaptionEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size=256, out_size=256, rnn_type="GRU", padidx=0):
        super(CaptionEncoder, self).__init__()
        self.padidx = padidx
        self.out_size = out_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = getattr(nn, rnn_type)(emb_size, out_size, batch_first=True)

        self.emb_ent = nn.Embedding(vocab_size, emb_size)
        self.rnn_ent = getattr(nn, rnn_type + "Cell")(emb_size, out_size)

        self.emb.apply(weight_init)
        self.rnn.apply(weight_init)
        self.emb_ent.apply(weight_init)
        self.rnn_ent.apply(weight_init)

    # x: (bs, seq)
    def forward(self, x, caption=True):
        if caption:
            # lengths: (bs)
            lengths = x.ne(self.padidx).sum(dim=1)
            emb = self.emb(x)
            # packed: PackedSequence of (bs, seq, emb_size)
            packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.rnn(packed)
            # output: (bs, seq, out_size)
            output = pad_packed_sequence(output, batch_first=True, padding_value=self.padidx)[0]
            lengths = lengths.view(-1, 1, 1).expand(-1, -1, self.out_size) - 1
            out = torch.gather(output, 1, lengths).squeeze(1)
        else:
            x = x[:, 1]
            # emb: (bs, emb_size)
            emb = self.emb_ent(x)
            # out: (bs, out_size)
            out = self.rnn_ent(emb)
        # normed_out: (bs, out_size)
        normed_out = l2normalize(out)
        return normed_out

# Discriminator for adversarial loss
class ModalDiscriminator(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, 32)
        self.layer2 = nn.Linear(32, 1)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

        self.layer1.apply(weight_init)
        self.layer2.apply(weight_init)

    # x: (bs, C)
    # returns: (bs)
    def forward(self, x):
        out = self.act1(self.layer1(x))
        return self.act2(self.layer2(out).squeeze(-1))



if __name__ == '__main__':

    cnn = MSE()

    ds = MSR_VTT_Resnet("/groups1/gaa50131/datasets/MSR-VTT/", mode="train")
    for i in range(10):
        enc_im = cnn(ds[i]["feature"].unsqueeze(0))
        print(enc_im.size())

        cap = CaptionEncoder(vocab_size=100)
        seq = torch.randint(100, (1, 30), dtype=torch.long)
        seq2 = torch.randint(100, (1, 30), dtype=torch.long)
        len = torch.randint(1, 31, (1,), dtype=torch.long)
        enc_pos = cap(seq, len)
        enc_neg = cap(seq2, len)
        print(enc_pos.size())


        print(mask.size())
        print(loss)



