import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn


class DPCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        """
        Compute loss (dot product over feature dimension)
        Args:
            pred, gt: torch.Tensor (B, N, C, H, W)
        Returns:
            loss: torch.Tensor, sum of all losses
            losses: loss dict
        """
        assert pred.size() == gt.size()
        B, N, C, H, W = pred.size()
        pred = pred.permute(0, 1, 3, 4, 2).reshape(B * N * H * W, -1)
        gt = gt.permute(2, 0, 1, 3, 4).reshape(-1, B * N * H * W)
        # lossmat: (BNHW, BNHW)
        lossmat = torch.matmul(pred, gt)
        # target: (BNHW)
        target = torch.arange(B * N * H * W, dtype=torch.long, device=pred.device)
        loss = self.criterion(lossmat, target)

        with torch.no_grad():
            # top1: (BNHW)
            top1 = lossmat.argmax(1)
            acc = torch.eq(top1, target).sum().item() / top1.size(0) * 100
        return loss, {"XELoss": loss.item(), "Accuracy (%)": acc}


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, gt):
        """
        Compute loss (dot product over feature dimension)
        Args:
            pred: torch.Tensor (B, num_classes)
            gt: torch.Tensor (B), torch.long
        Returns:
            loss: torch.Tensor, sum of all losses
            losses: loss dict
        """
        loss = self.criterion(pred, gt)

        with torch.no_grad():
            # top1: (BNHW)
            top1 = pred.argmax(1)
            acc = torch.eq(top1, gt).sum().item() / top1.size(0) * 100
        return loss, {"XELoss": loss.item(), "Accuracy (%)": acc}


class BERTCPCLoss(nn.Module):
    def __init__(self, mse_weight):
        super().__init__()
        self.xeloss = nn.CrossEntropyLoss()
        self.mse_weight = mse_weight
        self.mseloss = nn.MSELoss()

    def forward(self, in_seq, out_seq, drop_idx, keep_idx):
        """
        Compute loss (dot product over feature dimension)
        Args:
            in_seq, out_seq: torch.Tensor (B, S, D)
            drop_idx: torch.Tensor (B, dropnum), torch.long
            keep_idx: torch.Tensor (B, S-dropnum), torch.long
        Returns:
            loss: torch.Tensor, sum of all losses
            losses: loss dict
        """
        B, S, D = in_seq.size()
        B, dropnum = drop_idx.size()
        predictions = torch.empty(B, dropnum, D, device=in_seq.device)
        inputs = torch.empty(B, S - dropnum, D, device=in_seq.device)
        reconstructions = torch.empty(B, S - dropnum, D, device=in_seq.device)
        # target: (B, dropnum)
        target = torch.empty(B, dropnum, dtype=torch.long, device=in_seq.device)
        for i, (inp, out, drop, keep) in enumerate(zip(in_seq, out_seq, drop_idx, keep_idx)):
            # pred: (dropnum, D)
            pred = out.index_select(0, drop)
            predictions[i] = pred
            # orig, recon: (S-dropnum, D)
            orig = inp.index_select(0, keep)
            inputs[i] = orig
            recon = out.index_select(0, keep)
            reconstructions[i] = recon
            target[i] = int(i * S) + drop
        # target: (B * dropnum)
        target = target.flatten()
        outputs = predictions.reshape(B * dropnum, -1)
        in_seq = in_seq.permute(2, 0, 1).reshape(-1, B * S)
        # lossmat: (B * dropnum, B * S)
        lossmat = torch.matmul(outputs, in_seq)
        with torch.no_grad():
            # top1: (B * dropnum)
            top1 = lossmat.argmax(1)
            acc = torch.eq(top1, target).sum().item() / top1.size(0) * 100
        xe = self.xeloss(lossmat, target)
        mse = self.mseloss(inputs, reconstructions)

        return xe + mse, {"XELoss": xe.item(), "MSELoss": mse.item(), "Accuracy (%)": acc}
