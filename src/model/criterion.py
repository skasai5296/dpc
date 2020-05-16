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


class DPCClassificationLoss(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, in_seq, out_seq, mask_idx):
        """
        Compute loss (dot product over feature dimension)
        Args:
            in_seq, out_seq: torch.Tensor (B, S, D)
            mask_idx: torch.Tensor (B, S), torch.bool
        Returns:
            loss: torch.Tensor, sum of all losses
            losses: loss dict
        """
        B, S = mask_idx.size()
        for input, output, mask in zip(in_seq, out_seq, mask_idx):
            print(input)
            print(output)
            print(mask)
            sys.exit(0)

        in_masked = in_seq * mask_idx
        out_masked = out_seq * mask_idx
        loss = self.criterion(in_masked, out_masked)
        return loss, {"XELoss": loss.item(), "Accuracy (%)": acc}
