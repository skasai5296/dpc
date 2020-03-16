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
        self.loss = nn.CrossEntropyLoss()

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
        out = self.loss(lossmat, target)
        return out, {"XELoss": out.item()}
