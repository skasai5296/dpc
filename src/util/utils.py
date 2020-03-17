import csv
import glob
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ModelSaver:
    """Saves and Loads model and optimizer parameters"""

    def __init__(self, savedir, init_val=-1e9):
        self.savedir = savedir
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        self.best = init_val
        self.epoch = 1

    def load_ckpt(self, model, optimizer=None):
        try:
            path = sorted(glob.glob(os.path.join(self.savedir, "*.ckpt")))[-1]
        except IndexError:
            print("no checkpoint, not loading")
            return
        if os.path.exists(path):
            print(f"loading model from {path}")
            ckpt = torch.load(path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            if optimizer is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            self.best = ckpt["bestscore"]
            self.epoch = ckpt["epoch"] + 1
            print(
                f"best score is set to {self.best}, restarting from epoch "
                + f"{self.epoch}"
            )
        else:
            print(f"{path} does not exist, not loading")

    def save_ckpt_if_best(self, model, optimizer, metric, delete_prev=False):
        if delete_prev:
            for file in glob.glob(os.path.join(self.savedir, "*.ckpt")):
                os.remove(file)
        path = os.path.join(self.savedir, f"ep{self.epoch:03d}.ckpt")
        if metric > self.best:
            print(
                f"score {metric} is better than previous best score of "
                + f"{self.best}, saving to {path}"
            )
            self.best = metric
            ckpt = {"optimizer": optimizer.state_dict()}
            if hasattr(model, "module"):
                ckpt["model"] = model.module.state_dict()
            else:
                ckpt["model"] = model.state_dict()
            ckpt["epoch"] = self.epoch
            ckpt["bestscore"] = self.best
            torch.save(ckpt, path)
        else:
            print(
                f"score {metric} is not better than previous best score of "
                + f"{self.best}, not saving"
            )
        self.epoch += 1


class Timer:
    """Computes and stores the time"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.begin = time.time()

    def __str__(self):
        return sec2str(time.time() - self.begin)


class Logger:
    def __init__(self, path, header):
        path = Path(path)
        self.log_file = path.open("w")
        self.logger = csv.writer(self.log_file, delimiter="\t")

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class NPairLoss(nn.Module):
    def __init__(self, margin=0.2, method="max", scale=1e5):
        super(NPairLoss, self).__init__()
        self.margin = margin
        self.method = method
        self.scale = scale

    # x1, x2 : (n_samples, dim)
    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        assert self.method in ["max", "sum", "top10", "top25"]
        n_samples = x1.size(0)
        # sim_mat : (n_samples, n_samples)
        sim_mat = x1.mm(x2.t())
        # pos : (n_samples, 1)
        pos = sim_mat.diag().view(-1, 1)
        # positive1, 2 : (n_samples, n_samples)
        positive1 = pos.expand_as(sim_mat)
        positive2 = pos.t().expand_as(sim_mat)

        # mask for diagonals
        mask = (torch.eye(n_samples) > 0.5).to(sim_mat.device)
        # caption negatives
        lossmat_i = (
            (self.margin + sim_mat - positive1).clamp(min=0).masked_fill(mask, 0)
        )
        # image negatives
        lossmat_c = (
            (self.margin + sim_mat - positive2).clamp(min=0).masked_fill(mask, 0)
        )
        # max of hinges loss
        if self.method == "max":
            # lossmat : (n_samples)
            lossmat_i = lossmat_i.max(dim=1)[0]
            lossmat_c = lossmat_c.max(dim=0)[0]
        # sum of hinges loss
        elif self.method == "sum":
            lossmat_i /= n_samples - 1
            lossmat_c /= n_samples - 1
        elif self.method == "top10":
            lossmat_i = lossmat_i.topk(10, dim=1)[0] / 10
            lossmat_c = lossmat_c.topk(10, dim=1)[0] / 10
        elif self.method == "top25":
            lossmat_i = lossmat_i.topk(25, dim=1)[0] / 25
            lossmat_c = lossmat_c.topk(25, dim=1)[0] / 25

        loss = lossmat_i.sum() + lossmat_c.sum()

        return loss / n_samples


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.Embedding):
        init.uniform_(m.weight.data)


def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)
