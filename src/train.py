import argparse
import hashlib
import os
import random
import subprocess
from pprint import pprint

import numpy as np
import torch
import yaml
from addict import Dict
from torch import nn

import wandb
from dataset.helper import get_dataloader
from model.helper import get_model_and_loss, get_optimizer_and_scheduler
from util.utils import AverageMeter, ModelSaver, Timer


def train_epoch(loader, model, optimizer, criterion, device, CONFIG, epoch):
    train_timer = Timer()
    metrics = [AverageMeter("XELoss"), AverageMeter("MSELoss"), AverageMeter("Accuracy (%)")]
    if CONFIG.model in ("DPC"):
        metrics.pop(1)
    model.train()
    for it, data in enumerate(loader):

        clip = data["clip"].to(device)
        if it == 1 and torch.cuda.is_available():
            subprocess.run(["nvidia-smi"])

        optimizer.zero_grad()
        output = model(clip)
        loss, lossdict = criterion(*output)

        for metric in metrics:
            metric.update(lossdict[metric.name])
        loss.backward()
        if CONFIG.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.grad_clip)
        optimizer.step()
        if it % 10 == 9:
            metricstr = " | ".join([f"train {metric}" for metric in metrics])
            print(
                f"epoch {epoch:03d}/{CONFIG.max_epoch:03d} | train | "
                f"{train_timer} | iter {it+1:06d}/{len(loader):06d} | "
                f"{metricstr}",
                flush=True,
            )
            if CONFIG.use_wandb:
                for metric in metrics:
                    wandb.log({f"train {metric.name}": metric.avg}, commit=False)
                wandb.log({"iteration": it + epoch * len(loader)})
            for metric in metrics:
                metric.reset()


def validate(loader, model, criterion, device, CONFIG, epoch):
    val_timer = Timer()
    metrics = [AverageMeter("XELoss"), AverageMeter("MSELoss"), AverageMeter("Accuracy (%)")]
    global_metrics = [AverageMeter("XELoss"), AverageMeter("MSELoss"), AverageMeter("Accuracy (%)")]
    if CONFIG.model in ("DPC"):
        metrics.pop(1)
        global_metrics.pop(1)
    model.eval()
    for it, data in enumerate(loader):

        clip = data["clip"].to(device)
        if it == 1 and torch.cuda.is_available():
            subprocess.run(["nvidia-smi"])

        with torch.no_grad():
            output = model(clip)
            loss, lossdict = criterion(*output)

        for metric in metrics:
            metric.update(lossdict[metric.name])
        for metric in global_metrics:
            metric.update(lossdict[metric.name])
        if it % 10 == 9:
            metricstr = " | ".join([f"validation {metric}" for metric in metrics])
            print(
                f"epoch {epoch:03d}/{CONFIG.max_epoch:03d} | valid | "
                f"{val_timer} | iter {it+1:06d}/{len(loader):06d} | "
                f"{metricstr}",
                flush=True,
            )
            for metric in metrics:
                metric.reset()
        # validating for 100 steps is enough
        if it == 100:
            break
    if CONFIG.use_wandb:
        for metric in global_metrics:
            wandb.log({f"epoch {metric.name}": metric.avg}, commit=False)
    return global_metrics[-1].avg


if __name__ == "__main__":
    global_timer = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="cfg/default.yml", help="path to configuration yml file",
    )
    parser.add_argument(
        "--resume", action="store_true", help="denotes if to continue training, will use config",
    )
    opt = parser.parse_args()
    print(f"loading configuration from {opt.config}")
    CONFIG = Dict(yaml.safe_load(open(opt.config)))
    print("CONFIGURATIONS:")
    pprint(CONFIG)

    """  Set Random Seeds  """
    if CONFIG.seed >= 0:
        random.seed(CONFIG.seed)
        np.random.seed(CONFIG.seed)
        torch.manual_seed(CONFIG.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if CONFIG.use_wandb:
        idhash = hashlib.sha256(CONFIG.config_name.encode()).hexdigest()
        wandb.init(
            name=CONFIG.config_name,
            id=idhash,
            resume=idhash,
            config=CONFIG,
            project=CONFIG.project_name,
        )

    """  Model Components  """
    model, criterion = get_model_and_loss(CONFIG, finetune=False)
    if CONFIG.use_wandb:
        wandb.watch(model)
    optimizer, scheduler = get_optimizer_and_scheduler(CONFIG, model)

    """  Load from Checkpoint  """
    saver = ModelSaver(os.path.join(CONFIG.outpath, CONFIG.config_name))
    if opt.resume:
        saver.load_ckpt(model, optimizer, scheduler)

    """  Devices  """
    gpu_ids = list(map(str, CONFIG.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU numbers {}".format(CONFIG.gpu_ids))
    else:
        device = torch.device("cpu")
        print("using CPU")
    model = model.to(device)
    # for sending pretrained weights to GPU for optimizer
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        subprocess.run(["nvidia-smi"])

    """  Dataset  """
    train_dl, val_dl = get_dataloader(CONFIG)

    """  Training Loop  """
    for ep in range(saver.epoch, CONFIG.max_epoch + 1):
        print(f"global time {global_timer} | start training epoch {ep}")
        train_epoch(train_dl, model, optimizer, criterion, device, CONFIG, ep)
        print(f"global time {global_timer} | start validation epoch {ep}")
        val_acc = validate(val_dl, model, criterion, device, CONFIG, ep)
        if CONFIG.use_wandb:
            wandb.log(
                {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "iteration": len(train_dl) * (ep + 1),
                }
            )
        scheduler.step(val_acc)
        saver.save_ckpt_if_best(
            model, optimizer, scheduler, val_acc, delete_prev=CONFIG.only_best_checkpoint,
        )
        print(f"global time {global_timer} | val accuracy: {val_acc:.5f}% | " f"end epoch {ep}")
