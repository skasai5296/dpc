import argparse
import os
import random
from pprint import pprint

import numpy as np
import torch
import wandb
import yaml
from addict import Dict
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.kinetics import Kinetics700, collate_fn, get_transforms
from model.criterion import ClassificationLoss
from model.model import BERTCPCClassification, DPCClassification
from util import spatial_transforms, temporal_transforms
from util.utils import AverageMeter, ModelSaver, Timer


def train_epoch(loader, model, optimizer, criterion, device, CONFIG, epoch):
    train_timer = Timer()
    metrics = [AverageMeter("XELoss"), AverageMeter("Accuracy (%)")]
    model.train()
    for it, data in enumerate(loader):
        clip = data["clip"].to(device)
        label = data["label"].to(device)

        optimizer.zero_grad()
        out = model(clip)
        loss, lossdict = criterion(out, label)

        for metric in metrics:
            metric.update(lossdict[metric.name])
        loss.backward()
        if CONFIG.finetune_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG.finetune_grad_clip)
        optimizer.step()

        if it % 100 == 99:
            metricstr = " | ".join([f"train {metric}" for metric in metrics])
            print(
                f"epoch {epoch:03d}/{CONFIG.max_epoch:03d} | train | "
                f"{train_timer} | iter {it+1:06d}/{len(loader):06d} | "
                f"{metricstr}",
                flush=True,
            )
            if CONFIG.use_wandb:
                for metric in metrics:
                    wandb.log({f"finetune train {metric.name}": metric.avg})
            for metric in metrics:
                metric.reset()


def validate(loader, model, criterion, device, CONFIG, epoch):
    val_timer = Timer()
    metrics = [AverageMeter("XELoss"), AverageMeter("Accuracy (%)")]
    global_metrics = [AverageMeter("XELoss"), AverageMeter("Accuracy (%)")]
    model.eval()
    for it, data in enumerate(loader):
        # batch size 1
        clip = data["clip"].to(device)
        label = data["label"].to(device)

        with torch.no_grad():
            # batch size 1
            out = model(clip).mean(0).unsqueeze(0)
            loss, lossdict = criterion(out, label)

        for metric in metrics:
            metric.update(lossdict[metric.name])
        for metric in global_metrics:
            metric.update(lossdict[metric.name])
        if it % 100 == 99:
            metricstr = " | ".join([f"validation {metric}" for metric in metrics])
            print(
                f"epoch {epoch:03d}/{CONFIG.max_epoch:03d} | valid | "
                f"{val_timer} | iter {it+1:06d}/{len(loader):06d} | "
                f"{metricstr}",
                flush=True,
            )
            for metric in metrics:
                metric.reset()
        # validating for 100 samples is enough
        if it == 1000:
            break
    if CONFIG.use_wandb:
        for metric in global_metrics:
            wandb.log({f"finetune epoch {metric.name}": metric.avg})
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
        # torch.manual_seed_all(CONFIG.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if CONFIG.use_wandb:
        wandb.init(
            name=f"{CONFIG.config_name}_finetune", config=CONFIG, project=CONFIG.project_name,
        )

    """  Model Components  """
    if CONFIG.bert:
        model = BERTCPCClassification(
            CONFIG.input_size,
            CONFIG.hidden_size,
            CONFIG.num_layers,
            CONFIG.num_heads,
            CONFIG.n_clip,
            700,
        )
    else:
        model = DPCClassification(
            CONFIG.input_size,
            CONFIG.hidden_size,
            CONFIG.kernel_size,
            CONFIG.num_layers,
            CONFIG.n_clip,
            CONFIG.pred_step,
            CONFIG.finetune_dropout,
            700,
        )

    """  Load Pretrained Weights  """
    saver = ModelSaver(os.path.join(CONFIG.outpath, CONFIG.config_name))
    saver.load_ckpt(model.dpc, optimizer=None, scheduler=None, start_epoch=CONFIG.finetune_from)

    """  Model Components  """
    criterion = ClassificationLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG.finetune_lr, weight_decay=CONFIG.finetune_weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG.finetune_dampening_rate,
        patience=CONFIG.finetune_patience,
        verbose=True,
    )

    """  Load from Checkpoint  """
    saver = ModelSaver(os.path.join(CONFIG.outpath, f"{CONFIG.config_name}_finetune"))
    if opt.resume:
        saver.load_ckpt(model, optimizer, scheduler)
    else:
        saver.epoch = 1

    """  Devices  """
    gpu_ids = list(map(str, CONFIG.finetune_gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU numbers {}".format(CONFIG.finetune_gpu_ids))
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

    """  Dataset  """
    sp_t, tp_t = get_transforms("train", CONFIG)
    train_ds = Kinetics700(
        CONFIG.data_path,
        CONFIG.video_path,
        CONFIG.ann_path,
        clip_len=CONFIG.clip_len,
        n_clip=CONFIG.n_clip,
        downsample=CONFIG.downsample,
        spatial_transform=sp_t,
        temporal_transform=tp_t,
        mode="train",
    )
    sp_t, tp_t = get_transforms("val", CONFIG, finetune=True)
    val_ds = Kinetics700(
        CONFIG.data_path,
        CONFIG.video_path,
        CONFIG.ann_path,
        clip_len=CONFIG.clip_len,
        n_clip=CONFIG.n_clip,
        downsample=CONFIG.downsample,
        spatial_transform=sp_t,
        temporal_transform=tp_t,
        mode="val",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=CONFIG.finetune_batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CONFIG.finetune_batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn,
    )

    """  Training Loop  """
    for ep in range(saver.epoch, CONFIG.max_epoch + 1):
        print(f"global time {global_timer} | start training epoch {ep}")
        train_epoch(train_dl, model, optimizer, criterion, device, CONFIG, ep)
        print(f"global time {global_timer} | start validation epoch {ep}")
        val_acc = validate(val_dl, model, criterion, device, CONFIG, ep)
        if CONFIG.use_wandb:
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
        scheduler.step(val_acc)
        saver.save_ckpt_if_best(
            model, optimizer, scheduler, val_acc, delete_prev=CONFIG.only_best_checkpoint,
        )
        print(f"global time {global_timer} | val accuracy: {val_acc:.5f}% | " f"end epoch {ep}")
