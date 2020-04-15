import argparse
import os
from pprint import pprint

import torch
import wandb
import yaml
from addict import Dict
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.kinetics import Kinetics700, collate_fn, get_transforms
from model.criterion import DPCLoss
from model.model import DPC
from util import spatial_transforms, temporal_transforms
from util.utils import AverageMeter, ModelSaver, Timer


def train_epoch(loader, model, optimizer, criterion, device, CONFIG, epoch):
    train_timer = Timer()
    model.train()
    m = model.module if hasattr(model, "module") else model
    train_loss = AverageMeter("train XEloss")
    train_acc = AverageMeter("train Accuracy")
    for it, data in enumerate(loader):
        clip = data["clip"].to(device)

        optimizer.zero_grad()
        pred, gt = m(clip)
        loss, losses = criterion(pred, gt)

        train_loss.update(losses["XELoss"])
        train_acc.update(losses["Accuracy (%)"])
        loss.backward()
        if CONFIG.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=CONFIG.grad_clip)
        optimizer.step()

        if CONFIG.use_wandb:
            wandb.log({train_loss.name: train_loss.val, train_acc.name: train_acc.val})
        if it % 10 == 9:
            print(
                f"epoch {epoch:03d}/{CONFIG.max_epoch:03d} | train | "
                f"{train_timer} | iter {it+1:06d}/{len(loader):06d} | "
                f"{train_loss} | {train_acc}",
                flush=True,
            )
            train_loss.reset()
            train_acc.reset()


def validate(loader, model, criterion, device, CONFIG, epoch):
    val_timer = Timer()
    val_loss = AverageMeter("validation loss")
    val_acc = AverageMeter("validation accuracy")
    gl_val_loss = AverageMeter("epoch val XELoss")
    gl_val_acc = AverageMeter("epoch val Accuracy")
    model.eval()
    m = model.module if hasattr(model, "module") else model
    for it, data in enumerate(loader):
        clip = data["clip"].to(device)

        with torch.no_grad():
            pred, gt = m(clip)
            loss, losses = criterion(pred, gt)

        val_loss.update(losses["XELoss"])
        val_acc.update(losses["Accuracy (%)"])
        gl_val_loss.update(losses["XELoss"])
        gl_val_acc.update(losses["Accuracy (%)"])
        if it % 10 == 9:
            print(
                f"epoch {epoch:03d}/{CONFIG.max_epoch:03d} | valid | "
                f"{val_timer} | iter {it+1:06d}/{len(loader):06d} | "
                f"{val_loss} | {val_acc}",
                flush=True,
            )
            val_loss.reset()
            val_acc.reset()
        # validating for 100 steps is enough
        if it == 100:
            break
    if CONFIG.use_wandb:
        wandb.log({gl_val_loss.name: gl_val_loss.avg, gl_val_acc.name: gl_val_acc.avg})
    return gl_val_acc.avg


if __name__ == "__main__":
    global_timer = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="cfg/debug.yml",
        help="path to configuration yml file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="denotes if to continue training, will use config",
    )
    opt = parser.parse_args()
    print(f"loading configuration from {opt.config}")
    CONFIG = Dict(yaml.safe_load(open(opt.config)))
    print("CONFIGURATIONS:")
    pprint(CONFIG)

    if CONFIG.use_wandb:
        wandb.init(name=CONFIG.config_name, config=CONFIG, project=CONFIG.project_name)

    """  Model Components  """
    model = DPC(
        CONFIG.input_size,
        CONFIG.hidden_size,
        CONFIG.kernel_size,
        CONFIG.num_layers,
        CONFIG.n_clip,
        CONFIG.pred_step,
        CONFIG.dropout,
    )
    criterion = DPCLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=CONFIG.dampening_rate, patience=CONFIG.patience
    )

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
    sp_t, tp_t = get_transforms("val", CONFIG)
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
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn,
    )

    """  Training Loop  """
    for ep in range(saver.epoch, CONFIG.max_epoch):
        print(f"global time {global_timer} | start training epoch {ep}")
        train_epoch(train_dl, model, optimizer, criterion, device, CONFIG, ep)
        print(f"global time {global_timer} | start validation epoch {ep}")
        val_acc = validate(val_dl, model, criterion, device, CONFIG, ep)
        scheduler.step(val_acc)
        saver.save_ckpt_if_best(
            model,
            optimizer,
            scheduler,
            val_acc,
            delete_prev=CONFIG.only_best_checkpoint,
        )
        print(
            f"global time {global_timer} | val accuracy: {val_acc:.5f}% | "
            f"end epoch {ep}"
        )
