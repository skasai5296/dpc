import argparse
import os
from pprint import pprint

import torch
import wandb
import yaml
from addict import Dict
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.kinetics import Kinetics700, collate_fn, get_stats
from model.criterion import DPCLoss
from model.model import DPC
from util import spatial_transforms, temporal_transforms
from util.utils import ModelSaver, Timer


def train_epoch(loader, model, optimizer, criterion, device, CONFIG, epoch):
    train_timer = Timer()
    model.train()
    m = model.module if hasattr(model, "module") else model
    for it, data in enumerate(loader):
        clip = data["clip"].to(device)

        optimizer.zero_grad()
        pred, gt = model(clip)

        loss, losses = criterion(pred, gt)

        if CONFIG.use_wandb:
            wandb.log(losses)
        lossstr = " | ".join([f"{name}: {val:7f}" for name, val in losses.items()])

        loss.backward()
        if CONFIG.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=CONFIG.grad_clip)
        optimizer.step()

        if it % 10 == 9:
            print(f"train {train_timer} | iter {it+1} / {len(loader)} | {lossstr}")


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

    if CONFIG.basic.use_wandb:
        wandb.init(config=CONFIG, project=CONFIG.basic.project_name)

    model = DPC(
        CONFIG.input_size,
        CONFIG.hidden_size,
        CONFIG.kernel_size,
        CONFIG.num_layers,
        CONFIG.n_clip,
        CONFIG.pred_step,
        CONFIG.dropout,
    )
    saver = ModelSaver(CONFIG.outpath)
    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay
    )
    saver.load_ckpt(model, optimizer)

    gpu_ids = list(map(str, CONFIG.gpu_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("using GPU numbers {}".format(CONFIG.gpu_ids))
    else:
        device = torch.device("cpu")
        print("using CPU")

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    mean, std = get_stats()
    sp_t = spatial_transforms.Compose(
        [
            spatial_transforms.Resize(CONFIG.resize),
            spatial_transforms.RandomResizedCrop(size=(CONFIG.resize, CONFIG.resize)),
            spatial_transforms.RandomHorizontalFlip(),
            spatial_transforms.ToTensor(),
            spatial_transforms.Normalize(mean=mean, std=std),
        ]
    )
    tp_t = temporal_transforms.Compose(
        [
            temporal_transforms.TemporalSubsampling(CONFIG.downsample),
            temporal_transforms.TemporalRandomCrop(
                size=CONFIG.clip_len * CONFIG.n_clip
            ),
        ]
    )

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
    train_dl = DataLoader(
        train_ds,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn,
    )
    criterion = DPCLoss()
    for ep in range(saver.epoch, CONFIG.max_epoch):
        train_epoch(train_dl, model, optimizer, criterion, device, CONFIG, ep)
