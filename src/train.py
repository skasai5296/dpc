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
from model.model import DPC
from util import spatial_transforms, temporal_transforms
from util.utils import ModelSaver, Timer


def train_epoch(loader, model, optimizer, criterion, device, CONFIG, epoch):
    train_timer = Timer()
    model.train()
    m = model.module if hasattr(model, "module") else model
    for it, data in enumerate(loader):
        clip = data["clip"].to(device)
        print(clip.size())

        optimizer.zero_grad()
        out, _ = model(clip)
        print(out.size())

        loss, losses = criterion(out)

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
        default="../cfg/debug.yml",
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
    print("\n\n")

    if CONFIG.basic.use_wandb:
        wandb.init(config=CONFIG, project=CONFIG.basic.project_name)

    clip_len = 5
    n_clip = 8
    downsample = 1
    input_size = 512
    hidden_size = 512
    kernel_size = 3
    num_layers = 1
    resize = 112
    batch_size = 4
    num_workers = 4
    outpath = "out"
    gpu_ids = [0]

    model = DPC(
        CONFIG.input_size, CONFIG.hidden_size, CONFIG.kernel_size, CONFIG.num_layers
    )
    saver = ModelSaver(CONFIG.outpath)
    optimizer = optim.Adam(model.parameters())
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
            temporal_transforms.TemporalRandomCrop(
                size=CONFIG.clip_len * CONFIG.n_clip
            ),
            temporal_transforms.TemporalSubsampling(CONFIG.downsample),
        ]
    )

    ds = Kinetics700(
        "/home/seito/ssd2/kinetics/",
        "videos_700_hdf5",
        "kinetics-700-hdf5.json",
        clip_len=CONFIG.clip_len,
        n_clip=CONFIG.n_clip,
        downsample=CONFIG.downsample,
        spatial_transforms=sp_t,
        temporal_transforms=tp_t,
        mode="train",
    )
    dl = DataLoader(
        ds,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        collate_fn=collate_fn,
    )
