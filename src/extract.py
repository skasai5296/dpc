import argparse
import hashlib
import os
import subprocess
from pprint import pprint

import torch
import yaml
from addict import Dict
from torch import nn
from torchvision.io import read_video
from torch.utils.data import DataLoader

from dataset.gtea import GTEA
from dataset.utils import (clipify, collate_fn, get_transforms,
                           get_transforms_finetune)
from model.helper import get_model_and_loss
from util.utils import AverageMeter, ModelSaver, Timer


def extract_features(dataset, model, device, CONFIG):
    test_timer = Timer()
    model.eval()
    for it, data in enumerate(dataset):
        clip = data["clip"].to(device)
        id = data["id"]
        if it == 1 and torch.cuda.is_available():
            subprocess.run(["nvidia-smi"])

        # (T/n_clip, n_clip, C, clip_len, H, W)
        duration = data["duration"]
        with torch.no_grad():
            out = model(clip, flag="extract")
        # (T, 7 * 7, D)
        out = out.reshape(-1, 7 * 7, CONFIG.hidden_size)
        out = out.mean(1)[:duration]
        print(out.size())
        torch.save(out, os.path.join(dataset.root_path, f"feature/{id}.pth"))

        if it % 10 == 9:
            print(
                f"extracting features | {test_timer} | iter {it+1:06d}/{len(dataset):06d} | ",
                flush=True,
            )


if __name__ == "__main__":

    global_timer = Timer()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="cfg/default.yml", help="path to configuration yml file",
    )
    opt = parser.parse_args()
    print(f"loading configuration from {opt.config}")
    CONFIG = Dict(yaml.safe_load(open(opt.config)))
    print("CONFIGURATIONS:")
    pprint(CONFIG)

    root = "/groups1/gaa50131/datasets/GTEA"
    if not os.path.exists(os.path.join(root, "feature")):
        os.mkdir(os.path.join(root, "feature"))
    assert CONFIG.model == "FGCPC"

    """  Model Components  """
    model, _ = get_model_and_loss(CONFIG, finetune=False)

    """  Load from Checkpoint  """
    saver = ModelSaver(os.path.join(CONFIG.outpath, CONFIG.config_name))
    saver.load_ckpt(model, None, None)

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
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        subprocess.run(["nvidia-smi"])

    sp_t, tp_t = get_transforms("extract", resize=112, clip_len=1, n_clip=8, downsample=1)
    """  Dataset  """
    ds = GTEA(
        root,
        clip_len=1,
        n_clip=8,
        downsample=1,
        spatial_transform=sp_t,
        temporal_transform=tp_t,
    )

    """  Training Loop  """
    print(f"global time {global_timer} | start extracting")
    extract_features(ds, model, device, CONFIG)
    print(f"global time {global_timer} | done extracting")
