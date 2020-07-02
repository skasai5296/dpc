import argparse
import hashlib
import os
import subprocess
from pprint import pprint

import torch
import yaml
from addict import Dict
from torch import nn

import wandb
from dataset.helper import get_dataloader
from model.helper import get_model_and_loss
from util.utils import AverageMeter, ModelSaver, Timer

def test(loader, model, criterion, device, CONFIG):
    test_timer = Timer()
    metrics = [AverageMeter("XELoss"), AverageMeter("Accuracy (%)")]
    global_metrics = [AverageMeter("XELoss"), AverageMeter("Accuracy (%)")]
    model.eval()
    for it, data in enumerate(loader):
        clip = data["clip"].to(device)
        label = data["label"].to(device)
        if it == 1 and torch.cuda.is_available():
            subprocess.run(["nvidia-smi"])

        with torch.no_grad():
            out = model(clip)
            loss, lossdict = criterion(out, label)

        for metric in metrics:
            metric.update(lossdict[metric.name])
        for metric in global_metrics:
            metric.update(lossdict[metric.name])
        if it % 10 == 9:
            metricstr = " | ".join([f"test {metric}" for metric in metrics])
            print(
                f"test | {test_timer} | iter {it+1:06d}/{len(loader):06d} | "
                f"{metricstr}",
                flush=True,
            )
            for metric in metrics:
                metric.reset()
    metric = global_metrics[-1]
    if CONFIG.use_wandb:
        wandb.log({f"test {metric.name}": metric.avg}, commit=False)
    return metric.avg


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

    CONFIG.new_config_name = f"ft_{CONFIG.dataset}_{CONFIG.config_name}"

    idhash = hashlib.sha256(CONFIG.new_config_name.encode()).hexdigest()
    if CONFIG.use_wandb:
        wandb.init(
            name=CONFIG.new_config_name,
            id=idhash,
            resume=idhash,
            config=CONFIG,
            project=CONFIG.project_name,
        )

    """  Model Components  """
    model, criterion = get_model_and_loss(CONFIG, finetune=True)

    """  Load from Checkpoint  """
    saver = ModelSaver(os.path.join(CONFIG.outpath, CONFIG.new_config_name))
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

    """  Dataset  """
    _, test_dl = get_dataloader(CONFIG)

    """  Training Loop  """
    print(f"global time {global_timer} | start test")
    test_acc = test(test_dl, model, criterion, device, CONFIG)
    print(f"global time {global_timer} | test accuracy: {test_acc:.5f}%")
