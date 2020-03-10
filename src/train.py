import torch
from torch import nn
from torchvision import transforms

from dataset.kinetics import Kinetics700, get_stats
from model.model import DPC
from util.utils import sec2str, weight_init

model = DPC(512, 512, 3, 1)

mean, std = get_stats()
sp_t = transforms.Compose(
    [
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)
tp_t = None
ds = Kinetics700(
    "/home/seito/ssd2/kinetics/",
    "videos_700_hdf5",
    "kinetics-700-hdf5.json",
    spatial_transforms=sp_t,
    temporal_transforms=tp_t,
    mode="val",
)
device = torch.device("cuda")
model = model.to(device)
for i in range(100):
    feature = ds[i]["feature"].unsqueeze(0).to(device)
    print(feature.size())
    with torch.no_grad():
        out, _ = model(feature)
        print(out.size())
