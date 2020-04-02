import glob
import io
import json
import os
import random
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from addict import Dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from util import spatial_transforms, temporal_transforms


class VideoLoaderHDF5(object):
    def __call__(self, video_path, frame_indices):
        with h5py.File(video_path, "r") as f:
            video_data = f["video"]

            video = []
            for i in frame_indices:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


class Kinetics700(Dataset):
    def __init__(
        self,
        root_path,
        hdf_path,
        ann_path,
        clip_len,
        n_clip,
        downsample,
        spatial_transform,
        temporal_transform,
        mode="train",
    ):
        self.loader = VideoLoaderHDF5()
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.clip_len = clip_len
        self.n_clip = n_clip
        self.downsample = downsample
        assert mode in ("train", "val")

        ft_dir = os.path.join(root_path, hdf_path)
        classes = [os.path.basename(i) for i in glob.glob(os.path.join(ft_dir, "*"))]
        ann_path = os.path.join(root_path, ann_path)
        with open(ann_path, "r") as f:
            obj = json.load(f)
        labels = set(obj["labels"])
        ann = obj["database"]
        self.data = []
        self.classes = []
        for classname in classes:
            if classname == "test":
                continue
            if classname not in self.classes:
                self.classes.append(classname)
            assert classname in labels
            filepath = sorted(glob.glob(os.path.join(ft_dir, classname, "*")))
            for p in filepath:
                id = os.path.basename(p)[:-5]
                if id not in ann.keys():
                    # print("{} not in annotation".format(id))
                    continue
                subset = ann[id]["subset"]
                if mode == "train" and subset != "training":
                    continue
                elif mode == "val" and subset != "validation":
                    continue
                duration = ann[id]["annotations"]["segment"]
                if duration[1] < clip_len * n_clip * downsample:
                    # print("{} too short".format(id))
                    continue
                obj = {"class": classname, "path": p, "id": id, "duration": duration}
                self.data.append(obj)
        print("{} videos for {}".format(len(self), mode))

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Get tensor of video.
        Returns:
            {
                clip:       torch.Tensor(n_clip, C, clip_len, H, W)
                class:      str,
                id:         str,
            }
        """
        obj = self.data[index]
        path = obj["path"]
        start, end = obj["duration"]
        frame_indices = range(start, end + 1)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            # assert len(frame_indices) == self.clip_len * self.n_clip
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = torch.stack([self.spatial_transform(img) for img in clip], 1)
        clip = clipify(clip, self.clip_len)
        label = self.classes.index(obj["class"])
        return {"clip": clip, "label": label}


def clipify(tensor, clip_len):
    """
    Divide tensor of video frames into clips
    Args:
        tensor: torch.Tensor(C, n_clip*clip_len, H, W)
        clip_len: int, number of frames for a single clip
        n_clip: int, number of clips to form
    Returns:
        torch.Tensor(n_clip, C, clip_len, H, W), sampled clips
    """
    tensor = torch.stack(torch.split(tensor, clip_len, dim=1))
    return tensor


def get_stats():
    """get mean and std of Kinetics"""
    maximum = 255.0
    mean = [110.63666788 / maximum, 103.16065604 / maximum, 96.29023126 / maximum]
    std = [38.7568578 / maximum, 37.88248729 / maximum, 40.02898126 / maximum]
    return mean, std


def get_transforms(mode, CONFIG, finetune=False):
    assert mode in ("train", "val", "finetune")
    mean, std = get_stats()
    if mode == "train":
        sp_t = spatial_transforms.Compose(
            [
                spatial_transforms.Resize(int(CONFIG.resize * 8 / 7)),
                spatial_transforms.RandomResizedCrop(
                    size=(CONFIG.resize, CONFIG.resize), scale=(0.5, 1.0)
                ),
                spatial_transforms.RandomHorizontalFlip(),
                spatial_transforms.RandomGrayscale(p=0.5),
                spatial_transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
                ),
                spatial_transforms.ToTensor(),
                spatial_transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif mode == "val":
        sp_t = spatial_transforms.Compose(
            [
                spatial_transforms.Resize(CONFIG.resize),
                spatial_transforms.CenterCrop(size=(CONFIG.resize, CONFIG.resize)),
                spatial_transforms.ToTensor(),
                spatial_transforms.Normalize(mean=mean, std=std),
            ]
        )
    if finetune:
        tp_t = temporal_transforms.Compose(
            [
                temporal_transforms.TemporalSubsampling(CONFIG.downsample),
                temporal_transforms.SlidingWindow(size=CONFIG.clip_len, stride=4),
            ]
        )
    else:
        tp_t = temporal_transforms.Compose(
            [
                temporal_transforms.TemporalSubsampling(CONFIG.downsample),
                temporal_transforms.TemporalRandomCrop(
                    size=CONFIG.clip_len * CONFIG.n_clip
                ),
            ]
        )
    return sp_t, tp_t


def collate_fn(datalist):
    clips = []
    labels = []
    for data in datalist:
        clip = data["clip"]
        label = data["label"]
        clips.append(clip)
        labels.append(label)
    clips = torch.stack(clips)
    labels = torch.tensor(labels, dtype=torch.long)
    return {"clip": clips, "label": labels}


if __name__ == "__main__":
    cfg_path = "../cfg/debug.yml"
    CONFIG = Dict(yaml.safe_load(open(opt.config)))
    mean, std = get_stats()
    sp_t, tp_t = get_transforms("train", CONFIG)
    ds = Kinetics700(
        "/home/seito/ssd2/kinetics/",
        "videos_700_hdf5",
        "kinetics-700-hdf5.json",
        spatial_transform=sp_t,
        temporal_transform=tp_t,
        mode="val",
    )
    for i in range(10):
        print(ds[i]["feature"].size())
        print(ds[i]["class"])
        print(ds[i]["id"])
        print(ds[i]["duration"])
