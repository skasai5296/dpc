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
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        clip_len=8,
        n_clip=5,
        downsample=1,
        spatial_transforms=None,
        temporal_transforms=None,
        mode="train",
    ):
        self.loader = VideoLoaderHDF5()
        self.spatial_transforms = spatial_transforms
        self.temporal_transforms = temporal_transforms
        self.clip_len = clip_len
        self.n_clip = n_clip
        self.downsample = downsample

        ft_dir = os.path.join(root_path, hdf_path)
        classes = [os.path.basename(i) for i in glob.glob(os.path.join(ft_dir, "*"))]
        ann_path = os.path.join(root_path, ann_path)
        with open(ann_path, "r") as f:
            obj = json.load(f)
        labels = set(obj["labels"])
        ann = obj["database"]
        self.data = []
        for classname in classes:
            if classname == "test":
                continue
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
        if self.temporal_transforms is not None:
            frame_indices = self.temporal_transforms(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transforms is not None:
            self.spatial_transforms.randomize_parameters()
            clip = torch.stack([self.spatial_transforms(img) for img in clip], 1)
        clip = clipify(clip, self.clip_len, self.n_clip)
        return {"clip": clip}


def clipify(tensor, clip_len, n_clip):
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


def collate_fn(datalist):
    clips = []
    for data in datalist:
        clip = data["clip"]
        clips.append(clip)
    clips = torch.stack(clips)
    return {"clip": clips}


if __name__ == "__main__":
    mean, std = get_stats()
    sp_t = transforms.Compose(
        [
            transforms.RandomCrop(224),
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
    for i in range(10):
        print(ds[i]["feature"].size())
        print(ds[i]["class"])
        print(ds[i]["id"])
        print(ds[i]["duration"])
