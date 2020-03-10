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
        spatial_transforms=None,
        temporal_transforms=None,
        mode="train",
    ):
        self.loader = VideoLoaderHDF5()
        self.spatial_transforms = spatial_transforms
        self.temporal_transforms = temporal_transforms
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
                    continue
                subset = ann[id]["subset"]
                if mode == "train" and subset != "training":
                    continue
                elif mode == "val" and subset != "validation":
                    continue
                duration = ann[id]["annotations"]["segment"]
                obj = {"class": classname, "path": p, "id": id, "duration": duration}
                self.data.append(obj)
        print("{} videos for {}".format(self.__len__(), mode))

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Get tensor of video.
        Returns:
            {
                feature:    torch.Tensor(C, T, H, W)
                class:      str,
                path:       str,
                id:         str,
            }
        """
        obj = self.data[index]
        path = obj["path"]
        start, end = obj["duration"]
        frame_indices = range(start, end + 1)
        if self.temporal_transforms is not None:
            frame_indices = self.temporal_transforms(frame_indices)
        ft = self.loader(path, frame_indices)
        ft = torch.stack([self.spatial_transforms(frame) for frame in ft], 1)
        ft = clipify(ft, 8)
        obj["feature"] = ft
        return obj


def clipify(tensor, clip_size):
    C, T, H, W = tensor.size()
    tensor = torch.split(tensor, clip_size, dim=1)
    if any([ten.size(1) != clip_size for ten in tensor]):
        tensor = tensor[:-1]
    return torch.stack(tensor)


def get_stats():
    """get mean and std of Kinetics"""
    maximum = 255.0
    mean = [110.63666788 / maximum, 103.16065604 / maximum, 96.29023126 / maximum]
    std = [38.7568578 / maximum, 37.88248729 / maximum, 40.02898126 / maximum]
    return mean, std


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
