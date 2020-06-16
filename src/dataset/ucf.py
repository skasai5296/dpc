import glob
import json
import os

import torch
from torch.utils.data import Dataset

from utils import (VideoLoaderHDF5, clipify, collate_fn, get_stats,
                   get_transforms)


class UCF101(Dataset):
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
        self.mode = mode
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
        failcnt = 0
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
                    failcnt += 1
                    continue
                subset = ann[id]["subset"]
                if mode == "train" and subset != "training":
                    continue
                elif mode == "val" and subset != "validation":
                    continue
                duration = ann[id]["annotations"]["segment"]
                if duration[1] < clip_len * n_clip * downsample:
                    # print("{} too short".format(id))
                    failcnt += 1
                    continue
                obj = {"class": classname, "path": p, "id": id, "duration": duration}
                self.data.append(obj)
        print("using {}/{} videos for {}".format(len(self), len(self) + failcnt, mode))

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
        frame_indices = range(start - 1, end - 1)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            assert len(frame_indices) == self.clip_len * self.n_clip
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = torch.stack([self.spatial_transform(img) for img in clip], 1)
        clip = clipify(clip, self.clip_len)
        label = self.classes.index(obj["class"])
        return {"clip": clip, "label": label}


if __name__ == "__main__":
    sp_t, tp_t = get_transforms("train", resize=112, clip_len=5, n_clip=8, downsample=2)
    ds = UCF101(
        "/groups1/gaa50131/datasets/ucf101",
        "hdf5",
        "anno/ucf101_01.json",
        spatial_transform=sp_t,
        temporal_transform=tp_t,
        clip_len=5,
        n_clip=8,
        downsample=2,
        mode="train",
    )
    for i in range(20):
        print(ds[i]["clip"].size())
        print(ds[i]["label"])
