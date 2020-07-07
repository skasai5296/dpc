import glob
import json
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

from dataset.utils import clipify, get_transforms


class GTEA(Dataset):
    def __init__(
        self, root_path, clip_len, n_clip, downsample, spatial_transform, temporal_transform,
    ):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.clip_len = clip_len
        self.n_clip = n_clip
        self.downsample = downsample
        self.data = []
        self.root_path = root_path
        for p in glob.glob(os.path.join(root_path, "*.avi")):
            id = os.path.basename(p)[:-4]
            self.data.append({"path": p, "id": id})

        print("using {} videos".format(len(self)))

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Get tensor of video.
        Returns:
            {
                clip:       torch.Tensor(B, N, C, T, H, W)
                class:      str,
                id:         str,
            }
        """
        obj = self.data[index]
        path = obj["path"]
        id = obj["id"]
        # vframes: (T, H, W, C)
        vframes, aframes, info = read_video(path, pts_unit="sec")
        # vframes: (T, C, H, W)
        vframes = vframes.permute(0, 3, 1, 2)
        start, end = 0, vframes.size(0)
        frame_indices = range(start, end)
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            vframes = torch.stack([self.spatial_transform(img) for img in vframes], 0)
        # vframes: (T, C, H, W)
        clips = []
        for f in frame_indices:
            clips.append(torch.stack([vframes[t] for t in f], 0))
        # vframes: (B, T, C, H, W)
        clip = torch.stack(clips, 0)
        # vframes: (B/N, N, C, T, H, W)
        clip = torch.stack([clipify(c.permute(1, 0, 2, 3), self.clip_len) for c in clip], 0)
        print(id, info, clip.size())
        return {"clip": clip, "id": id, "duration": end}


if __name__ == "__main__":
    sp_t, tp_t = get_transforms("extract", resize=112, clip_len=5, n_clip=8, downsample=3)
    ds = GTEA(
        "/groups1/gaa50131/datasets/GTEA",
        clip_len=5,
        n_clip=8,
        downsample=3,
        spatial_transform=sp_t,
        temporal_transform=tp_t,
    )
    for i in range(20):
        print(ds[i]["clip"].size())
