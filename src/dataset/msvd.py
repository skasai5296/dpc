import sys, os
import io
import json

import h5py
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import spacy

# load spacy
nlp = spacy.load('en')

class VideoLoaderHDF5(object):

    def __call__(self, video_path, frame_indices=None, interval=2):
        with h5py.File(video_path, 'r') as f:
            video_data = f['video']

            video = []
            if frame_indices == None:
                video = [Image.open(io.BytesIO(frame)) for frame in video_data[::interval]]
                return video

            for i in frame_indices[::interval]:
                if i < len(video_data):
                    video.append(Image.open(io.BytesIO(video_data[i])))
                else:
                    return video

        return video


class MSVD_Resnet(Dataset):

    def __init__(self, root_path, feature_dir, mode="train", align_size=(10, 8, 10)):
        map_path = os.path.join(root_path, "youtube_video_to_id_mapping.txt")
        ann_path = os.path.join(root_path, "sents_{}_lc_nopunc.txt".format(mode))
        ft_dir = os.path.join(root_path, feature_dir)
        self.align_size= align_size
        split_ids = []
        self.data = []
        self.lid2num = {}
        self.num2lid = {}
        if mode == "train":
            self.offset = 0
        elif mode == "val":
            self.offset = 1200
        else:
            self.offset = 1300
        with open(map_path, "r") as f:
            for line in f:
                lid, id = line.rstrip().split()
                num = int(id[3:])
                self.lid2num[lid] = num
                self.num2lid[num] = lid
        with open(ann_path, "r") as f:
            for i, line in enumerate(f):
                id, cap = line.rstrip().split("\t", 1)
                num = int(id[3:])
                assert num in self.num2lid.keys()
                if id not in split_ids:
                    obj = {
                        "video_id": num,
                        "video_path": os.path.join(ft_dir, self.num2lid[num] + ".pth"),
                        "caption": [cap],
                        "cap_id": [i],
                    }
                    self.data.append(obj)
                    split_ids.append(id)
                else:
                    self.data[self.__len__()-1]["caption"].append(cap)
                    self.data[self.__len__()-1]["cap_id"].append(i)

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        id = data["video_id"]
        pth = data["video_path"]
        caption = data["caption"][:5]
        assert len(caption) == 5
        cap_id = data["cap_id"]
        ft = torch.load(pth)
        if self.align_size:
            # interpolate features to given size
            if (ft.size(1), ft.size(2), ft.size(3)) != self.align_size:
                ft = F.interpolate(ft.unsqueeze(0), size=self.align_size, mode='trilinear', align_corners=False).squeeze(0)

        return {"feature": ft, "id": id, "caption": caption, "cap_id": cap_id, "number": len(caption)}

class MSVD_SlowFast(Dataset):

    def __init__(self, root_path, feature_dir, mode="train", align_size_slow=(10, 8, 10), align_size_fast=(100, 8, 10)):
        map_path = os.path.join(root_path, "youtube_video_to_id_mapping.txt")
        ann_path = os.path.join(root_path, "sents_{}_lc_nopunc.txt".format(mode))
        ft_dir = os.path.join(root_path, feature_dir)
        slow_dir = os.path.join(ft_dir, "semantic")
        fast_dir = os.path.join(ft_dir, "motion")
        self.align_size_slow = align_size_slow
        self.align_size_fast = align_size_fast
        split_ids = []
        self.data = []
        self.lid2num = {}
        self.num2lid = {}
        if mode == "train":
            self.offset = 0
        elif mode == "val":
            self.offset = 1200
        else:
            self.offset = 1300
        with open(map_path, "r") as f:
            for line in f:
                lid, id = line.rstrip().split()
                num = int(id[3:])
                self.lid2num[lid] = num
                self.num2lid[num] = lid
        with open(ann_path, "r") as f:
            for i, line in enumerate(f):
                id, cap = line.rstrip().split("\t", 1)
                num = int(id[3:])
                # remove ids with too long videos (feature extraction problems)
                if self.num2lid[num] in ['D2FbgK_kkE8_121_151', 'wON-YuA1GjA_3_63', '5dv8MSWlxoQ_10_40']:
                    continue
                assert num in self.num2lid.keys()
                if id not in split_ids:
                    obj = {
                        "video_id": num,
                        "video_path_slow": os.path.join(slow_dir, self.num2lid[num] + ".pth"),
                        "video_path_fast": os.path.join(fast_dir, self.num2lid[num] + ".pth"),
                        "caption": [cap],
                        "cap_id": [i],
                    }
                    self.data.append(obj)
                    split_ids.append(id)
                else:
                    self.data[self.__len__()-1]["caption"].append(cap)
                    self.data[self.__len__()-1]["cap_id"].append(i)

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        id = data["video_id"]
        slowpth = data["video_path_slow"]
        fastpth = data["video_path_fast"]
        caption = data["caption"][:5]
        assert len(caption) == 5
        cap_id = data["cap_id"]
        slowft = torch.load(slowpth)
        fastft = torch.load(fastpth)
        if self.align_size_slow and self.align_size_fast:
            # interpolate features to given size
            if (slowft.size(1), slowft.size(2), slowft.size(3)) != self.align_size_slow:
                slowft = F.interpolate(slowft.unsqueeze(0), size=self.align_size_slow, mode='trilinear', align_corners=False).squeeze(0)
            if (fastft.size(1), fastft.size(2), fastft.size(3)) != self.align_size_fast:
                fastft = F.interpolate(slowft.unsqueeze(0), size=self.align_size_fast, mode='trilinear', align_corners=False).squeeze(0)

        return {"feature_slow": slowft, "feature_fast": fastft, "id": id, "caption": caption, "cap_id": cap_id, "number": len(caption)}

    def get_features_from_id(self, num):
        assert self.offset < num < 1971
        num -= self.offset
        data = self.data[num]
        slowpth = data["video_path_slow"]
        fastpth = data["video_path_fast"]
        slowft = torch.load(slowpth)
        fastft = torch.load(fastpth)
        if self.align_size_slow and self.align_size_fast:
            # interpolate features to given size
            if (slowft.size(1), slowft.size(2), slowft.size(3)) != self.align_size_slow:
                slowft = F.interpolate(slowft.unsqueeze(0), size=self.align_size_slow, mode='trilinear', align_corners=False).squeeze(0)
            if (fastft.size(1), fastft.size(2), fastft.size(3)) != self.align_size_fast:
                fastft = F.interpolate(slowft.unsqueeze(0), size=self.align_size_fast, mode='trilinear', align_corners=False).squeeze(0)

        return slowft, fastft


if __name__ == "__main__":
    ds = MSVD_SlowFast("/groups1/gaa50131/datasets/MSVD/", "features/sfnl152_k700_16f", mode="train")
    #ds = MSVD_Resnet("/groups1/gaa50131/datasets/MSVD/", "features/r50_k700_16f", mode="train")
    for i in range(100):
        print(ds[i]["feature_slow"].size())
        print(ds[i]["feature_fast"].size())
        #print(ds[i]["feature"].size())



