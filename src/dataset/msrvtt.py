import sys, os
import io
import json

import yaml
from addict import Dict
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


class MSR_VTT_Resnet(Dataset):

    def __init__(self, CONFIG, mode="train"):
        ft_dir = os.path.join(CONFIG.dataset.root_path, CONFIG.dataset.feature_path)
        self.feature_size = CONFIG.dataset.feature_size_slow
        self.video_ids = set()
        split_ids = set()
        self.data = []
        if mode == "train":
            begin, end = 0, 6513
        elif mode == "val":
            begin, end = 6513, 7010
        elif mode == "test":
            begin, end = 7010, 10000
        with open(os.path.join(CONFIG.dataset.root_path, "videodatainfo_2017.json"), "r") as f:
            ann = json.load(f)
            for video in ann["videos"]:
                # filter by mode
                if begin <= video["id"] < end:
                    split_ids.add(video["video_id"])
            for sentence in ann["sentences"]:
                video_id = sentence["video_id"]
                if not video_id in split_ids:
                    continue
                if not video_id in self.video_ids:
                    self.video_ids.add(video_id)
                    obj = {"video_id": int(video_id[5:]), "feature_path": os.path.join(ft_dir, video_id + ".pth"), "caption": [sentence["caption"]], "cap_id": [sentence["sen_id"]]}
                    self.data.append(obj)
                else:
                    self.data[self.__len__()-1]["caption"].append(sentence["caption"])
                    self.data[self.__len__()-1]["cap_id"].append(sentence["sen_id"])

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        id = data["video_id"]
        pth = data["feature_path"]
        caption = data["caption"]
        cap_id = data["cap_id"]
        ft = torch.load(pth)
        # interpolate features to given size
        try:
            assert list(ft.size()[1:]) == self.feature_size
        except AssertionError:
            ft = F.interpolate(ft.unsqueeze(0), size=self.feature_size, mode='trilinear', align_corners=False).squeeze(0)

        return {"feature": ft, "id": id, "caption": caption, "cap_id": cap_id, "number": len(caption)}

class MSR_VTT_SlowFast(Dataset):

    def __init__(self, CONFIG, mode="train"):
        ft_dir = os.path.join(CONFIG.dataset.root_path, CONFIG.dataset.feature_path)
        slow_dir = os.path.join(ft_dir, "sem_int")
        fast_dir = os.path.join(ft_dir, "mot_int")
        self.feature_size_slow = CONFIG.dataset.feature_size_slow
        self.feature_size_fast = CONFIG.dataset.feature_size_fast
        self.video_ids = []
        split_ids = set()
        self.data = []
        if mode == "train":
            begin, end = 0, 6513
        elif mode == "val":
            begin, end = 6513, 7010
        elif mode == "test":
            begin, end = 7010, 10000
        with open(os.path.join(CONFIG.dataset.root_path, CONFIG.dataset.ann_path), "r") as f:
            ann = json.load(f)
            for video in ann["videos"]:
                # filter by mode
                if begin <= video["id"] < end:
                    split_ids.add(int(video["id"]))
            for sentence in ann["sentences"]:
                video_id = int(sentence["video_id"][5:])
                if not video_id in split_ids:
                    continue
                if not video_id in self.video_ids:
                    self.video_ids.append(video_id)
                    obj = {
                        "video_id": video_id,
                        "feature_path_slow": os.path.join(slow_dir, "video" + str(video_id) + ".pth"),
                        "feature_path_fast": os.path.join(fast_dir, "video" + str(video_id) + ".pth"),
                        "caption": [sentence["caption"]],
                        "cap_id": [sentence["sen_id"]]
                    }
                    self.data.append(obj)
                else:
                    self.data[self.__len__()-1]["caption"].append(sentence["caption"])
                    self.data[self.__len__()-1]["cap_id"].append(sentence["sen_id"])

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        id = data["video_id"]
        slowpth = data["feature_path_slow"]
        fastpth = data["feature_path_fast"]
        caption = data["caption"]
        cap_id = data["cap_id"]
        slowft = torch.load(slowpth)
        fastft = torch.load(fastpth)
        # interpolate features to given size
        try:
            assert list(slowft.size()[1:]) == self.feature_size_slow
            assert list(fastft.size()[1:]) == self.feature_size_fast
        except AssertionError:
            slowft = F.interpolate(slowft.unsqueeze(0), size=self.feature_size_slow, mode='trilinear', align_corners=False).squeeze(0)
            fastft = F.interpolate(fastft.unsqueeze(0), size=self.feature_size_fast, mode='trilinear', align_corners=False).squeeze(0)

        return {"feature_slow": slowft, "feature_fast": fastft, "id": id, "caption": caption, "cap_id": cap_id, "number": len(caption)}

    def get_features_from_id(self, id):
        n = self.video_ids.index(id)
        data = self.data[n]
        slowpth = data["feature_path_slow"]
        fastpth = data["feature_path_fast"]
        slowft = torch.load(slowpth)
        fastft = torch.load(fastpth)
        # interpolate features to given size
        try:
            assert list(slowft.size()[1:]) == self.feature_size_slow
            assert list(fastft.size()[1:]) == self.feature_size_fast
        except AssertionError:
            slowft = F.interpolate(slowft.unsqueeze(0), size=self.feature_size_slow, mode='trilinear', align_corners=False).squeeze(0)
            fastft = F.interpolate(slowft.unsqueeze(0), size=self.feature_size_fast, mode='trilinear', align_corners=False).squeeze(0)

        return slowft, fastft

class EmbedDataset(Dataset):
    """Dataset to create when evaluating model"""

    def __init__(self, loader, feature_model, caption_model, vocab, args, max_instances=1000):
        """
        Args:
            loader: DataLoader for validation images and captions
            model: trained model to evaluate
        """
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.embedded = {"video": [], "video_id": [], "caption": [], "cap_id": [], "raw_caption": []}
        # number of captions per video
        cap_per_vid = 20 if args.dataset == "msrvtt" else 5
        iters = int(max_instances / loader.batch_size + 1)
        self.v2c = {}
        self.c2v = {}
        for it, data in enumerate(loader):
            ft = data["feature"]
            caption = [c for cap in data["caption"] for c in cap[:cap_per_vid]]
            self.embedded["raw_caption"].extend(caption)
            caption = vocab.return_idx(caption)
            ft = ft.to(device)
            caption = caption.to(device)
            cap_id = [c for cap in data["cap_id"] for c in cap[:cap_per_vid]]
            with torch.no_grad():
                rel_map, emb_vid = feature_model(ft)
                emb_cap = caption_model(caption)
            self.embedded["video"].append(emb_vid.cpu().numpy())
            self.embedded["caption"].append(emb_cap.cpu().numpy())
            self.embedded["video_id"].extend(data["id"])
            self.embedded["cap_id"].extend(cap_id)
            if it == iters:
                break
        self.embedded["video"] = np.concatenate(self.embedded["video"], axis=0)
        self.embedded["caption"] = np.concatenate(self.embedded["caption"], axis=0)
        self.embedded["video_id"] = self.embedded["video_id"]
        self.embedded["cap_id"] = self.embedded["cap_id"]
        if len(self.embedded["video"]) > max_instances:
            self.embedded["video"] = self.embedded["video"][:max_instances]
            self.embedded["caption"] = self.embedded["caption"][:max_instances * cap_per_vid]
            self.embedded["video_id"] = self.embedded["video_id"][:max_instances]
            self.embedded["cap_id"] = self.embedded["cap_id"][:max_instances * cap_per_vid]

    def __len__(self):
        return len(self.embedded["img_id"])

class EmbedDataset_MSE(Dataset):
    """Dataset to create when evaluating model"""

    def __init__(self, loader, feature_model, caption_model, vocab, args, max_instances=1000):
        """
        Args:
            loader: DataLoader for validation images and captions
            model: trained model to evaluate
        """
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.embedded = {"video": [], "semantic": [], "motion": [], "video_id": [], "caption": [], "cap_id": [], "raw_caption": []}
        # number of captions per video
        cap_per_vid = 20 if args.dataset == "msrvtt" else 5
        iters = int(max_instances / loader.batch_size + 1)
        for it, data in enumerate(loader):
            slowft = data["feature_slow"]
            fastft = data["feature_fast"]
            caption = [c for cap in data["caption"] for c in cap]
            self.embedded["raw_caption"].extend(caption)
            caption = vocab.return_idx(caption)
            slowft = slowft.to(device)
            fastft = fastft.to(device)
            caption = caption.to(device)
            cap_id = [c for cap in data["cap_id"] for c in cap]
            with torch.no_grad():
                (sem_map, mot_map), emb_vid = feature_model(slowft, fastft)
                emb_vid = emb_vid.cpu().numpy()
                emb_cap = caption_model(caption).cpu().numpy()
                sem_map = F.adaptive_avg_pool3d(sem_map, 1).squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy()
                mot_map = F.adaptive_avg_pool3d(mot_map, 1).squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy()
            self.embedded["video"].append(emb_vid)
            self.embedded["semantic"].append(sem_map)
            self.embedded["motion"].append(mot_map)
            self.embedded["caption"].append(emb_cap)
            self.embedded["video_id"].extend(data["id"])
            self.embedded["cap_id"].extend(cap_id)
            if it == iters:
                break
        self.embedded["video"] = np.concatenate(self.embedded["video"], axis=0)
        self.embedded["semantic"] = np.concatenate(self.embedded["semantic"], axis=0)
        self.embedded["motion"] = np.concatenate(self.embedded["motion"], axis=0)
        self.embedded["caption"] = np.concatenate(self.embedded["caption"], axis=0)
        self.embedded["video_id"] = self.embedded["video_id"]
        self.embedded["cap_id"] = self.embedded["cap_id"]
        # fix size if over limit
        self.embedded["video"] = self.embedded["video"][:max_instances]
        self.embedded["semantic"] = self.embedded["semantic"][:max_instances]
        self.embedded["motion"] = self.embedded["motion"][:max_instances]
        self.embedded["caption"] = self.embedded["caption"][:max_instances * cap_per_vid]
        self.embedded["raw_caption"] = self.embedded["raw_caption"][:max_instances * cap_per_vid]
        self.embedded["video_id"] = self.embedded["video_id"][:max_instances]
        self.embedded["cap_id"] = self.embedded["cap_id"][:max_instances]

    def __len__(self):
        return len(self.embedded["img_id"])


if __name__ == "__main__":
    CONFIG = Dict(yaml.safe_load(open("../../cfg/sample.yml")))
    if CONFIG.dataset.feature == "resnet":
        ds = MSR_VTT_Resnet(CONFIG, mode="train")
    elif CONFIG.dataset.feature == "slowfast":
        ds = MSR_VTT_SlowFast(CONFIG, mode="train")
    for i in range(1000):
        print(ds[i]["feature_slow"].size())
