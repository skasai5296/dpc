import sys, os, glob
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



class Kinetics700(Dataset):

    def __init__(self, root_path, feature_dir, classname):
        ft_dir = os.path.join(root_path, feature_dir)
        classes = [os.path.basename(i) for i in glob.glob(os.path.join(ft_dir, "*"))]
        try:
            assert classname in classes
        except AssertionError:
            print("class {} is not in features directory {}".format(classname, ft_dir))
            print("available classes: {}".format(classes))
        self.data = []
        path = os.path.join(ft_dir, classname)
        mot_path = sorted(glob.glob(os.path.join(path, "motion/*")))
        sem_path = sorted(glob.glob(os.path.join(path, "semantic/*")))
        for sempath, motpath in zip(sem_path, mot_path):
            obj = {"class": classname, "slow_path": sempath, "fast_path": motpath, "id": os.path.basename(motpath)[:-4]}
            self.data.append(obj)

    # return number of features
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        obj = self.data[index]
        assert os.path.basename(obj["slow_path"]) == os.path.basename(obj["fast_path"])
        slowft = torch.load(obj["slow_path"])
        fastft = torch.load(obj["fast_path"])
        obj["slowft"] = slowft
        obj["fastft"] = fastft
        return obj


if __name__ == "__main__":
    ds = Kinetics700("/groups1/gaa50131/datasets/kinetics/", "features")
    for i in range(10):
        print(ds[i]["tensor"].size())
        print(ds[i]["class"])
