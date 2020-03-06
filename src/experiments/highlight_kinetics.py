import sys, os, glob
import argparse
import time
import random
import io
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import faiss
import h5py
import cv2
import spacy

from dataset.kinetics import Kinetics700
from utils import Logger, sec2str
from model import MSE, CaptionEncoder
from vocab import Vocabulary

nlp = spacy.load('en_core_web_sm')

def search_video(dset, emb, loss):
    video = dset.embedded["video"]
    video_ids = dset.embedded["video_id"]

    nd, d = video.shape
    print("# videos: {}, dimension: {}".format(nd, d), flush=True)
    cpu_index = faiss.IndexFlatIP(d)
    cpu_index.add(video)
    D, I = cpu_index.search(emb, 5)
    nearest = [video_ids[idx] for idx in I[0]]
    print("top 5 nearest video ids are {}".format(nearest))
    print("top 5 cosine similarities are {}".format(D[0].tolist()))
    return nearest

# attention : (1, T', H', W'), torch.Tensor
def place_map(hdfpath, attention, savedir):
    print("creating heatmap images...", flush=True)
    transform = transforms.ToTensor()
    with h5py.File(hdfpath, 'r') as f:
        video_data = f['video']
        video = np.stack([np.asarray(Image.open(io.BytesIO(frame))) for frame in video_data])
        # (T, H, W, 3), numpy.ndarray
        T, H, W, _ = video.shape
        # (T, H, W), numpy.ndarray
        attention = F.interpolate(attention.unsqueeze(0), size=(T, H, W), mode='trilinear', align_corners=False).squeeze().detach().cpu().numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        attention = (attention * 255).clip(min=0, max=255)
    for i, (frame, att) in enumerate(zip(video, attention)):
        heatmap = cv2.applyColorMap(np.uint8(att), cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap, 0.6, frame, 0.4, 0)
        cv2.imwrite(os.path.join(savedir, "{:05d}.png".format(i)), fin)
    print("done! saved frames to {}".format(savedir), flush=True)
    return H, W

def frame2vid(beforedir, aftername, H, W):
    print("converting frames to video...", flush=True)
    H = H if H%2 == 0 else H+1
    W = W if W%2 == 0 else W+1
    cmd = ['ffmpeg', '-i', os.path.join(beforedir, '%05d.png'),
        '-r', '15', '-crf', '15', '-vf', "pad={}:{}".format(W, H), '-y',
        '-pix_fmt', 'yuv420p', os.path.join(beforedir, aftername)
    ]
    subprocess.call(cmd)
    for f in glob.glob(os.path.join(beforedir, "*.png")):
        os.remove(f)
    print("done! saved in {}".format(aftername), flush=True)

# for visualization
def get_map(ft_map, pos, temp):
    assert ft_map.size()[0:2] == pos.size()
    B, C, T, H, W = ft_map.size()
    # ft_map : (B, THW, C)
    ft_map = ft_map.view(B, C, -1).transpose(1, 2)
    # emb_exp : (B, C, 1)
    emb_exp = pos.unsqueeze(-1)
    # map : (B, THW)
    map = F.softmax((ft_map @ emb_exp).squeeze(-1) / temp, dim=1)
    return map.view(B, T, H, W)


def main():
    args = parse_args()

    transform = transforms.ToTensor()

    test_dset = Kinetics700(root_path=args.root_path, feature_dir=args.feature_dir, classname=args.classname)
    vocab_path = "captions_msrvtt.txt"
    noun_path = "nouns_msrvtt.txt"
    verb_path = "verbs_msrvtt.txt"

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(vocab_path)
    vocab.load_entities(noun_path, verb_path)

    ftenc = MSE(ft_size_slow=args.ft_size_slow, ft_size_fast=args.ft_size_fast, out_size=args.out_size)
    capenc = CaptionEncoder(vocab_size=len(vocab), emb_size=args.emb_size, out_size=args.out_size, rnn_type=args.rnn_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ftenc = ftenc.to(device)
    capenc = capenc.to(device)

    # load from model
    ckpt_dir = os.path.join("models", args.config_name)
    checkpoint = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
    print("loading model checkpoint from {} ...".format(checkpoint), flush=True)
    ckpt = torch.load(checkpoint, map_location=device)
    ftenc.load_state_dict(ckpt["encoder_state"])
    capenc.load_state_dict(ckpt["decoder_state"])

    ftenc = nn.DataParallel(ftenc)
    capenc = nn.DataParallel(capenc)

    save_path = os.path.join("out", args.config_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, 0o777)

    for i, obj in enumerate(test_dset):
        id = obj["id"]
        slowft = obj["slowft"]
        fastft = obj["fastft"]
        classname = obj["class"]
        (slowmap, fastmap), _ = ftenc(slowft.unsqueeze(0), fastft.unsqueeze(0))
        nmap = fastmap if not args.slow else slowmap

        for cap in args.caption:
            capid = vocab.return_idx([cap])
            with torch.no_grad():
                # emb : (1, d) tensor
                emb = capenc(capid, False).cpu().numpy()
            emb = torch.Tensor(emb).to(device)

            # nmap : (1, d, T, H, W) tensor
            rel_map = get_map(nmap, emb, args.temperature)

            video_path = os.path.join(args.root_path, args.data_dir, classname, "{}.hdf5".format(id))
            H, W = place_map(video_path, rel_map, save_path)
            frame2vid(save_path, "{}_{}.mp4".format(id, cap.replace(" ", "_").rstrip(".")), H, W)
        if i % 10 == 9:
            break

    print("done visualizing!!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--caption', type=str, nargs='+', required=True)
    parser.add_argument('--classname', type=str, default="pouring milk")
    parser.add_argument('--temperature', type=float, default=0.001)
    parser.add_argument('--slow', action="store_true")

    # configurations of dataset (paths)
    parser.add_argument('--root_path', type=str, default='/groups1/gaa50131/datasets/kinetics')
    parser.add_argument('--feature_dir', type=str, default='features')
    parser.add_argument('--data_dir', type=str, default='videos_700_hdf5')
    parser.add_argument('--config_name', type=str, default="default", help='Name of configurations, will denote where to save data')

    # configurations of models
    parser.add_argument('--rnn_type', type=str, default="GRU", help="architecture of rnn")
    parser.add_argument('--ft_size_slow', type=int, default=2048, help="precomputed feature sizes")
    parser.add_argument('--ft_size_fast', type=int, default=256, help="precomputed feature sizes")
    parser.add_argument('--emb_size', type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=1024, help="embedding size for output vectors for motion-semantic space")

    # training config
    parser.add_argument('--n_cpu', type=int, default=32, help="number of threads for dataloading")
    parser.add_argument('--max_len', type=int, default=30, help="max length of sentences")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")

    parser.add_argument('--batch_size', type=int, default=8, help="irrelevant")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
