import sys, os, glob
import shutil
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
import faiss
import h5py
import cv2
import spacy

from dataset.msrvtt import MSR_VTT_SlowFast, EmbedDataset_MSE
from dataset.msvd import MSVD_SlowFast
from utils import Logger, AlignmentLoss, NPairLoss, weight_init, collater, collater_mse, sec2str
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

def frame2vid(beforedir, aftername):
    print("converting frames to video...", flush=True)
    cmd = ['ffmpeg', '-i', os.path.join(beforedir, '%05d.png'),
        '-r', '15', '-crf', '15', '-s', "320x240", '-y',
        '-pix_fmt', 'yuv420p', os.path.join(beforedir, aftername)
    ]
    subprocess.call(cmd)
    for f in glob.glob(os.path.join(beforedir, "*.png")):
        os.remove(f)
    print("done! saved in {}".format(aftername), flush=True)

# for visualization
def get_map(ft_map, pos, temp=0.0001):
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

    assert args.dataset in ["msrvtt", "msvd"]
    if args.dataset == "msrvtt":
        test_dset = MSR_VTT_SlowFast(root_path=args.root_path, feature_dir=args.feature_dir, mode='test')
    elif args.dataset == "msvd":
        test_dset = MSVD_SlowFast(root_path=args.root_path, feature_dir=args.feature_dir, mode='test')
    vocab_path = "captions_{}.txt".format(args.dataset)
    noun_path = "nouns_{}.txt".format(args.dataset)
    verb_path = "verbs_{}.txt".format(args.dataset)

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

    for cap in args.caption:
        c = nlp(cap)[0]
        if args.slow:
            if c.lemma_ not in vocab.nouns:
                print("word {} is not in vocabulary nouns, skipping".format(c.text))
                continue
        else:
            if c.lemma_ not in vocab.verbs:
                print("word {} is not in vocabulary verbs, skipping".format(c.text))
                continue
        capid = vocab.return_idx([c.lemma_])
        with torch.no_grad():
            emb = capenc(capid, False).cpu().numpy()
        top5 = args.video_id

        # emb : (1, d) tensor
        emb = torch.Tensor(emb).to(device)
        for i, id in enumerate(top5):
            slowft, fastft = test_dset.get_features_from_id(id)
            (slowmap, fastmap), _ = ftenc(slowft.unsqueeze(0), fastft.unsqueeze(0))
            nmap = fastmap if not args.slow else slowmap
            # nmap : (1, d, T, H, W) tensor
            rel_map = get_map(nmap, emb, args.temperature)

            video_path = os.path.join(args.root_path, args.data_dir, "video{}.hdf5".format(id))
            place_map(video_path, rel_map, save_path)
            frame2vid(save_path, "video{}_{}.mp4".format(id, cap.replace(" ", "_").rstrip(".")))

    print("done visualizing!!")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--caption', type=str, nargs='+', required=True)
    parser.add_argument('--video_id', type=int, nargs='+', required=True, default=None)
    parser.add_argument('--temperature', type=float, default=0.001)
    parser.add_argument('--slow', action="store_true", help="inference done on nouns and the slow path")

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--root_path', type=str, default='/groups1/gaa50131/datasets/MSR-VTT')
    parser.add_argument('--feature_dir', type=str, default='features/sfnl152_k700_16f')
    parser.add_argument('--data_dir', type=str, default='TestHdf5')
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
