import sys, os
import shutil
import argparse
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from dataset.msrvtt import MSR_VTT_SlowFast, EmbedDataset_MSE
from dataset.msvd import MSVD_SlowFast
from utils import Logger, collater_mse, sec2str
from model import MotionEncoder, MSE, CaptionEncoder
from vocab import Vocabulary
from train_slowfast import validate

def main():
    args = parse_args()

    assert args.caption is not None
    assert args.dataset in ["msvd", "msrvtt"]
    if args.dataset == "msrvtt":
        test_dset = MSR_VTT_SlowFast(root_path=args.root_path, feature_dir=args.data_dir, mode='test')
    elif args.dataset == "msvd":
        test_dset = MSVD_SlowFast(root_path=args.root_path, feature_dir=args.feature_dir, mode='test')
    vocab_path = "captions_{}.txt".format(args.dataset)
    noun_path = "nouns_{}.txt".format(args.dataset)
    verb_path = "verbs_{}.txt".format(args.dataset)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, collate_fn=collater_mse, shuffle=False, drop_last=False)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(vocab_path)
    vocab.load_entities(noun_path, verb_path)

    ftenc = MSE(ft_size_slow=args.ft_size_slow, ft_size_fast=args.ft_size_fast, out_size=args.out_size)
    capenc = CaptionEncoder(vocab_size=len(vocab), emb_size=args.emb_size, out_size=args.out_size, rnn_type=args.rnn_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ftenc = ftenc.to(device)
    capenc = capenc.to(device)

    ckpt_dir = os.path.join("models", args.config_name)
    checkpoint = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
    print("loading model checkpoint from {} ...".format(checkpoint), flush=True)
    ckpt = torch.load(checkpoint, map_location=device)
    ftenc.load_state_dict(ckpt["encoder_state"])
    capenc.load_state_dict(ckpt["decoder_state"])

    ftenc = nn.DataParallel(ftenc)
    capenc = nn.DataParallel(capenc)

    cap_id = vocab.return_idx(args.caption)
    with torch.no_grad():
        emb = capenc(cap_id).cpu().numpy()

    begin = time.time()
    print("creating test dataset...")
    # use all data
    dset = EmbedDataset_MSE(test_loader, ftenc, capenc, vocab, args, max_instances=3000)
    print("test dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    video = dset.embedded["video"]
    dist_mat = emb @ video.T
    mp4_dir = os.path.join(args.root_path, "TestVideo")
    savedir = os.path.join("out", "nearestneighbor")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    for cap, dist in zip(args.caption, dist_mat):
        I = np.argsort(dist, axis=0)[::-1][:5]
        print("nearest 5 to caption \'{}\': {}".format(cap, 7010 + I))
        D = np.take_along_axis(dist, I, axis=0)
        print("distances: {}".format(D))
        for i, id in enumerate(I):
            savepath = os.path.join(savedir, cap.rstrip(".").replace(" ", "_") + "_{}.mp4".format(i))
            id += 7010
            shutil.copyfile(os.path.join(mp4_dir, "video{}.mp4".format(id)), savepath)

    print("done with evaluation")

def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--root_path', type=str, default='/groups1/gaa50131/datasets/MSR-VTT')
    parser.add_argument('--data_dir', type=str, default='features/sfnl152_k700_16f')
    parser.add_argument('--config_name', type=str, default="default", help='Name of configurations, will denote where to fetch model')
    parser.add_argument('--caption', type=str, nargs='+', help='captions to retrieve from')

    # configurations of architectures (match loading model details)
    parser.add_argument('--rnn_type', type=str, default="GRU", help="architecture of rnn")
    parser.add_argument('--ft_size_slow', type=int, default=2048, help="precomputed feature sizes")
    parser.add_argument('--ft_size_fast', type=int, default=256, help="precomputed feature sizes")
    parser.add_argument('--emb_size', type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=1024, help="embedding size for output vectors for motion-semantic space")
    parser.add_argument('--imsize', type=int, default=224, help="image size (video size) to train on.")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size. must be a large number for negatives")
    parser.add_argument('--sem_weight', type=float, default=0, help="reranking factor using semantic space")
    parser.add_argument('--mot_weight', type=float, default=0, help="reranking factor using motion space")

    # miscellaneous
    parser.add_argument('--n_cpu', type=int, default=32, help="number of threads for dataloading")
    parser.add_argument('--margin', type=float, default=0.2, help="margin for pairwise ranking loss")
    parser.add_argument('--max_len', type=int, default=30, help="max length of sentences")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()




