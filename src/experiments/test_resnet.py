import sys, os
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

from dataset.msrvtt import MSR_VTT_Resnet
from dataset.msvd import MSVD_Resnet
from utils import Logger, collater, sec2str
from model import MotionEncoder, CaptionEncoder
from vocab import Vocabulary
from train import validate


def main():
    args = parse_args()

    assert args.dataset in ["msvd", "msrvtt"]
    if args.dataset == "msrvtt":
        test_dset = MSR_VTT_Resnet(root_path=args.root_path, feature_dir=args.data_dir, mode='test')
    elif args.dataset == "msvd":
        test_dset = MSVD_Resnet(root_path=args.root_path, feature_dir=args.feature_dir, mode='test')
    vocab_path = "captions_{}.txt".format(args.dataset)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, collate_fn=collater, shuffle=False, drop_last=False)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(vocab_path)

    ftenc = MotionEncoder(ft_size=args.ft_size, out_size=args.out_size)
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

    validate("test phase", test_loader, ftenc, capenc, vocab, args)
    print("done with evaluation")

def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--root_path', type=str, default='/groups1/gaa50131/datasets/MSR-VTT')
    parser.add_argument('--data_dir', type=str, default='features/r50_k700_16f')
    parser.add_argument('--config_name', type=str, default="default", help='Name of configurations, will denote where to save data')

    # configurations of architectures (match loading model details)
    parser.add_argument('--rnn_type', type=str, default="GRU", help="architecture of rnn")
    parser.add_argument('--ft_size', type=int, default=2048, help="precomputed feature sizes")
    parser.add_argument('--emb_size', type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=1024, help="embedding size for output vectors for motion-semantic space")
    parser.add_argument('--imsize', type=int, default=224, help="image size (video size) to train on.")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size. must be a large number for negatives")

    # miscellaneous
    parser.add_argument('--n_cpu', type=int, default=32, help="number of threads for dataloading")
    parser.add_argument('--margin', type=float, default=0.2, help="margin for pairwise ranking loss")
    parser.add_argument('--max_len', type=int, default=30, help="max length of sentences")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()




