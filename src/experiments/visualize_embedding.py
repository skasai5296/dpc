import sys, os, glob
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from dataset.msrvtt import MSR_VTT_SlowFast, EmbedDataset_MSE
from dataset.msvd import MSVD_SlowFast
from utils import Logger, collater_mse, sec2str
from model import MotionEncoder, MSE, CaptionEncoder
from vocab import Vocabulary
# from train import validate # for resnet
from train_slowfast import validate # for slowfast


def main():
    args = parse_args()

    savedir = os.path.join("out", args.config_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir, 0o777)

    assert args.dataset in ["msrvtt", "msvd"]
    if args.dataset == "msrvtt":
        test_dset = MSR_VTT_SlowFast(root_path=args.root_path, feature_dir=args.feature_dir, mode='test')
    elif args.dataset == "msvd":
        test_dset = MSVD_SlowFast(root_path=args.root_path, feature_dir=args.feature_dir, mode='test')
    vocab_path = "captions_{}.txt".format(args.dataset)

    test_loader = DataLoader(test_dset, batch_size=args.batch_size, collate_fn=collater_mse, shuffle=False, drop_last=False)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(vocab_path)

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

    print("begin embedding visualization...", flush=True)
    begin = time.time()
    dset = EmbedDataset_MSE(test_loader, ftenc, capenc, vocab, args)
    print("eval dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    video = dset.embedded["video"]
    cap = dset.embedded["caption"]
    n_v = video.shape[0]
    n_c = cap.shape[0]
    all = np.concatenate([video, cap], axis=0)

    emb_file = os.path.join(savedir, "embedding_{}.npy".format(n_v))
    save_file = os.path.join(savedir, "{}.npy".format(args.method))
    vis_file = os.path.join(savedir, "{}.png".format(args.method))
    np.save(emb_file, all)
    print("saved embeddings to {}".format(emb_file), flush=True)
    dimension_reduction(emb_file, save_file, method=args.method)
    plot_embeddings(save_file, n_v, vis_file, method=args.method)

def dimension_reduction(numpyfile, dstfile, method="T-SNE"):
    all = np.load(numpyfile)
    begin = time.time()
    print("conducting {} on data...".format(method), flush=True)
    if method == "T-SNE":
        all = TSNE(n_components=2).fit_transform(all)
    elif method == "PCA":
        all = PCA(n_components=2).fit_transform(all)
    else:
        raise NotImplementedError()
    print("done | {} ".format(sec2str(time.time()-begin)), flush=True)
    np.save(dstfile, all)
    print("saved {} embeddings to {}".format(method, dstfile), flush=True)

def plot_embeddings(numpyfile, n_v, out_file, method="T-SNE"):
    all = np.load(numpyfile)
    assert all.shape[1] == 2
    fig = plt.figure(clear=True)
    fig.suptitle("visualization of embeddings using {}".format(method))
    plt.scatter(all[:n_v, 0], all[:n_v, 1], s=2, c="red", label="video")
    plt.scatter(all[n_v::5, 0], all[n_v::5, 1], s=2, c="blue", label="caption")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.savefig(out_file)
    print("saved {} plot to {}".format(method, out_file), flush=True)

def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--root_path', type=str, default='/groups1/gaa50131/datasets/MSR-VTT')
    parser.add_argument('--feature_dir', type=str, default='features/sfnl152_k700_16f')
    parser.add_argument('--config_name', type=str, default="default", help='Name of configuration, will denote where to load model and where to save output')
    parser.add_argument('--method', type=str, default="PCA", help='Name of dimensionality reduction method, should be {T-SNE | PCA}')

    # configurations of architectures (match loading model details)
    parser.add_argument('--rnn_type', type=str, default="GRU", help="architecture of rnn")
    parser.add_argument('--ft_size_slow', type=int, default=2048, help="precomputed feature sizes")
    parser.add_argument('--ft_size_fast', type=int, default=256, help="precomputed feature sizes")
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




