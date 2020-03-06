import sys, os, glob
import argparse
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torchvision
import torchvision.transforms as transforms
import faiss

from dataset.msrvtt import MSR_VTT_Resnet, EmbedDataset
from dataset.msvd import MSVD_Resnet
from utils import Logger, AlignmentLoss, NPairLoss, NPairLoss_Caption, weight_init, collater, sec2str, get_pos
from model import MotionEncoder, CaptionEncoder
from vocab import Vocabulary


def train(epoch, loader, ftenc, capenc, optimizer, losses, vocab, args):
    begin = time.time()
    maxit = int(len(loader.dataset) / args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    cumloss = 0
    for it, data in enumerate(loader):
        """feature, id, caption, cap_id"""

        feature = data["feature"]
        id = data["id"]
        captions = data["caption"]
        # choose only one caption per video
        indexes = [np.random.randint(0, num) for num in data["number"]]
        caption = [cap[idx] for cap, idx in zip(captions, indexes)]
        cap_id = data["cap_id"]
        # get embeddings for captions
        caption_idx = vocab.return_idx(caption)

        optimizer.zero_grad()
        feature = feature.to(device)
        ft_map, global_emb = ftenc(feature)
        caption_idx = caption_idx.to(device)
        cap_emb = capenc(caption_idx)

        loss = 0
        # n-pair loss
        loss += losses[0](global_emb, cap_emb)

        loss.backward()

        if args.grad_clip > 0:
            clip_grad_norm_(ftenc.parameters(), args.grad_clip)
        optimizer.step()
        cumloss += loss.item()

        if it % args.log_every == args.log_every-1:
            print("epoch {} | {} | {:06d}/{:06d} iterations | loss: {:.08f}".format(epoch, sec2str(time.time()-begin), it+1, maxit, cumloss/args.log_every), flush=True)
            cumloss = 0

def validate(epoch, loader, ftenc, capenc, vocab, args):
    begin = time.time()
    print("begin evaluation for epoch {}".format(epoch), flush=True)
    dset = EmbedDataset(loader, ftenc, capenc, vocab, args)
    print("eval dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    video = dset.embedded["video"]
    cap = dset.embedded["caption"]
    video_ids = dset.embedded["video_id"]
    cap_ids = dset.embedded["cap_id"]

    nd = video.shape[0]
    nq = cap.shape[0]
    d = video.shape[1]
    cpu_index = faiss.IndexFlatIP(d)

    print("# videos: {}, # captions: {}, dimension: {}".format(nd, nq, d), flush=True)
    # vid2cap
    cpu_index.add(cap)
    D, I = cpu_index.search(video, nq)
    data = {}
    allrank = []

    # captions per instance
    k_instances = 20
    for i in range(k_instances):
        gt = (np.arange(nd) * k_instances).reshape(-1, 1) + i
        rank = np.where(I == gt)[1]
        allrank.append(rank)
    allrank = np.stack(allrank)
    allrank = np.amin(allrank, 0)
    for rank in [1, 5, 10, 20]:
        data["v2c_recall@{}".format(rank)] = round(100 * np.sum(allrank < rank) / len(allrank), 4)
    data["v2c_median@r"] = round(np.median(allrank) + 1, 1)
    data["v2c_mean@r"] = round(np.mean(allrank), 2)

    # cap2vid
    cpu_index.reset()
    cpu_index.add(video)
    D, I = cpu_index.search(cap, nd)
    gt = np.arange(nq).reshape(-1, 1) // k_instances
    allrank = np.where(I == gt)[1]
    for rank in [1, 5, 10, 20]:
        data["c2v_recall@{}".format(rank)] = round(100 * np.sum(allrank < rank) / len(allrank), 4)
    data["c2v_median@r"] = round(np.median(allrank) + 1, 1)
    data["c2v_mean@r"] = round(np.mean(allrank), 2)

    print("-"*50)
    print("results of cross-modal retrieval")
    for key, val in data.items():
        print("{}:\t{:.2f}".format(key, val), flush=True)
    print("-"*50)
    return data

def main():
    args = parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    assert args.dataset in ["msvd", "msrvtt"]
    if args.dataset == "msrvtt":
        train_dset = MSR_VTT_Resnet(root_path=args.root_path, feature_dir=args.data_dir, mode='train', align_size=(10, 8, 10))
        val_dset = MSR_VTT_Resnet(root_path=args.root_path, feature_dir=args.data_dir, mode='val', align_size=(10, 8, 10))
    elif args.dataset == "msvd":
        train_dset = MSVD_Resnet(root_path=args.root_path, feature_dir=args.data_dir, mode='train', align_size=(10, 8, 10))
        val_dset = MSVD_Resnet(root_path=args.root_path, feature_dir=args.data_dir, mode='val', align_size=(10, 8, 10))
    vocab_path = "captions_{}.txt".format(args.dataset)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, collate_fn=collater, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, collate_fn=collater, shuffle=False, drop_last=False)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(vocab_path)

    ftenc = MotionEncoder(ft_size=args.ft_size, out_size=args.out_size)
    capenc = CaptionEncoder(vocab_size=len(vocab), emb_size=args.emb_size, out_size=args.out_size, rnn_type=args.rnn_type)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ftenc = ftenc.to(device)
    capenc = capenc.to(device)

    cfgs = [{'params' : ftenc.parameters(), 'lr' : args.lr_cnn},
            {'params' : capenc.parameters(), 'lr' : args.lr_rnn}]
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(cfgs, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(cfgs, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(cfgs, alpha=args.alpha, weight_decay=args.weight_decay)
    if args.scheduler == 'Plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.dampen_factor, patience=args.patience, verbose=True)
    elif args.scheduler == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.patience, gamma=args.dampen_factor)
    losses = [
            # global alignment loss for joint space
            NPairLoss(margin=args.margin, method=args.method),
            # contrastive component loss
            AlignmentLoss(margin=args.margin),
            # caption embedding loss
            NPairLoss_Caption(margin=args.margin, method=args.method),
    ]

    if args.resume:
        ckpt_dir = os.path.join("models", args.config_name)
        checkpoint = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
        print("loading model and optimizer checkpoint from {} ...".format(checkpoint), flush=True)
        ckpt = torch.load(checkpoint)
        ftenc.load_state_dict(ckpt["encoder_state"])
        capenc.load_state_dict(ckpt["decoder_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if args.scheduler != 'None':
            scheduler.load_state_dict(ckpt["scheduler_state"])
        offset = ckpt["epoch"]
        data = ckpt["stats"]
        bestscore = 0
        for rank in [1, 5, 10, 20]:
            bestscore += int(100 * (data["v2c_recall@{}".format(rank)] + data["c2v_recall@{}".format(rank)]))
    else:
        offset = 0
        bestscore = -1
    ftenc = nn.DataParallel(ftenc)
    capenc = nn.DataParallel(capenc)

    # for logging metrics
    logdir = os.path.join("out", args.config_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir, 0o777)
    metric = ["epoch", "lr"]
    for name in ["v2c", "c2v"]:
        for rank in [1, 5, 10, 20]:
            metric.append("{}_recall@{}".format(name, rank))
        for w in ["median", "mean"]:
            metric.append("{}_{}@r".format(name, w))
    val_logger = Logger(os.path.join(logdir, "stats.tsv"), metric)

    es_cnt = 0

    assert offset < args.max_epochs
    for ep in range(offset, args.max_epochs):
        train(ep+1, train_loader, ftenc, capenc, optimizer, losses, vocab, args)
        data = validate(ep+1, val_loader, ftenc, capenc, vocab, args)
        data["epoch"] = ep+1
        data["lr"] = optimizer.param_groups[0]["lr"]
        val_logger.log(data)
        totalscore = 0
        for rank in [1, 5, 10, 20]:
            totalscore += int(100 * (data["v2c_recall@{}".format(rank)] + data["c2v_recall@{}".format(rank)]))
        if args.scheduler == 'Plateau':
            scheduler.step(totalscore)
        elif args.scheduler != 'None':
            scheduler.step()

        # save checkpoint
        ckpt = {
                "stats": data,
                "epoch": ep+1,
                "encoder_state": ftenc.module.state_dict(),
                "decoder_state": capenc.module.state_dict(),
                "optimizer_state": optimizer.state_dict()
        }
        if args.scheduler != 'None':
            ckpt["scheduler_state"] = scheduler.state_dict()
        savedir = os.path.join("models", args.config_name)
        if not os.path.exists(savedir):
            os.makedirs(savedir, 0o777)

        savepath = os.path.join(savedir, "epoch_{:04d}_score_{:05d}.ckpt".format(ep+1, totalscore))
        if totalscore > bestscore:
            # delete original files in folder
            files = glob.glob(os.path.join(savedir, "*"))
            for f in files:
                os.remove(f)
            print("score: {:05d}, saving model and optimizer checkpoint to {} ...".format(totalscore, savepath), flush=True)
            bestscore = totalscore
            torch.save(ckpt, savepath)
            es_cnt = 0
        else:
            print("score: {:05d}, no improvement from best score of {:05d}, not saving".format(totalscore, bestscore), flush=True)
            es_cnt += 1
            if es_cnt == args.es_cnt:
                print("early stopping at epoch {} because of no improvement for {} epochs".format(ep+1, args.es_cnt))
                break
        print("done for epoch {:04d}".format(ep+1), flush=True)

    print("done training!!")



def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--root_path', type=str, default='/groups1/gaa50131/datasets/MSR-VTT')
    parser.add_argument('--data_dir', type=str, default='features/r50_k700_16f')
    parser.add_argument('--config_name', type=str, default="default", help='Name of configurations, will denote where to save data')
    parser.add_argument('--resume', action="store_true", help='denotes if to continue training, will use config name')

    # configurations of models
    parser.add_argument('--rnn_type', type=str, default="GRU", help="architecture of rnn")
    parser.add_argument('--ft_size', type=int, default=2048, help="precomputed feature sizes")
    parser.add_argument('--emb_size', type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=1024, help="embedding size for output vectors for motion-semantic space")

    # training config
    parser.add_argument('--n_cpu', type=int, default=32, help="number of threads for dataloading")
    parser.add_argument('--margin', type=float, default=0.2, help="margin for pairwise ranking loss")
    parser.add_argument('--method', type=str, default="sum", help="how many negatives to get in batch. [max, sum, top10, top25]")
    parser.add_argument('--max_epochs', type=int, default=30, help="max number of epochs to train for")
    parser.add_argument('--max_len', type=int, default=30, help="max length of sentences")
    parser.add_argument('--log_every', type=int, default=10, help="log every x iterations")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")

    # hyperparams
    parser.add_argument('--imsize', type=int, default=224, help="image size (video size) to train on.")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size. must be a large number for negatives")
    parser.add_argument('--lr_cnn', type=float, default=1e-3, help="learning rate of cnn")
    parser.add_argument('--lr_rnn', type=float, default=1e-3, help="learning rate of rnn")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum for SGD")
    parser.add_argument('--alpha', type=float, default=0.99, help="alpha for RMSprop")
    parser.add_argument('--beta1', type=float, default=0.9, help="beta1 for Adam")
    parser.add_argument('--beta2', type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument('--optimizer', type=str, default='Adam', help="optimizer, [SGD, Adam, RMSprop]")
    parser.add_argument('--scheduler', type=str, default='None', help="learning rate scheduler, [Plateau, Step, None]")
    parser.add_argument('--weight_decay', type=float, default=0, help="weight decay of all parameters, unrecommended")
    parser.add_argument('--grad_clip', type=float, default=2., help="gradient norm clipping")
    parser.add_argument('--patience', type=int, default=5, help="patience of learning rate scheduler")
    parser.add_argument('--es_cnt', type=int, default=10, help="threshold epoch for early stopping")
    parser.add_argument('--dampen_factor', type=float, default=0.1, help="dampening factor for learning rate scheduler")

    parser.add_argument('--random_seed', type=int, default=None, help="random seed for training, for measuring effectiveness")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
