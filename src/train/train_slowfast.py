import sys, os, glob
import argparse
import time
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torchvision
import torchvision.transforms as transforms
import faiss

from dataset.msrvtt import MSR_VTT_SlowFast, EmbedDataset_MSE
from dataset.msvd import MSVD_SlowFast
from utils import Logger, AlignmentLoss, NPairLoss, NPairLoss_Caption, weight_init, collater_mse, sec2str, get_pos, AverageMeter
from model import MSE, CaptionEncoder, ModalDiscriminator
from vocab import Vocabulary


def train(epoch, loader, ftenc, capenc, dis, optimizer, losses, vocab, args):
    begin = time.time()
    maxit = int(len(loader.dataset) / args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    meter_gl = AverageMeter("l_gl", ":.5f")
    meter_sem = AverageMeter("l_sem", ":.5f")
    meter_mot = AverageMeter("l_mot", ":.5f")
    meter_cap = AverageMeter("l_cap", ":.5f")
    meter_adv = AverageMeter("l_adv", ":.5f")
    meter_dv = AverageMeter("D(v)", ":1.5f")
    meter_dc = AverageMeter("D(c)", ":1.5f")
    for it, data in enumerate(loader):
        """feature, id, caption, cap_id"""

        feature_slow = data["feature_slow"]
        feature_fast = data["feature_fast"]
        id = data["id"]
        captions = data["caption"]
        # choose only one caption per video
        indexes = [np.random.randint(0, num) for num in data["number"]]
        caption = [cap[idx] for cap, idx in zip(captions, indexes)]
        cap_id = data["cap_id"]
        # get embeddings for captions
        caption_idx = vocab.return_idx(caption)

        ftenc.zero_grad()
        capenc.zero_grad()

        feature_slow = feature_slow.to(device)
        feature_fast = feature_fast.to(device)
        (sem_map, mot_map), global_emb = ftenc(feature_slow, feature_fast)
        caption_idx = caption_idx.to(device)
        cap_emb = capenc(caption_idx)

        # n-pair loss
        l_gl = losses[0](global_emb, cap_emb)
        l_gl.backward()
        optimizer.step()
        meter_gl.update(l_gl.item())

        # alignment loss for motion and semantic spaces
        if args.lambda_noun > 0 or args.lambda_verb > 0:
            (sem_map, mot_map), global_emb = ftenc(feature_slow, feature_fast)

            pos = get_pos(caption)
            nouns = pos["noun"]
            neg_nouns = [vocab.replace_noun(captions[i], noun) for i, noun in enumerate(nouns)]
            verbs = pos["verb"]
            neg_verbs = [vocab.replace_verb(captions[i], verb) for i, verb in enumerate(verbs)]

            # get embeddings for positive nouns
            nouns = vocab.return_idx(nouns)
            # get embeddings for positive verbs
            verbs = vocab.return_idx(verbs)
            # get embeddings for negative nouns
            neg_nouns = vocab.return_idx(neg_nouns)
            # get embeddings for negative verbs
            neg_verbs = vocab.return_idx(neg_verbs)

            pos_noun_emb = capenc(nouns, False)
            neg_noun_emb = capenc(neg_nouns, False)
            pos_verb_emb = capenc(verbs, False)
            neg_verb_emb = capenc(neg_verbs, False)

            # sum for dataparallel
            l_align_sem = losses[1](sem_map, pos_noun_emb, neg_noun_emb)[1].sum()
            l_align_mot = losses[1](mot_map, pos_verb_emb, neg_verb_emb)[1].sum()
            l_align_sem = args.lambda_noun * l_align_sem
            l_align_mot = args.lambda_verb * l_align_mot
            (l_align_sem + l_align_mot).backward()
            optimizer.step()
            meter_sem.update(l_align_sem.item())
            meter_mot.update(l_align_mot.item())

        # loss for expanding caption space
        if args.align_caption > 0:
            cap_emb = capenc(caption_idx)
            # get another positive caption
            indexes2 = [(idx + np.random.randint(1, num)) % num for num, idx in zip(data["number"], indexes)]
            poscap = [cap[idx] for cap, idx in zip(captions, indexes2)]
            poscap_idx = vocab.return_idx(poscap)
            poscap_idx = poscap_idx.to(device)
            cap_emb_2 = capenc(poscap_idx)
            l_cap = losses[2](cap_emb, cap_emb_2)
            l_cap = args.align_caption * l_cap
            l_cap.backward()
            optimizer.step()
            meter_cap.update(l_cap.item())

        if args.grad_clip > 0:
            clip_grad_norm_(ftenc.parameters(), args.grad_clip)

        if args.lambda_adv > 0:
            pos_label = 1
            neg_label = 0
            d_loss = 0
            g_loss = 0
            loss = 0
            if args.label_smoothing:
                pos_label = np.random.uniform(0.8, 1.2)
                neg_label = np.random.uniform(0, 0.3)
            label = torch.full((), pos_label, device=device)
            dis.zero_grad()
            # train discriminator on visual embeddings (real)
            out = dis(global_emb.detach())
            l_adv_v1 = losses[3](out.mean(), label)
            l_adv_v1 = args.lambda_adv * l_adv_v1
            l_adv_v1.backward()
            D_v1 = out.mean().item()
            d_loss += l_adv_v1.item()
            # train discriminator on caption embeddings (fake)
            label.fill_(neg_label)
            out = dis(cap_emb.detach())
            l_adv_c1 = losses[3](out.mean(), label)
            l_adv_c1 = args.lambda_adv * l_adv_c1
            l_adv_c1.backward()
            optimizer.step()
            D_c1 = out.mean().item()
            d_loss += l_adv_c1.item()

            capenc.zero_grad()
            ftenc.zero_grad()
            # train generator on caption embeddings (real)
            cap_emb = capenc(caption_idx)
            label.fill_(pos_label)
            out = dis(cap_emb)
            l_adv_c2 = losses[3](out.mean(), label)
            l_adv_c2 = args.lambda_adv * l_adv_c2
            l_adv_c2.backward()
            dis.zero_grad()
            optimizer.step()
            D_c2 = out.mean().item()
            g_loss += l_adv_c2.item()
            # train generator on visual embeddings (fake)
            _, global_emb = ftenc(feature_slow, feature_fast)
            label.fill_(neg_label)
            out = dis(global_emb)
            l_adv_v2 = losses[3](out.mean(), label)
            l_adv_v2 = args.lambda_adv * l_adv_v2
            l_adv_v2.backward()
            dis.zero_grad()
            optimizer.step()
            D_v2 = out.mean().item()
            g_loss += l_adv_v2.item()

            meter_adv.update(d_loss + g_loss)
            meter_dv.update((D_v1 + D_v2) / 2)
            meter_dc.update((D_c1 + D_c2) / 2)

        if it % args.log_every == args.log_every-1:
            print("{} | epoch {} | {:06d}/{:06d} iterations | {}, {}, {}, {}, {}, {}, {}".format(
                        sec2str(time.time()-begin), epoch, it+1, maxit, str(meter_gl), str(meter_sem), str(meter_mot), str(meter_cap), str(meter_adv), str(meter_dv), str(meter_dc)), flush=True)
            meter_gl.reset()
            meter_sem.reset()
            meter_mot.reset()
            meter_cap.reset()
            meter_adv.reset()
            meter_dv.reset()
            meter_dc.reset()

def validate(epoch, loader, ftenc, capenc, vocab, args):
    begin = time.time()
    print("begin evaluation for epoch {}".format(epoch), flush=True)
    dset = EmbedDataset_MSE(loader, ftenc, capenc, vocab, args)
    print("eval dataset created | {} ".format(sec2str(time.time()-begin)), flush=True)
    video = dset.embedded["video"]
    semantic = dset.embedded["semantic"]
    motion = dset.embedded["motion"]
    cap = dset.embedded["caption"]
    rawcap = dset.embedded["raw_caption"]
    video_ids = dset.embedded["video_id"]
    cap_ids = dset.embedded["cap_id"]

    pos = get_pos(rawcap)
    nouns = vocab.return_idx(pos["noun"])
    verbs = vocab.return_idx(pos["verb"])
    noun_emb = capenc(nouns, False).cpu().detach().numpy()
    verb_emb = capenc(verbs, False).cpu().detach().numpy()
    assert video.shape == semantic.shape == motion.shape
    assert cap.shape == noun_emb.shape == verb_emb.shape

    nv = video.shape[0]
    nc = cap.shape[0]
    dim = video.shape[1]

    print("# videos: {}, # captions: {}, dimension: {}".format(nv, nc, dim), flush=True)
    sem_weight = [args.sem_weight] if args.sem_weight > 0 else [0, 1, 10, 50, 100, 1000]
    mot_weight = [args.mot_weight] if args.mot_weight > 0 else [0, 1, 10, 50, 100, 1000]
    bestscore = 0
    bestdict = None
    for sem in sem_weight:
        for mot in mot_weight:
            # vid2cap
            dist = video @ cap.T
            if sem > 0:
                dist += sem * semantic @ noun_emb.T
            if mot > 0:
                dist += mot * motion @ verb_emb.T
            I = np.argsort(dist, axis=1)[:, ::-1]
            D = np.take_along_axis(dist, I, axis=1)

            data = {}
            allrank = []
            # captions per instance
            k_instances = 20 if args.dataset == "msrvtt" else 5
            for i in range(k_instances):
                gt = (np.arange(nv) * k_instances).reshape(-1, 1) + i
                rank = np.where(I == gt)[1]
                allrank.append(rank)
            allrank = np.stack(allrank)
            allrank = np.amin(allrank, 0)
            for rank in [1, 5, 10, 20]:
                data["v2c_recall@{}".format(rank)] = round(100 * np.sum(allrank < rank) / len(allrank), 4)
            data["v2c_median@r"] = round(np.median(allrank) + 1, 1)
            data["v2c_mean@r"] = round(np.mean(allrank), 2)

            # cap2vid
            dist = dist.T
            I = np.argsort(dist, axis=1)[:, ::-1]
            D = np.take_along_axis(dist, I, axis=1)
            gt = np.arange(nc).reshape(-1, 1) // k_instances
            allrank = np.where(I == gt)[1]
            for rank in [1, 5, 10, 20]:
                data["c2v_recall@{}".format(rank)] = round(100 * np.sum(allrank < rank) / len(allrank), 4)
            data["c2v_median@r"] = round(np.median(allrank) + 1, 1)
            data["c2v_mean@r"] = round(np.mean(allrank), 2)
            data["sem"] = sem
            data["mot"] = mot

            score = 0
            for rank in [1, 5, 10]:
                score += data["v2c_recall@{}".format(rank)] + data["c2v_recall@{}".format(rank)]
            if score > bestscore:
                bestdict = data
                bestscore = score
    print("-"*50)
    print("results of cross-modal retrieval")
    for key, val in bestdict.items():
        print("{}:\t{:.2f}".format(key, val), flush=True)
    print("-"*50)
    return bestdict

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

    assert args.dataset in ["msrvtt", "msvd"]
    if args.dataset == "msrvtt":
        train_dset = MSR_VTT_SlowFast(root_path=args.root_path, feature_dir=args.data_dir, mode='train')
        val_dset = MSR_VTT_SlowFast(root_path=args.root_path, feature_dir=args.data_dir, mode='val')
    elif args.dataset == "msvd":
        train_dset = MSVD_SlowFast(root_path=args.root_path, feature_dir=args.data_dir, mode='train')
        val_dset = MSVD_SlowFast(root_path=args.root_path, feature_dir=args.data_dir, mode='val')
    vocab_path = "captions_{}.txt".format(args.dataset)
    noun_path = "nouns_{}.txt".format(args.dataset)
    verb_path = "verbs_{}.txt".format(args.dataset)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, collate_fn=collater_mse, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, collate_fn=collater_mse, shuffle=False, drop_last=False)

    vocab = Vocabulary(max_len=args.max_len)
    vocab.load_vocab(vocab_path)
    vocab.load_entities(noun_path, verb_path)

    ftenc = MSE(ft_size_slow=args.ft_size_slow, ft_size_fast=args.ft_size_fast, out_size=args.out_size)
    capenc = CaptionEncoder(vocab_size=len(vocab), emb_size=args.emb_size, out_size=args.out_size, rnn_type=args.rnn_type)
    dis = ModalDiscriminator(args.out_size)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    ftenc = ftenc.to(device)
    capenc = capenc.to(device)
    if args.lambda_adv > 0:
        dis = dis.to(device)

    cfgs = [{'params' : ftenc.parameters(), 'lr' : args.lr_cnn},
            {'params' : capenc.parameters(), 'lr' : args.lr_rnn},
            {'params': dis.parameters(), 'lr': args.lr_cnn},
    ]
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
            NPairLoss(margin=args.margin, method=args.method, scale=args.scale),
            # contrastive component loss
            AlignmentLoss(margin=args.margin, scale=args.scale),
            # caption embedding loss
            NPairLoss_Caption(margin=args.margin, method="max", scale=args.scale),
            # distribution alignment loss
            nn.BCELoss(),
    ]

    if args.resume:
        ckpt_dir = os.path.join("models", args.config_name)
        checkpoint = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[0])
        print("loading model and optimizer checkpoint from {} ...".format(checkpoint), flush=True)
        ckpt = torch.load(checkpoint)
        ftenc.load_state_dict(ckpt["encoder_state"])
        capenc.load_state_dict(ckpt["decoder_state"])
        dis.load_state_dict(ckpt["discriminator_state"])
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
    dis = nn.DataParallel(dis)
    losses[1] = nn.DataParallel(losses[1])

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
        train(ep+1, train_loader, ftenc, capenc, dis, optimizer, losses, vocab, args)
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
                "discriminator_state": dis.module.state_dict(),
                "optimizer_state": optimizer.state_dict()
        }
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
    parser.add_argument('--data_dir', type=str, default='features/sfnl152_k700_16f')
    parser.add_argument('--config_name', type=str, default="default", help='Name of configurations, will denote where to save data')
    parser.add_argument('--resume', action="store_true", help='denotes if to continue training, will use config name')

    # configurations of models
    parser.add_argument('--rnn_type', type=str, default="GRU", help="architecture of rnn")
    parser.add_argument('--ft_size_slow', type=int, default=2048, help="precomputed feature sizes")
    parser.add_argument('--ft_size_fast', type=int, default=256, help="precomputed feature sizes")
    parser.add_argument('--emb_size', type=int, default=300, help="embedding size of vocabulary")
    parser.add_argument('--out_size', type=int, default=1024, help="embedding size for output vectors for motion-semantic space")

    # training config
    parser.add_argument('--n_cpu', type=int, default=32, help="number of threads for dataloading")
    parser.add_argument('--margin', type=float, default=0.2, help="margin for pairwise ranking loss")
    parser.add_argument('--scale', type=float, default=1e5, help="scale parameter for loss components")
    parser.add_argument('--method', type=str, default="sum", help="how many negatives to get in batch. [max, sum, top10, top25]")
    parser.add_argument('--max_epochs', type=int, default=30, help="max number of epochs to train for")
    parser.add_argument('--max_len', type=int, default=30, help="max length of sentences")
    parser.add_argument('--log_every', type=int, default=10, help="log every x iterations")
    parser.add_argument('--no_cuda', action='store_true', help="disable cuda")
    parser.add_argument('--label_smoothing', action='store_true', help="induce label smoothing during adversarial training")
    parser.add_argument('--sem_weight', type=float, default=0, help="reranking factor using semantic space")
    parser.add_argument('--mot_weight', type=float, default=0, help="reranking factor using motion space")

    # weight loss
    parser.add_argument('--lambda_noun', type=float, default=0., help="weight for contrastive noun loss")
    parser.add_argument('--lambda_verb', type=float, default=0., help="weight for contrastive verb loss")
    parser.add_argument('--lambda_adv', type=float, default=0., help="weight for adversarial modal alignment loss")
    parser.add_argument('--align_caption', type=float, default=0., help="weight for caption space alignment")

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
