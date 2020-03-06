import time
import random
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import spacy

nlp = spacy.load('en_core_web_sm')

class Logger():

    def __init__(self, path, header):
        path = Path(path)
        self.log_file = path.open('w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


# loss for global image
class NPairLoss(nn.Module):
    def __init__(self, margin=0.2, method='max', scale=1e5):
        super(NPairLoss, self).__init__()
        self.margin = margin
        self.method = method
        self.scale = scale

    # im, sen : (n_samples, dim)
    def forward(self, im, sen):
        assert im.size() == sen.size()
        assert self.method in ["max", "sum", "top10", "top25"]
        n_samples = im.size(0)
        # sim_mat : (n_samples, n_samples), scale for loss visualization
        sim_mat = im.mm(sen.t()) * self.scale
        # pos : (n_samples, 1)
        pos = sim_mat.diag().view(-1, 1)
        # positive1, 2 : (n_samples, n_samples)
        positive1 = pos.expand_as(sim_mat)
        positive2 = pos.t().expand_as(sim_mat)

        # mask for diagonals
        mask = (torch.eye(n_samples) > 0.5).to(sim_mat.device)
        # caption negatives
        lossmat_i = (self.margin + sim_mat - positive1).clamp(min=0).masked_fill(mask, 0)
        # image negatives
        lossmat_c = (self.margin + sim_mat - positive2).clamp(min=0).masked_fill(mask, 0)
        # max of hinges loss
        if self.method == "max":
            # lossmat : (n_samples)
            lossmat_i = lossmat_i.max(dim=1)[0]
            lossmat_c = lossmat_c.max(dim=0)[0]
        # sum of hinges loss
        elif self.method == "sum":
            lossmat_i /= n_samples - 1
            lossmat_c /= n_samples - 1
        elif self.method == "top10":
            lossmat_i = lossmat_i.topk(10, dim=1)[0] / 10
            lossmat_c = lossmat_c.topk(10, dim=1)[0] / 10
        elif self.method == "top25":
            lossmat_i = lossmat_i.topk(25, dim=1)[0] / 25
            lossmat_c = lossmat_c.topk(25, dim=1)[0] / 25

        loss = lossmat_i.sum() + lossmat_c.sum()

        return loss / n_samples


# loss for caption space
class NPairLoss_Caption(nn.Module):
    def __init__(self, margin=0.2, method='max', scale=1e5):
        super(NPairLoss_Caption, self).__init__()
        self.margin = margin
        self.method = method
        self.scale = scale

    # cap1, cap2 : (n_samples, dim)
    def forward(self, cap1, cap2):
        n_samples = cap1.size(0)
        # sim_mat : (n_samples, n_samples), scale for loss visualization
        sim_mat = cap1.mm(cap2.t()) * self.scale
        # pos : (n_samples, 1)
        pos = sim_mat.diag().view(-1, 1)
        # positive1 : (n_samples, n_samples)
        positive1 = pos.expand_as(sim_mat)
        positive2 = pos.t().expand_as(sim_mat)

        # mask for diagonals
        mask = (torch.eye(n_samples) > 0.5).to(sim_mat.device)
        # caption negatives
        lossmat_c1 = (self.margin + sim_mat - positive1).clamp(min=0).masked_fill(mask, 0)
        lossmat_c2 = (self.margin + sim_mat - positive2).clamp(min=0).masked_fill(mask, 0)
        # max of hinges loss
        if self.method == "max":
            # lossmat : (n_samples)
            lossmat_c1 = lossmat_c1.max(dim=1)[0]
            lossmat_c2 = lossmat_c2.max(dim=0)[0]
        # sum of hinges loss
        elif self.method == "sum":
            lossmat_c1 /= n_samples
            lossmat_c2 /= n_samples
        elif self.method == "top10":
            lossmat_c1 = lossmat_c1.topk(10, dim=1)[0] / 10
            lossmat_c2 = lossmat_c2.topk(10, dim=1)[0] / 10
        elif self.method == "top25":
            lossmat_c1 = lossmat_c1.topk(25, dim=1)[0] / 25
            lossmat_c2 = lossmat_c2.topk(25, dim=1)[0] / 25

        loss = lossmat_c1.sum() + lossmat_c2.sum()

        return loss / n_samples

# loss for alignment, (T, H, W) ranking loss
class AlignmentLoss(nn.Module):
    def __init__(self, margin=0.2, temperature=0.001, scale=1e5):
        super(AlignmentLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.loss = nn.TripletMarginLoss(margin=margin)
        self.scale = scale

    # map : (B, C, T, H, W), relevance map
    # pos, neg : (B, C), normalized on the C dimension
    def forward(self, ft_map, pos, neg):
        assert ft_map.size()[0:2] == neg.size() == pos.size()
        B, C, T, H, W = ft_map.size()
        # ft_map : (B, THW, C)
        ft_map = ft_map.view(B, C, -1).transpose(1, 2)
        # emb_exp : (B, C, 1)
        emb_exp = pos.unsqueeze(-1)
        # map : (B, THW)
        map = (ft_map @ emb_exp).squeeze(-1)
        # map : (B, THW), softmax over last three dimensions with temperature
        map = F.softmax(map / self.temperature, dim=1)
        # posmap, negmap : (B, THW), scale for loss visualization
        posmap = (ft_map @ pos.unsqueeze(-1)).squeeze(-1) * self.scale
        negmap = (ft_map @ neg.unsqueeze(-1)).squeeze(-1) * self.scale
        # lossmap : (B, THW)
        lossmap = torch.clamp(self.margin - posmap + negmap, min=0)
        loss = torch.sum(lossmap * map) / B
        return map.view(B, T, H, W), loss


def get_pos(captions):
    nouns = []
    verbs = []
    for cap in captions:
        proc = nlp(cap)
        noun_list = [ent.lemma_ for ent in proc if ent.pos_ == 'NOUN']
        verb_list = [ent.lemma_ for ent in proc if ent.pos_ == 'VERB' and ent.lemma_ not in ['be', 'is', 'are']]
        if len(noun_list) > 0:
            nouns.append(random.choice(noun_list))
        else:
            nouns.append("<unk>")
        if len(verb_list) > 0:
            verbs.append(random.choice(verb_list))
        else:
            verbs.append("<unk>")
    pos = {"noun": nouns, "verb": verbs}
    return pos

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.zeros_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param.data)
            else:
                init.zeros_(param.data)
    elif isinstance(m, nn.Embedding):
        init.uniform_(m.weight.data)


def collater(data):
    # disallow batch of 1
    assert len(data) > 1
    obj = {"feature": [], "id": [], "caption": [], "cap_id": [], "number": []}
    for dat in data:
        obj["feature"].append(dat["feature"])
        obj["id"].append(dat["id"])
        obj["caption"].append(dat["caption"])
        obj["cap_id"].append(dat["cap_id"])
        obj["number"].append(dat["number"])
    obj["feature"] = torch.stack(obj["feature"])
    return obj

def collater_mse(data):
    # disallow batch of 1
    assert len(data) > 1
    obj = {"feature_slow": [], "feature_fast":[], "id": [], "caption": [], "cap_id": [], "number": []}
    for dat in data:
        obj["feature_slow"].append(dat["feature_slow"])
        obj["feature_fast"].append(dat["feature_fast"])
        obj["id"].append(dat["id"])
        obj["caption"].append(dat["caption"])
        obj["cap_id"].append(dat["cap_id"])
        obj["number"].append(dat["number"])
    obj["feature_slow"] = torch.stack(obj["feature_slow"])
    obj["feature_fast"] = torch.stack(obj["feature_fast"])
    return obj

def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)

def visualize(metrics, config_name):
    savedir = os.path.join("out", config_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir, 0o777)
    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Recall for config {}".format(config_name))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Recall")
    for k, v in metrics.items():
        if "recall" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savedir, "recall.png"))

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Median ranking for config {}".format(config_name))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Median ranking")
    for k, v in metrics.items():
        if "median" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savedir, "median.png"))

    fig = plt.figure(clear=True)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.set_title("Mean ranking for config {}".format(config_name))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean ranking")
    for k, v in metrics.items():
        if "mean" in k:
            ax.plot(v, label=k)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(os.path.join(savedir, "median.png"))

