import sys, os
import argparse
import time
import json
import pickle
import random

import torch
import torchtext
import spacy

from utils import sec2str

nlp = spacy.load('en_core_web_sm')


class Vocabulary():
    def __init__(self, min_freq=3, max_len=30):
        self.min_freq = min_freq
        self.max_len = max_len

    """
    build vocabulary from textfile.
    """
    def load_vocab(self, textfile):
        before = time.time()
        print("building vocabulary...", flush=True)
        self.text_proc = torchtext.data.Field(sequential=True, init_token="<bos>", eos_token="<eos>", lower=True, fix_length=self.max_len, tokenize="spacy", batch_first=True)
        with open(textfile, 'r', encoding="utf-8", errors="ignore") as f:
            sentences = f.readlines()
        sent_proc = list(map(self.text_proc.preprocess, sentences))
        self.text_proc.build_vocab(sent_proc, min_freq=self.min_freq)
        self.len = len(self.text_proc.vocab)
        self.padidx = self.text_proc.vocab.stoi["<pad>"]
        print("done building vocabulary, minimum frequency is {} times".format(self.min_freq), flush=True)
        print("{} | # of words in vocab: {}".format(sec2str(time.time() - before), self.len), flush=True)

    def load_entities(self, nounfile, verbfile):
        before = time.time()
        print("getting entities...", flush=True)
        self.nouns = set()
        self.verbs = set()
        with open(nounfile, 'r', encoding="utf-8", errors="ignore") as f:
            self.nouns = set(f.read().splitlines())
        with open(verbfile, 'r', encoding="utf-8", errors="ignore") as f:
            self.verbs = set(f.read().splitlines())
        # remove these tokens
        notok = ["<unk>", "be", "is", "are"]
        for stop in notok:
            if stop in self.nouns:
                self.nouns.remove(stop)
            if stop in self.verbs:
                self.verbs.remove(stop)
        print("done getting entities", flush=True)
        print("{} | # of nouns: {}, # of verbs: {}".format(sec2str(time.time() - before), len(self.nouns), len(self.verbs)), flush=True)

    # sentence_batch: list of str
    # return indexes of sentence batch as torch.LongTensor
    def return_idx(self, sentence_batch):
        try:
            assert isinstance(sentence_batch, list)
        except AssertionError:
            print("passing {} to return_idx, should be list of str".format(type(sentence_batch)))
        preprocessed = list(map(self.text_proc.preprocess, sentence_batch))
        out = self.text_proc.process(preprocessed)
        return out

    # return sentence batch from indexes from torch.LongTensor
    def return_sentences(self, ten):
        if isinstance(ten, torch.Tensor):
            ten = ten.tolist()
        out = []
        for idxs in ten:
            tokenlist = [self.text_proc.vocab.itos[idx] for idx in idxs]
            out.append(" ".join(tokenlist))
        return out

    # get a negative noun that is not in the sentences
    def replace_noun(self, sentences, noun):
        if noun == "<unk>":
            return "<unk>"
        prohibit = set()
        for sentence in sentences:
            proc = nlp(sentence)
            prohibit.update([ent.lemma_ for ent in proc if ent.pos_ == 'NOUN'])
        rep = random.choice(list(self.nouns - prohibit))
        return rep

    # get a negative verb that is not in the sentences
    def replace_verb(self, sentences, verb):
        if verb == "<unk>":
            return "<unk>"
        prohibit = set()
        for sentence in sentences:
            proc = nlp(sentence)
            prohibit.update([ent.lemma_ for ent in proc if ent.pos_ == 'VERB'])
        rep = random.choice(list(self.verbs - prohibit))
        return rep

    def __len__(self):
        return self.len

# build caption txt file from msr-vtt annotation json file
def msrvtt_cap2txt(jsonfile, dst):
    sentences = []
    with open(jsonfile, 'r') as f:
        alldata = json.load(f)
    for ann in alldata["sentences"]:
        sentences.append(ann["caption"].strip())
    with open(dst, 'w+') as f:
        f.write("\n".join(sentences).replace("\xe9", "e").replace("\u2019", "\'").replace("\u0432", "B"))

# build caption txt file from MSVD annotation txt files
def msvd_cap2txt(txtfiles, dst):
    sentences = []
    for file in txtfiles:
        with open(file, 'r') as f:
            for line in f:
                _, cap = line.rstrip().split("\t", 1)
                sentences.append(cap)
    with open(dst, 'w+') as f:
        f.write("\n".join(sentences))

# build noun and verb set file from caption txt file
def make_entlist(txtfile, noundst, verbdst):
    sentences = []
    nouns = set()
    verbs = set()
    with open(txtfile, 'r') as f:
        for i, cap in enumerate(f):
            proc = nlp(cap.rstrip())
            nouns.update([ent.lemma_ for ent in proc if ent.pos_ == 'NOUN'])
            verbs.update([ent.lemma_ for ent in proc if ent.pos_ == 'VERB'])
            if i % 1000 == 999:
                print(i+1, flush=True)
    with open(noundst, 'w+') as f:
        f.write("\n".join(list(nouns)))
    with open(verbdst, 'w+') as f:
        f.write("\n".join(list(verbs)))

def parse_args():
    parser = argparse.ArgumentParser()

    # configurations of dataset (paths)
    parser.add_argument('--dataset', type=str, default='msrvtt')
    parser.add_argument('--root_path', type=str, default='/groups1/gaa50131/datasets/MSR-VTT')
    args = parser.parse_args()

    return args

# for debugging
if __name__ == '__main__':

    args = parse_args()

    if args.dataset == "msrvtt":
        file = os.path.join(args.root_path, "videodatainfo_2017.json")
    elif args.dataset == "msvd":
        file = [os.path.join(args.root_path, i) for i in ["sents_train_lc_nopunc.txt", "sents_val_lc_nopunc.txt"]]
    dest = "captions_{}.txt".format(args.dataset)
    noundest = "nouns_{}.txt".format(args.dataset)
    verbdest = "verbs_{}.txt".format(args.dataset)
    if args.dataset == "msrvtt":
        if not os.path.exists(dest):
            msrvtt_cap2txt(file, dest)
    elif args.dataset == "msvd":
        if not os.path.exists(dest):
            msvd_cap2txt(file, dest)
    if not (os.path.exists(noundest) and os.path.exists(verbdest)):
        make_entlist(dest, noundest, verbdest)

    vocab = Vocabulary()
    vocab.load_vocab(dest)
    sentence = ["The cat and the hat sat on a mat."]
    ten = vocab.return_idx(sentence)
    print(ten)
    sent = vocab.return_sentences(ten)
    print(sent)


