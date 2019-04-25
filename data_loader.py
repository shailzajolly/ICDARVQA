from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import pickle

import torch
from torch.utils.data import Dataset, SubsetRandomSampler
import numpy as np
from tqdm import tqdm

from constants import *


class VqaDataset(Dataset):

    def __init__(self, root, ques_len=14, ans_len=4):
        """
        root (str): path to data directory
        seqlen (int): maximum words in a question
        """

        print("Loading preprocessed files...")
        qas = pickle.load(open(os.path.join(root, 'data_qa.pkl'), 'rb'))
        idx2word, word2idx = pickle.load(open(os.path.join(root, 'dict_q.pkl'), 'rb'))
        idx2ans, ans2idx = pickle.load(open(os.path.join(root, 'dict_ans.pkl'), 'rb'))

        print("Setting up everything...")
        self.vqas = []
        for qa in tqdm(qas):

            que = []
            for i, word in enumerate(qa['question_toked']):
                if i == ques_len:
                    break
                que.append(word2idx.get(word, UNK_TOKEN)) # append UNK index if word not in vocab

            ans = []
            for i, word in enumerate(qa['answer'].split()):
                if i == ans_len-1:
                    break
                ans.append(ans2idx.get(word, UNK_TOKEN))
            ans.append(EOS_TOKEN)

            self.vqas.append({
                'v': os.path.join('data', 'vfeats', '{}.npy'.format(qa['file_path'])),
                'q': que,
                'a': ans,
                'q_txt': qa['question'],
                'a_txt': qa['answers']
            })

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        v = torch.from_numpy(np.load(self.vqas[idx]['v']))
        q = torch.LongTensor(self.vqas[idx]['q'])
        a = torch.LongTensor(self.vqas[idx]['a'])
        q_txt = self.vqas[idx]['q_txt']
        a_txt = self.vqas[idx]['a_txt']

        return v, q, a, q_txt, a_txt

    @staticmethod
    def get_n_classes(fpath=os.path.join('data', 'dict_ans.pkl')):
        idx2ans, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2ans)

    @staticmethod
    def get_vocab_size(fpath=os.path.join('data', 'dict_q.pkl')):
        idx2word, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2word)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (v, q, a, mca, q_txt, a_txt).

    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

    Args:
        data: list of tuple (v, q, a, q_txt, a_txt).
            - v: torch tensor of shape (36,2048);
            - q: torch tensor of shape (?); variable length.
            - a: torch tensor of shape (?); variable length.
            - q_txt: str
            - a_txt: str

    Returns:

    """

    def merge(batch):
        return torch.stack(tuple(b for b in batch), 0)

    def merge_seq(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, torch.Tensor(lengths)



    data.sort(key=lambda x: len(x[1]), reverse=True)

    # seperate data fields
    v, q, a, q_txt, a_txt = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    v = merge(v)
    q, q_lengths = merge_seq(q)
    a, a_lengths = merge_seq(a)

    return v, q, a, q_lengths, a_lengths, q_txt, a_txt


def prepare_data(args):
    dataset_vqa = VqaDataset(root=args.data_root)
    random_seed = 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset_vqa)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset_vqa,
                                               sampler=train_sampler,
                                               batch_size=args.batch_size,
                                               num_workers=args.n_workers,
                                               pin_memory=args.pin_mem,
                                               collate_fn=collate_fn)


    val_loader = torch.utils.data.DataLoader(dataset_vqa,
                                             sampler=valid_sampler,
                                             batch_size=args.batch_size,
                                             num_workers=args.n_workers,
                                             pin_memory=args.pin_mem,
                                             collate_fn=collate_fn)


    vocab_size = VqaDataset.get_vocab_size()
    num_classes = VqaDataset.get_n_classes()
    return train_loader, val_loader, vocab_size, num_classes
