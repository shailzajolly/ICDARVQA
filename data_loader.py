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


class VqaDataset(Dataset):

    def __init__(self, root, seqlen=14):
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
            que = np.ones(seqlen, dtype=np.int64) * len(word2idx)
            for i, word in enumerate(qa['question_toked']):
                if i == seqlen:
                    break
                if word in word2idx:
                    que[i] = word2idx[word]

            ans = np.ones((101,2), dtype=np.int64) * len(ans2idx)
            for i, word in enumerate([qa['answer']] + qa['distractors']):
                for j, w in enumerate(word.split()):
                    if j == 2:
                        break
                    if w in ans2idx:
                        ans[i, j] = ans2idx[w]

            self.vqas.append({
                'v': os.path.join('data', 'vfeats', '{}.npy'.format(qa['file_path'])),
                'q': que,
                'a': ans,
                'gt': np.array(0),
                'q_txt': qa['question'],
                'a_txt': qa['answer']
            })

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        return np.load(self.vqas[idx]['v']), self.vqas[idx]['q'], self.vqas[idx]['a'],\
               self.vqas[idx]['gt'], self.vqas[idx]['q_txt'], self.vqas[idx]['a_txt']

    @staticmethod
    def get_n_classes(fpath=os.path.join('data', 'dict_ans.pkl')):
        idx2ans, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2ans)

    @staticmethod
    def get_vocab_size(fpath=os.path.join('data', 'dict_q.pkl')):
        idx2word, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2word)


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
                                               pin_memory=args.pin_mem)


    val_loader = torch.utils.data.DataLoader(dataset_vqa,
                                               sampler=valid_sampler,
                                               batch_size=args.batch_size,
                                               num_workers=args.n_workers,
                                               pin_memory=args.pin_mem)


    vocab_size = VqaDataset.get_vocab_size()
    num_classes = VqaDataset.get_n_classes()
    return train_loader, val_loader, vocab_size, num_classes
