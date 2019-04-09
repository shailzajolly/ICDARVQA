from __future__ import division, print_function, absolute_import

import os
import pdb
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model import Model
from utils import GOATLogger, save_ckpt, compute_score
from data_loader import prepare_data
from arguments import get_args


def evaluate(val_loader, model, epoch, device, logger):
    model.eval()
    loss = nn.CrossEntropyLoss()

    batches = len(val_loader)
    for step, (v, q, a, gt, _, _) in enumerate(tqdm(val_loader, ascii=True)):
        v = v.to(device)
        q = q.to(device)
        a = a.to(device)
        gt = gt.to(device)

        logits = model(v, q, a)
        output = loss(logits, gt)

        score = compute_score(logits, gt)

        logger.batch_info_eval(epoch, step, batches, output.item(), score)

    score = logger.batch_info_eval(epoch, -1, batches)
    return score


def train(train_loader, model, optim, epoch, device, logger, moving_loss):
    model.train()
    loss = nn.CrossEntropyLoss()
    smooth_const = 0.1

    batches = len(train_loader)
    start = time.time()
    for step, (v, q, a, gt, _, _) in enumerate(train_loader):
        data_time = time.time() - start

        v = v.to(device)
        q = q.to(device)
        a = a.to(device)
        gt = gt.to(device)

        logits = model(v, q, a)
        output = loss(logits, gt)

        optim.zero_grad()
        output.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()

        moving_loss = (output.item() if epoch == 0 and step ==0 else 
                        (1 - smooth_const) * moving_loss + smooth_const * output.item())

        batch_time = time.time() - start
        score = compute_score(logits, gt)
        logger.batch_info(epoch, step, batches, data_time, moving_loss, score, batch_time)
        start = time.time()

    return moving_loss


def main():

    parser = get_args()
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    logger = GOATLogger(args.mode, args.save, args.log_freq)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cpu:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")

        args.devices = torch.cuda.device_count()
        args.batch_size *= args.devices
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)

    # Get data
    train_loader, val_loader, vocab_size, num_answers = prepare_data(args)

    # Set up model
    model = Model(vocab_size, args.word_embed_dim, args.hidden_size, args.resnet_out, num_answers)
    model = nn.DataParallel(model).to(device)
    logger.loginfo("Parameters: {:.3f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # Set up optimizer
    optim = torch.optim.Adam(model.parameters(), 2e-4)

    last_epoch = 0
    bscore = 0.0
    moving_loss = 0.0

    if args.resume:
        logger.loginfo("Initialized from ckpt: " + args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        last_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])

    if args.mode == 'eval':
        _ = evaluate(val_loader, model, last_epoch, device, logger)
        return

    # Train
    for epoch in range(last_epoch, args.epoch):
        moving_loss = train(train_loader, model, optim, epoch, device, logger, moving_loss)
        score = evaluate(val_loader, model, epoch, device, logger)
        bscore = save_ckpt(score, bscore, epoch, model, optim, args.save, logger)

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
