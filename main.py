from __future__ import division, print_function, absolute_import

import time
import random
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from model import VqaEncoder, AnswerDecoder
from utils import GOATLogger
from data_loader import prepare_data
from arguments import get_args
from constants import *
from metrics import VqaMetric


def evaluate(val_loader, model, epoch, device, logger, vqa_metric):
    for module in model:
        module.eval()

    cr_loss = nn.CrossEntropyLoss(ignore_index=0)

    batches = len(val_loader)
    for step, (v, q, a, mca, q_lens, a_lens, _, a_txt) in enumerate(tqdm(val_loader, ascii=True)):

        v = v.to(device)
        q = q.to(device)
        a = a.to(device)
        mca = mca.to(device)
        q_lens = q_lens.to(device)

        batch_size = len(a)
        loss = 0
        print_loss = 0
        n_totals = 0

        joint_embed, mca_embed = model[0](v, q, mca, q_lens)

        decoder_out_idxs = []

        decoder_input = torch.LongTensor([SOS_TOKEN for _ in range(batch_size)])
        decoder_input = decoder_input.to(device)

        decoder_hidden = joint_embed.unsqueeze(0)

        for t in range(a.size(1)):
            decoder_output, decoder_hidden = model[1](decoder_input,
                                                      decoder_hidden,
                                                      mca_embed)

            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)])
            decoder_input = decoder_input.to(device)

            decoder_out_idxs.append(vqa_metric.get_real_idxs(decoder_input, mca))

            step_loss = cr_loss(decoder_output, a[:, t].view(-1))
            loss += step_loss
            nTotal = torch.sum(a[:, t] != PAD_TOKEN).float()
            print_loss += step_loss.item() * nTotal
            n_totals += nTotal

        score = vqa_metric.compute_score(decoder_out_idxs, a_txt)

        logger.batch_info_eval(epoch, step, batches, (print_loss/n_totals).item(), score)

    score = logger.batch_info_eval(epoch, -1, batches)
    return score


def train(train_loader,
          model,
          optims,
          epoch,
          device,
          logger,
          vqa_metric,
          moving_loss):

    for module in model:
        module.train()

    cr_loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    smooth_const = 0.1

    batches = len(train_loader)
    start = time.time()
    for step, (v, q, a, mca, q_lens, a_lens, _, a_txt) in enumerate(train_loader):
        data_time = time.time() - start

        v = v.to(device)
        q = q.to(device)
        a = a.to(device)
        mca = mca.to(device)
        q_lens = q_lens.to(device)

        batch_size = len(a)
        loss = 0
        print_loss = 0
        n_totals = 0

        joint_embed, mca_embed = model[0](v, q, mca, q_lens)

        decoder_out_idxs = []

        decoder_input = torch.LongTensor([SOS_TOKEN for _ in range(batch_size)]) # (batch_size, )
        decoder_input = decoder_input.to(device)

        decoder_hidden = joint_embed.unsqueeze(1) # (batch_size, 1, hidden_size). Due to multi-gpu training.

        for t in range(a.size(1)):
            decoder_output, decoder_hidden = model[1](decoder_input,
                                                      decoder_hidden,
                                                      mca_embed)

            
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)])
            decoder_input = decoder_input.to(device)

            decoder_out_idxs.append(vqa_metric.get_real_idxs(decoder_input, mca))

            step_loss = cr_loss(decoder_output, a[:,t].view(-1))
            loss += step_loss
            nTotal = torch.sum(a[:,t]!=PAD_TOKEN).float()
            print_loss += step_loss.item() * nTotal
            n_totals += nTotal
        
        for optim in optims:
            optim.zero_grad()

        loss.backward()

        for module in model:
            nn.utils.clip_grad_norm_(module.parameters(), 0.25)

        for optim in optims:
            optim.step()

        moving_loss = ((print_loss/n_totals).item() if epoch == 0 and step == 0 else
                        (1 - smooth_const) * moving_loss + smooth_const * (print_loss/n_totals).item())

        batch_time = time.time() - start
        score = vqa_metric.compute_score(decoder_out_idxs, a_txt)
        logger.batch_info(epoch, step, batches, data_time, moving_loss, score, batch_time)
        start = time.time()

    return moving_loss


def main():

    parser = get_args()
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    logger = GOATLogger(args.mode, args.save, args.log_freq)
    vqa_metric = VqaMetric(os.path.join('data', 'dict_ans.pkl'))

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

    vqa_enc = VqaEncoder(vocab_size, args.word_embed_dim, args.hidden_size, args.resnet_out, num_answers)
    ans_dec = AnswerDecoder(args.hidden_size)

    model = [vqa_enc, ans_dec]

    for idx, module in enumerate(model):
        model[idx] = nn.DataParallel(module).to(device)

    logger.loginfo("Parameters: {:.3f}M".format(sum(sum(p.numel() for p in module.parameters())
                                                    for module in model) / 1e6))

    # Set up optimizer
    optims = [torch.optim.Adam(module.parameters(), args.lr) for module in model]


    last_epoch = 0
    bscore = 0.0
    moving_loss = 0.0

    if args.resume:
        logger.loginfo("Initialized from ckpt: " + args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        last_epoch = ckpt['epoch']
        for idx, module in enumerate(model):
            module.load_state_dict(ckpt['state_dict'])
            optims[idx].load_state_dict(ckpt['optim_state_dict'])

    if args.mode == 'eval':
        _ = evaluate(val_loader, model, last_epoch, device, logger, vqa_metric)
        return

    # Train
    for epoch in range(last_epoch, args.epoch):
        moving_loss = train(train_loader, model, optims, epoch, device, logger, vqa_metric, moving_loss)
        score = evaluate(val_loader, model, epoch, device, logger, vqa_metric)
        #bscore = save_ckpt(score, bscore, epoch, model, optims, args.save, logger)

    logger.loginfo("Done")


if __name__ == '__main__':
    main()
