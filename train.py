# -*- coding: utf-8 -*-
import argparse
import sys

import torch
import torch.nn as nn
from torch import optim

import konnyaku

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='src_file')
    parser.add_argument('-t', dest='trg_file')
    parser.add_argument('-e', dest='epoch', type=int, default=10)
    parser.add_argument('-g', dest='device_id', type=int, default=-1)
    parser.add_argument('--emb', dest='emb_size', type=int, default=100)
    parser.add_argument('--hid', dest='hidden_size', type=int, default=100)
    parser.add_argument('--model', dest='model_file', default='model')
    parser.add_argument('--num', dest='num', type=int, default=-1)
    args = parser.parse_args()
    return args

def build_vocab(data_file):
    vcb = konnyaku.Vocabulary(set_special_tokens=True)
    with open(data_file, 'r') as f:
        for line in f:
            words = line.strip().split()
            for word in words:
                if not vcb.has(word):
                    vcb.add(word)
    return vcb

def padding(batch, pad_idx):
    max_len = max([len(x) for x in batch])
    padded = []
    for x in batch:
        padded.append(x + [pad_idx] * (max_len-len(x)))
    return padded

def main():
    args = parse_args()

    ### Chooses a device id
    if args.device_id > -1 and torch.cuda.is_available():
        device = torch.device('cuda', args.device_id)
    else:
        device = torch.device('cpu')

    train_src = args.src_file
    train_trg = args.trg_file

    ### Builds vocabularies
    src_vcb = build_vocab(train_src)
    trg_vcb = build_vocab(train_trg)

    ### Loads source and target
    src = konnyaku.dataset.load_data(train_src, src_vcb)
    trg = konnyaku.dataset.load_data(train_trg, trg_vcb)

    num = args.num
    if num > 0:
        src = src[:num]
        trg = trg[:num]
    else:
        num = len(src)

    emb_size = args.emb_size
    hidden_size = args.hidden_size

    ### Initializes a model
    encoder = konnyaku.Encoder(
        len(src_vcb), emb_size, hidden_size, device=device)
    decoder = konnyaku.Decoder(
        len(trg_vcb), emb_size, hidden_size, device=device)
    model = konnyaku.EncoderDecoder(
        encoder,
        decoder,
        trg_vcb,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vcb.word2index['<pad>'])

    epoch = args.epoch
    batch_size = 2
    for e in range(epoch):
        accum_loss = 0
        for i in range(0, num, batch_size):
            if i % 1000 == 0:
                print('{}/{}'.format(i+1, len(src)), flush=True)
            step = i + batch_size if i + batch_size < num else num
            x = src[i:step]
            y = trg[i:step]
            x = padding(x, src_vcb.word2index['<pad>'])
            y = padding(y, trg_vcb.word2index['<pad>'])
            x = torch.tensor(x, dtype=torch.long, device=device).view(-1, batch_size, 1)
            y = torch.tensor(y, dtype=torch.long, device=device).view(-1, batch_size, 1)

            state = model.encoder(x)
            loss = 0
            trg_len = y.shape[0]
            for t in range(trg_len-1):
                state = model.decoder.step(y[t], state)
                log_p = state['out']
                target = y[t+1].view(-1)
                loss_t = criterion(log_p, target)
                loss += loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accum_loss += loss.data

        avg_loss = accum_loss / len(src)
        print(e, 'loss', avg_loss)

    torch.save(model.state_dict(), args.model_file)

    with torch.no_grad():
        for i in range(len(src)):
            x = src[i]
            y = trg[i]
            src_words = [src_vcb.index2word[k] for k in x]
            trg_words = [trg_vcb.index2word[k] for k in y]
            x = torch.tensor([x], dtype=torch.long).view(-1, batch_size, 1)
            y = torch.tensor([y], dtype=torch.long).view(-1, batch_size, 1)

            print(i)
            print('Source', src_words)
            print('Target', trg_words)
            o = model.generate(x)
            print(o, flush=True)


if __name__ == '__main__':
    main()
