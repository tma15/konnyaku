# -*- coding: utf-8 -*-
import argparse
import configparser

import torch
import torch.nn as nn

import konnyaku

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model_file')
    parser.add_argument('-c', dest='conf_file')
    parser.add_argument('-s', dest='src_file')
    parser.add_argument('-t', dest='trg_file')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    conf = configparser.ConfigParser()
    conf.read(args.conf_file)

    device = torch.device('cpu')

    emb_size = conf['model'].getint('emb_size')
    hidden_size = conf['model'].getint('hidden_size')

    src_vcb = konnyaku.Vocabulary()
    src_vcb.load('src_vcb')
    trg_vcb = konnyaku.Vocabulary()
    trg_vcb.load('trg_vcb')

    ### Initializes a model
    model = konnyaku.EncoderDecoder(
        src_vcb,
        trg_vcb,
        emb_size,
        hidden_size,
    ).to(device)

    model.load_state_dict(torch.load(args.model_file))

    src = konnyaku.dataset.load_data(args.src_file, src_vcb)
    trg = konnyaku.dataset.load_data(args.trg_file, trg_vcb)

    criterion = nn.CrossEntropyLoss(ignore_index=trg_vcb.word2index['<pad>'])

    with torch.no_grad():
        accum_loss = 0
        for i in range(len(src)):
            x = src[i]
            y = trg[i]
            src_words = [src_vcb.index2word[k] for k in x]
            trg_words = [trg_vcb.index2word[k] for k in y]
            x = torch.tensor([x], dtype=torch.long).view(-1, 1, 1)
            y = torch.tensor([y], dtype=torch.long).view(-1, 1, 1)

            print(i)
            print('Source', src_words)
            print('Target', trg_words)
            o = model.generate(x)
            print('Output', o, flush=True)
            print('==')

            state = model.encoder(x)
            loss = 0
            trg_len = y.shape[0]
            for t in range(trg_len-1):
                state = model.decoder.step(y[t], state)
                out = state['out']
                target = y[t+1].view(-1)
                loss_t = criterion(out, target)
                loss += loss_t
            accum_loss += loss.data
        print('loss:{:.2f}', accum_loss / len(src))

if __name__ == '__main__':
    main()
