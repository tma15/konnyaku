# -*- coding: utf-8 -*-
import argparse
import configparser
import sys
import time

import torch
import torch.nn as nn
from torch import optim

import konnyaku

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='src_file')
    parser.add_argument('-t', dest='trg_file')
    parser.add_argument('-c', dest='conf_file')
    parser.add_argument('-e', dest='epoch', type=int, default=10)
    parser.add_argument('-b', dest='batch_size', type=int, default=10)
    parser.add_argument('-g', dest='device_id', type=int, default=-1)
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

def load_glove_vectors(glove_file, vcb, emb_size):
    ### Initializes word vectors with uniform distribution
    vectors = numpy.random.uniform(-0.1, 0.1, (len(vcb), emb_size))

    with open(glove, 'r') as f:
        for line in f:
            elems = line.strip().split()
            word = elems[0]
            if vcb.has(word):
                wid = vcb.word2index[word]
                vec = [float(v) for v in elems[1:]]
                vectors[wid] = numpy.array(vec)
    return vectors

def main():
    args = parse_args()
    conf = configparser.ConfigParser()
    conf.read(args.conf_file)

    emb_size = conf['model'].getint('emb_size', 100)
    hidden_size = conf['model'].getint('hidden_size', 100)

    ### Chooses a device
    if args.device_id > -1 and torch.cuda.is_available():
        device = torch.device('cuda', args.device_id)
    else:
        device = torch.device('cpu')

    train_src = args.src_file
    train_trg = args.trg_file

    ### Builds vocabularies
    src_vcb = build_vocab(train_src)
    trg_vcb = build_vocab(train_trg)

    src_vcb.save('src_vcb')
    trg_vcb.save('trg_vcb')

    ### Loads pretarined word embeddings
    if conf['model'].get('src_pretrained', False):
        initial_src_emb = load_glove_vectors(
            conf['model'].get('src_pretrained'), src_vcb, emb_size)
    else:
        initial_src_emb = None

    if conf['model'].get('trg_pretrained', False):
        initial_trg_emb = load_glove_vectors(
            conf['model'].get('trg_pretrained'), trg_vcb, emb_size)
    else:
        initial_trg_emb = None

    ### Loads source and target
    src = konnyaku.dataset.load_data(train_src, src_vcb)
    trg = konnyaku.dataset.load_data(train_trg, trg_vcb)

    num = args.num
    if num > 0:
        src = src[:num]
        trg = trg[:num]
    else:
        num = len(src)

    ### Initializes a model
    model = konnyaku.EncoderDecoder(
        src_vcb,
        trg_vcb,
        emb_size,
        hidden_size,
    ).to(device)

    ### Sets an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    ### Defines a training criterion
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vcb.word2index['<pad>'])

    epoch = args.epoch
    print('epoch\tloss\telapsed_time')
    batch_size = args.batch_size
    for e in range(epoch):
        accum_loss = 0
        start = time.time()
        for i in range(0, num, batch_size):
            step = i + batch_size if i + batch_size < num else num
            x = src[i:step]
            y = trg[i:step]
            x = padding(x, src_vcb.word2index['<pad>'])
            y = padding(y, trg_vcb.word2index['<pad>'])

            x = torch.tensor(x, dtype=torch.long).t()
            y = torch.tensor(y, dtype=torch.long).t()
            x = x.view(-1, x.shape[1], 1).to(device)
            y = y.view(-1, y.shape[1], 1).to(device)

            state = model.encoder(x)
            loss = 0
            trg_len = y.shape[0]
            for t in range(trg_len-1):
                state = model.decoder.step(y[t], state)
                out = state['out']
                target = y[t+1].view(-1)
                loss_t = criterion(out, target)
                loss += loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Since back-propagation of accum_loss is not needed,
            ### only a value obtained by forward caulcation is accumlated.
            accum_loss += loss.data

        avg_loss = accum_loss / num
        elapased_time = time.time() - start
        print('{0}\t{1:.2f}\t{2:.2f}'.format(e, avg_loss, elapased_time), flush=True)

    torch.save(model.state_dict(), args.model_file)

    with torch.no_grad():
        cpu = torch.device('cpu')
        model.to(cpu)
        for i in range(num):
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
            print(o, flush=True)


if __name__ == '__main__':
    main()
