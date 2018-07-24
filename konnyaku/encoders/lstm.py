# -*- coding: utf-8 -*-
import sys

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """A bidrectional LSTMs"""
    def __init__(self,
            vocab,
            emb_size,
            hidden_size,
            initial_emb=None):
        super(Encoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab = vocab

        self.emb = nn.Embedding(len(vocab), emb_size, padding_idx=0)
        if initial_emb is not None:
            self.emb.weight.data.copy_(torch.from_numpy(initial_emb))

        self.lstm_f = nn.LSTMCell(emb_size, hidden_size)
        self.lstm_b = nn.LSTMCell(emb_size, hidden_size)

    def forward(self, source):
        source_len = source.shape[0]
        batch_size = source.shape[1]

        embed = self.emb(source)

        h_f = source.new_zeros(batch_size, self.hidden_size, dtype=torch.float32)
        c_f = source.new_zeros(batch_size, self.hidden_size, dtype=torch.float32)
        h_b = source.new_zeros(batch_size, self.hidden_size, dtype=torch.float32)
        c_b = source.new_zeros(batch_size, self.hidden_size, dtype=torch.float32)

        hs_f = []
        cs_f = []
        for i in range(source_len):
            word_emb = embed[i].view(batch_size, self.emb_size)
            h_f, c_f = self.lstm_f(word_emb, (h_f, c_f))
            hs_f.append(h_f)
            cs_f.append(c_f)

        hs_b = []
        cs_b = []
        for i in reversed(range(source_len)):
            word_emb = embed[i].view(batch_size, self.emb_size)
            h_b, c_b = self.lstm_b(word_emb, (h_b, c_b))
            hs_b.insert(0, h_b)
            cs_b.insert(0, c_b)

        hs = []
        for i in range(source_len):
            hs_bi = hs_f[i] + hs_b[i]
            hs.append(hs_bi)
        hs = torch.stack(hs, dim=1)

        mask = source == self.vocab.word2index['<pad>']
        mask = mask.view(source_len, batch_size).t()

        state = {
            'hs_f': hs_f,
            'cs_f': cs_f,
            'hs_b': hs_b,
            'cs_b': cs_b,
            'hs': hs,
            'mask': mask,
        }
        return state
