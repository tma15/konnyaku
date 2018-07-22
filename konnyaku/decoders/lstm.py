# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
            vocab_size,
            emb_size,
            hidden_size,
            initial_emb=None):
        super(Decoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        if initial_emb is not None:
            self.emb.weight.data.copy_(torch.from_numpy(initial_emb))

        self.lstm = nn.LSTMCell(emb_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def step(self, word, state):
        batch_size = word.shape[0]

        embed = self.emb(word)

        if 'h' and 'c' in state:
            h = state['h']
            c = state['c']
        else:
            h = state['hs_b'][0]
            c = state['cs_b'][0]

        word_emb = self.emb(word)
        word_emb = word_emb.view(batch_size, self.emb_size)
        h, c = self.lstm(word_emb, (h, c))
        out = self.out(h)

        state['h'] = h
        state['c'] = c
        state['out'] = out
        return state
