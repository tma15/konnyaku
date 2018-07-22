# -*- coding: utf-8 -*-
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from konnyaku.decoders.lstm import Decoder

class AttentionalDecoder(Decoder):
    def __init__(self,
            vocab_size,
            emb_size,
            hidden_size,
            initial_emb=None):
        super(AttentionalDecoder, self).__init__(
            vocab_size,
            emb_size,
            hidden_size,
            initial_emb=initial_emb,
        )

        self.out = nn.Linear(2 * hidden_size, vocab_size)

    def context(self, h, encoder_hiddens, state):
        """Calculates a context vector base on dot attender"""
        batch_size = h.shape[0]
        src_len = encoder_hiddens.shape[1]

        ### Reshapes for matmul
        h = h.view(batch_size, self.hidden_size, 1)

        ### (batch_size, src_len, hidden_size) x (batch_size, hidden_size, 1)
        ### = (batch_size, src_len, 1)
        score = torch.matmul(encoder_hiddens, h)
        score = score.view(batch_size, src_len)

        p_att = F.softmax(score, dim=1)

        ### Reshapes for matmul
        p_att = p_att.view(batch_size, 1, src_len)

        context = torch.matmul(p_att, encoder_hiddens)
        context = context.view(batch_size, self.hidden_size)

        state['p_att'] = p_att
        state['context'] = context
        return state

    def step(self, word, state):
        batch_size = word.shape[0]

        embed = self.emb(word)

        if 'h' and 'c' in state:
            h = state['h']
            c = state['c']
        else:
            ### Initial states
            h = state['hs_b'][0]
            c = state['cs_b'][0]

        word_emb = self.emb(word)
        word_emb = word_emb.view(batch_size, self.emb_size)
        h, c = self.lstm(word_emb, (h, c))

        state = self.context(h, state['hs'], state)

        h_tilde = torch.cat((h, state['context']), dim=1)

        out = self.out(h_tilde)

        state['h'] = h
        state['c'] = c
        state['out'] = out
        return state
