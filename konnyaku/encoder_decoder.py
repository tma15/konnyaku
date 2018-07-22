# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import konnyaku

class EncoderDecoder(nn.Module):
    def __init__(self,
            src_vcb,
            trg_vcb,
            emb_size,
            hidden_size,
            initial_src_emb=None,
            initial_trg_emb=None):
        super(EncoderDecoder, self).__init__()

        self.encoder = konnyaku.Encoder(
            len(src_vcb), emb_size, hidden_size,
            initial_emb=initial_src_emb,
        )
        self.decoder = konnyaku.AttentionalDecoder(
            len(trg_vcb), emb_size, hidden_size,
            initial_emb=initial_trg_emb,
        )

        self.src_vcb = src_vcb
        self.trg_vcb = trg_vcb

    def generate(self, source, max_len=20):
        state = self.encoder(source)

        word = source.new_tensor(
            [[self.trg_vcb.word2index['<s>']]], dtype=torch.long)
        output = []
        for t in range(max_len):
            state = self.decoder.step(word, state)
            topv, topi = state['out'].data.topk(1)
            if topi.item() == self.trg_vcb.word2index['</s>']:
                break
            else:
                output.append(self.trg_vcb.index2word[topi.item()])

            word = topi
        return output

    def nbest_generate(self, source, n=5):
        raise NotImplementedError
