# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self,
            encoder,
            decoder,
            trg_vcb,
            device=None):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.trg_vcb = trg_vcb
        self.device = device

    def generate(self, source):
        state = self.encoder(source)

        max_len = 10
        word = torch.tensor(
            [[self.trg_vcb.word2index['<s>']]], dtype=torch.long, device=self.device)
#         print('word', word.shape, word)
        output = []
        for t in range(max_len):
            state = self.decoder.step(word, state)
            topv, topi = state['out'].data.topk(1)
#             if t == 0:
#                 print(topv)
#             print(topi)
            if topi.item() == self.trg_vcb.word2index['</s>']:
                break
            else:
                output.append(self.trg_vcb.index2word[topi.item()])

#             print('topi', topi.shape, topi)

            word = topi

#             word = topi.squeeze().detach()
#             print('word', word.shape, word)

        return output
