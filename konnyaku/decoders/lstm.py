# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, device=None):
        super(Decoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.device = device

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)
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

#         word = word.view(batch_size)
        word_emb = self.emb(word)
#         print('word', word.shape)
#         print('word_emb', word_emb.shape)
        word_emb = word_emb.view(batch_size, self.emb_size)
        h, c = self.lstm(word_emb, (h, c))
        out = self.out(h)

        state = {
            'h': h,
            'c': c,
            'out': out,
        }
        return state



if __name__ == '__main__':
    sent1 = [1, 2, 3, 4]
    sent2 = [1, 2, 3, 0]
    source = torch.tensor([sent1, sent2], dtype=torch.long)

    ### Reshapes to (src_len, batch_size, 1)
    source = source.view(-1, 2, 1)
    print(source.shape)

    vocab_size = 5
    emb_size = 2
    hidden_size = 3
    encoder = Encoder(vocab_size, emb_size, hidden_size)
    encoder(source)

