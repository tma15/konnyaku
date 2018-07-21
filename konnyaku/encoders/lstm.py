# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """A bidrectional LSTMs"""
    def __init__(self, vocab_size, emb_size, hidden_size, device=None):
        super(Encoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.device = device

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)

        self.lstm_f = nn.LSTMCell(emb_size, hidden_size)
        self.lstm_b = nn.LSTMCell(emb_size, hidden_size)

    def forward(self, source):
        source_len = source.shape[0]
        batch_size = source.shape[1]

        embed = self.emb(source)

        h_f = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c_f = torch.zeros(batch_size, self.hidden_size, device=self.device)
        h_b = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c_b = torch.zeros(batch_size, self.hidden_size, device=self.device)

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
        cs = []
        for i in range(source_len):
            hs = hs_f[i] + hs_b[i]

        state = {
            'hs_f': hs_f,
            'cs_f': cs_f,
            'hs_b': hs_b,
            'cs_b': cs_b,
            'hs': hs,
        }
        return state



if __name__ == '__main__':
    sent1 = [1, 2, 3, 4]
    sent2 = [1, 2, 3, 0]
#     source = torch.tensor(sent, dtype=torch.long)
    source = torch.tensor([sent1, sent2], dtype=torch.long)

    ### Reshapes to (src_len, batch_size, 1)
    source = source.view(-1, 2, 1)

    vocab_size = 5
    emb_size = 2
    hidden_size = 3
    encoder = Encoder(vocab_size, emb_size, hidden_size)
    encoder(source)

