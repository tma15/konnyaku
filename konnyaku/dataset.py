# -*- coding: utf-8 -*-

def load_data(data_file, vcb):
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            words = line.strip().split()
            sequence = [vcb.word2index['<s>']]
            for word in words:
                if vcb.has(word):
                    wid = vcb.word2index[word]
                else:
                    wid = vcb.word2index['<unk>']
                sequence.append(wid)
            sequence.append(vcb.word2index['</s>'])
            data.append(sequence)
    return data

