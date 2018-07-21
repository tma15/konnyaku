# -*- coding: utf-8 -*-

class Vocabulary:
    def __init__(self, set_special_tokens=False):
        self.index2word = []
        self.word2index = {}

        if set_special_tokens:
            self.add('<s>')
            self.add('</s>')
            self.add('<unk>')
            self.add('<pad>')

    def __len__(self):
        return len(self.index2word)

    def has(self, word):
        return word in self.word2index

    def add(self, word):
        index = len(self.index2word)

        self.index2word.append(word)
        self.word2index[word] = index

    def save(self, vocabfile):
        with open(vocabfile, 'w') as f:
            for word in self.index2word:
                print(word, filename=f)

    @classmethod
    def load(self, vocabfile):
        with open(vocabfile, 'r') as f:
            for line in f:
                line = line.strip()
                elems = line.split()
                self.add(elems[0])
