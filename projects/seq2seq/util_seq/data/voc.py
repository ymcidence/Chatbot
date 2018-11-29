import numpy as np
import re
import unicodedata

PAD_token = 0
SOS_token = 1
EOS_token = 2


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words = self.num_words + 1
        else:
            self.word2count[word] = self.word2count[word] + 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)

    def save(self, save_dir):
        to_save = {'name': self.name,
                   'trimmed': self.trimmed,
                   'word2index': self.word2index,
                   'word2count': self.word2count,
                   'index2word': self.index2word,
                   'num_words': self.num_words}

        np.save(save_dir + '/voc.npy', to_save)

    # noinspection PyUnresolvedReferences
    def load(self, load_dir):
        data = np.load(load_dir).item()
        self.name = data['name']
        self.trimmed = data['trimmed']
        self.word2index = data['word2index']
        self.word2count = data['word2count']
        self.index2word = data['index2word']
        self.num_words = data['num_words']


if __name__ == '__main__':
    # voc = Voc('hehe')
    # voc.load('data/voc.npy')
    pp = np.load('data/sentences.npy')
    print('hehe')
