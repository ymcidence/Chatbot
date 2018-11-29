from util_seq.data import voc
import numpy as np


class Dataset(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 50)
        self.max_length = kwargs.get('max_length', 20)
        self.embedding_size = kwargs.get('embedding_size')
        self.voc = voc.Voc('hehe')
        self.voc.load(kwargs.get('voc_dir'))
        self.pairs = np.load(kwargs.get('sentence_dir'))
        self.set_size = self.pairs.__len__()
        self.batch_end = 0
        self.seq = np.random.permutation(self.set_size)

    def _shuffle(self):
        self.seq = np.random.permutation(self.set_size)

    def next_batch(self):
        if self.batch_size + self.batch_end >= self.set_size:
            self._shuffle()
            self.batch_end = 0

        sentence_in = []
        sentence_out = []
        words_in = []
        words_out = []
        for i in range(self.batch_end, self.batch_end + self.batch_size):
            pos = self.seq[i]
            this_q = self.pairs[pos][0]
            this_q_ind = self.zero_padding(self.index_from_sentence(this_q))

            this_r = self.pairs[pos][1]
            this_r_ind = self.zero_padding(self.index_from_sentence(this_r))

            sentence_in.append(this_q_ind)
            sentence_out.append(this_r_ind)
            words_in.append(this_q)
            words_out.append(this_r)

        self.batch_end += self.batch_size
        return {'sentence_in': np.asarray(sentence_in, dtype=np.int32),
                'sentence_out': np.asarray(sentence_out, dtype=np.int32),
                'words_in': words_in,
                'words_out': words_out}

    def index_from_sentence(self, sentence):
        return np.asarray([self.voc.word2index[word] for word in sentence.split(' ')] + [voc.EOS_token])

    def zero_padding(self, sentence: np.ndarray, fill_value=voc.PAD_token):
        if sentence.shape[0] >= self.max_length:
            sentence = sentence[:self.max_length - 1]
        return np.pad(sentence, (0, self.max_length - sentence.shape[0]), 'constant', constant_values=fill_value)


if __name__ == '__main__':
    config = {'batch_size': 50,
              'max_length': 20,
              'embedding_size': 512,
              'voc_dir': 'data/voc.npy',
              'sentence_dir': 'data/sentences.npy'}
    data = Dataset(**config)
    a = data.next_batch()
    print('hehe')
