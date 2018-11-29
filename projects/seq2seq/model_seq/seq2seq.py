import tensorflow as tf
from time import gmtime, strftime
import os
from util_seq.data.dataset import Dataset
from util_seq.data import voc
import numpy as np

PAD_token = voc.PAD_token
SOS_token = voc.SOS_token
EOS_token = voc.EOS_token


def word_embedding(name, vocabulary_size, embedding_size) -> tf.Tensor:
    """

    :param name:
    :param vocabulary_size:
    :param embedding_size:
    :return:
    """

    with tf.variable_scope(name):
        word_vector = tf.get_variable('word_vector', [vocabulary_size, embedding_size])
    return word_vector


def rnn_encoder(name, input_tensor, input_length, embedding_size):
    """



    :param name:
    :param input_tensor:
    :param input_length:
    :param embedding_size:
    :return:
    """

    with tf.variable_scope(name):
        rnn_cell1 = tf.nn.rnn_cell.LSTMCell(embedding_size, name='cell1', dtype=tf.float32)
        rnn_cell2 = tf.nn.rnn_cell.LSTMCell(embedding_size, name='cell2', dtype=tf.float32)
        # noinspection PyUnresolvedReferences
        out, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell1, cell_bw=rnn_cell2, inputs=input_tensor,
                                                     sequence_length=input_length, dtype=tf.float32)

    return out, state


def rnn_decoder(name, input_hidden, input_length, vocabulary_size, embedding_size,
                data_helper: tf.contrib.seq2seq.Helper, max_length=20, reuse=False):
    """


    :param name:
    :param input_hidden: A tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor.
    :param input_length:
    :param vocabulary_size:
    :param embedding_size:
    :param data_helper:
    :param max_length:
    :param reuse:
    :return:
    """
    batch_size = tf.shape(input_hidden[0])[0]
    input_hidden = (input_hidden[0] + input_hidden[1]) / 2
    # input_state = (input_state[0] + input_state[1]) / 2
    # vocabulary_size, embedding_size = word_vector.get_shape().as_list()

    with tf.variable_scope(name, reuse=reuse):
        # starts = tf.constant(starts, dtype=tf.float32)
        # word_emb = tf.nn.embedding_lookup(word_vector, starts)
        rnn_cell = tf.nn.rnn_cell.LSTMCell(embedding_size, name='cell', dtype=tf.float32)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(embedding_size, input_hidden,
                                                                memory_sequence_length=input_length)
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(rnn_cell, attention_mechanism,
                                                        attention_layer_size=embedding_size / 2)

        out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocabulary_size, reuse=reuse)

        decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=data_helper,
                                                  initial_state=out_cell.zero_state(dtype=tf.float32,
                                                                                    batch_size=batch_size))
        out = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=False,
                                                maximum_iterations=max_length)

    return out[0]


# noinspection PyShadowingNames
class Sequence2Sequence(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size')
        self.max_length = kwargs.get('max_length')
        self.embedding_size = kwargs.get('embedding_size')
        self.voc = voc.Voc('hehe')
        self.voc.load(kwargs.get('voc_dir'))
        self.training = kwargs.get('training', True)
        self.sentence_in = tf.placeholder(tf.int32, shape=[None, None], name='sentence_in')
        self.sentence_out = tf.placeholder(tf.int32, shape=[None, None], name='sentence_out')
        self.net = self._net()
        self.g_step = tf.Variable(0, trainable=False, name='global_step')
        self.loss = self._loss()

    def _net(self):
        input_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self.sentence_in, PAD_token)), 1)
        output_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self.sentence_out, PAD_token)), 1)

        word_vector = word_embedding('word_embedding', self.voc.num_words, self.embedding_size)
        word_emb = tf.nn.embedding_lookup(word_vector, self.sentence_in)

        # batch_size = self.sentence_in.get_shape().as_list()[0]

        start_tensor = tf.cast(tf.ones([tf.shape(self.sentence_in)[0]], dtype=tf.int32) * SOS_token, dtype=tf.int32)

        data_helper = tf.contrib.seq2seq.TrainingHelper(word_emb, output_length) if self.training else \
            tf.contrib.seq2seq.GreedyEmbeddingHelper(
                word_vector, start_tokens=tf.to_int32(start_tensor), end_token=EOS_token)

        encoder_out, encoder_state = rnn_encoder('encoder', word_emb, input_length, self.embedding_size)

        decoder_out = rnn_decoder('decoder', encoder_out, input_length, self.voc.num_words, self.embedding_size,
                                  data_helper)

        return decoder_out

    def _loss(self):

        to_pad = self.max_length - tf.shape(self.net.rnn_output)[1]
        padded_logits = tf.pad(self.net.rnn_output, [[0, 0], [0, to_pad], [0, 0]], mode='CONSTANT',
                               constant_values=PAD_token)

        to_pad = self.max_length - tf.shape(self.sentence_out)[1]
        padded_targets = tf.pad(self.sentence_out, [[0, 0], [0, to_pad]], mode='CONSTANT',
                                constant_values=PAD_token)

        loss_mask = tf.to_float(tf.not_equal(padded_targets, PAD_token))
        loss_seq = tf.contrib.seq2seq.sequence_loss(logits=padded_logits, targets=padded_targets, weights=loss_mask)
        loss_seq = tf.reduce_mean(loss_seq)

        single_regu = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.name.find('word_vector') < 0]
        loss_regu = tf.add_n(single_regu) * 1e-5

        loss = loss_seq  # + loss_regu

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('loss_seq', loss_seq)
        tf.summary.scalar('loss_regu', loss_regu)

        self.a = padded_targets
        self.b = padded_logits
        self.c = loss_mask
        return loss

    def _op(self):
        optimizer = tf.train.AdamOptimizer(1e-3)
        train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # gradients = optimizer.compute_gradients(self.loss, var_list=train_list)
        # clipped_gradients = [(tf.clip_by_value(grad, -30., 30.), var) for grad, var in gradients]
        # op = optimizer.apply_gradients(clipped_gradients, global_step=self.g_step)

        op = optimizer.minimize(self.loss, global_step=self.g_step, var_list=train_list)

        return op

    @staticmethod
    def restore(restore_dir, sess: tf.Session, var_list=None):
        saver = tf.train.Saver(var_list=var_list)

        saver.restore(sess, restore_dir)

    @staticmethod
    def save(save_dir, sess, step=None, var_list=None):
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_dir + 'ymmodel', global_step=step)

    def train(self, data: Dataset, sess: tf.Session, max_iter, log_dir, restore_dir):
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        op = self._op()

        initial_op = tf.global_variables_initializer()
        sess.run(initial_op)

        if restore_dir is not None:
            self.restore(restore_dir, sess)

        summary_path = os.path.join(log_dir, 'tb', time_string) + os.sep
        save_path = os.path.join(log_dir, 'model', time_string) + os.sep

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        writer = tf.summary.FileWriter(summary_path)
        summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
        writer.add_graph(sess.graph)

        for i in range(max_iter):
            batch_data = data.next_batch()
            feed_dict = {self.sentence_in: batch_data.get('sentence_in'),
                         self.sentence_out: batch_data.get('sentence_out')}

            net_value, loss_value, summary_value, _ = sess.run([self.net.sample_id, self.loss, summary_op, op],
                                                               feed_dict=feed_dict)

            step = tf.train.global_step(sess, self.g_step)
            writer.add_summary(summary_value, step)

            if (i + 1) % 20 == 0:
                print('iter: {}, loss: {}'.format(i, loss_value))

            if (i + 1) % 100 == 0:
                hook_sentence = self.vector_to_sentence(net_value[0, :])
                print('q: {}, a: {}'.format(batch_data.get('words_in')[0], hook_sentence))

            if (i + 1) % 1000 == 0:
                self.save(save_dir=save_path, sess=sess, step=step)

    def forward(self, sentence, sess: tf.Session):
        feed_dict = {self.sentence_in: sentence}
        generated = sess.run(self.net.sample_id, feed_dict=feed_dict)
        return generated

    def vector_to_sentence(self, input_vector: np.ndarray):
        output_words = [self.voc.index2word[token.item()] for token in input_vector]

        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return output_words

    def test(self, sess: tf.Session, restore_dir):
        initial_op = tf.global_variables_initializer()
        sess.run(initial_op)

        self.restore(restore_dir, sess)

        # noinspection PyRedundantParentheses
        while not self.training:
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit':
                    break
                input_sentence = voc.unicode_to_ascii(input_sentence)
                input_sentence = np.asarray(
                    [self.voc.word2index[word] for word in input_sentence.split(' ')] + [voc.EOS_token])

                if input_sentence.shape[0] >= self.max_length:
                    input_sentence = input_sentence[:self.max_length - 1]

                input_sentence = np.pad(input_sentence, (0, self.max_length - input_sentence.shape[0]), 'constant',
                                        constant_values=PAD_token)

                output_words = self.forward(np.asarray([input_sentence]), sess)[0, :]
                output_words = [self.voc.index2word[token.item()] for token in output_words]

                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))
            except KeyError:
                print('I beg your pardon?')


def do_training():
    config = {'batch_size': 60,
              'max_length': 20,
              'embedding_size': 512,
              'voc_dir': 'data/voc.npy',
              'sentence_dir': 'data/sentences.npy',
              'training': True}

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=sess_config)

    data = Dataset(**config)
    model = Sequence2Sequence(**config)
    restore_d = None  # '/home/ymcidence/Workspace/ChatBotYH/data/log/model/Thu29Nov2018-130328/ymmodel-20000'
    model.train(data, sess, 100000, log_dir='data/log', restore_dir=restore_d)


def do_test():
    config = {'batch_size': 1,
              'max_length': 20,
              'embedding_size': 512,
              'voc_dir': 'data/voc.npy',
              'sentence_dir': 'data/sentences.npy',
              'training': False}

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=sess_config)

    model = Sequence2Sequence(**config)
    restore_dir = '/home/ymcidence/Workspace/ChatBotYH/data/log/model/Thu29Nov2018-130328/ymmodel-20000'
    model.test(sess, restore_dir=restore_dir)


if __name__ == '__main__':
    do_training()
    # do_test()
