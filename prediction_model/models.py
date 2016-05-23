
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.seq2seq import *
import tf_util
import model
from tensorflow.models.rnn.rnn_cell import *


class AbstractKBPredictionModel:

    def __init__(self, kb, size, batch_size, is_train=True, learning_rate=1e-2):
        self._kb = kb
        self._size = size
        self._batch_size = batch_size
        self._is_train = is_train
        self._init = model.default_init()

        with vs.variable_scope(self.name(), initializer=self._init):
            self._init_inputs()
            self.query = self._comp_f()

            self.candidates = tf.get_variable("E_candidate", [len(self.arg_vocab), self._size])

            lookup_individual = tf.nn.embedding_lookup(self.candidates, self._y_input)
            self._score = tf_util.batch_dot(lookup_individual, self.query)

            lookup = tf.nn.embedding_lookup(self.candidates, self._y_candidates)
            self._scores_with_negs = tf.squeeze(tf.batch_matmul(lookup, tf.expand_dims(self.query, [2])), [2])

            if is_train:
                self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.9)
                self.global_step = tf.Variable(0, trainable=False, name="step")

                self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0)

                current_batch_size = tf.gather(tf.shape(self._scores_with_negs), [0])
                labels = tf.constant([0], tf.int64)
                labels = tf.tile(labels, current_batch_size)
                loss = math_ops.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self._scores_with_negs, labels))

                train_params = tf.trainable_variables()
                self.training_weight = tf.Variable(1.0, trainable=False, name="training_weight")

                self._loss = loss / math_ops.cast(current_batch_size, dtypes.float32)
                in_params = self._input_params()
                if in_params is None:
                    self._grads = tf.gradients(self._loss, train_params, self.training_weight)
                else:
                    self._grads = tf.gradients(self._loss, train_params + in_params, self.training_weight)
                    self._input_grads = self._grads[len(train_params):]
                if len(train_params) > 0:
                    self._update = self.opt.apply_gradients(zip(self._grads[:len(train_params)], train_params),
                                                        global_step=self.global_step)
                else:
                    self._update = tf.assign_add(self.global_step, 1)

        self.saver = tf.train.Saver([v for v in tf.all_variables() if self.name() in v.name])

    def _input_params(self):
        return None

    def name(self):
        return self.__class__.__name__

    def _comp_f(self):
        return tf.cond(self._is_inv, self._comp_f_bw, self._comp_f_fw)

    def _comp_f_bw(self):
        """
        :return: a batch_size tensor of representations
        """
        pass

    def _comp_f_fw(self):
        """
        :return: a batch_size tensor of representations
        """
        pass

    def _init_inputs(self):
        self._rel_input = tf.placeholder(tf.int64, shape=[None], name="rel")
        self._x_input = tf.placeholder(tf.int64, shape=[None], name="x")
        self._y_candidates = tf.placeholder(tf.int64, shape=[None, None], name="candidates")
        self._y_input = tf.placeholder(tf.int64, shape=[None], name="y")
        self._is_inv = tf.placeholder(tf.bool, shape=(), name="invert")
        self._x_in = np.zeros([self._batch_size], dtype=np.int64)
        self._y_cands = np.zeros([self._batch_size, 2], dtype=np.int64)
        self._y_in = np.zeros([self._batch_size], dtype=np.int64)
        self._rel_in = np.zeros([self._batch_size], dtype=np.int64)

        self.arg_vocab = {}

        for arg in self._kb.get_symbols(1):
            self.arg_vocab[arg] = len(self.arg_vocab)
        for arg in self._kb.get_symbols(2):
            if arg not in self.arg_vocab:
                self.arg_vocab[arg] = len(self.arg_vocab)

        self._feed_dict = {}

    def _change_batch_size(self, batch_size):
        new_x_in = np.zeros([batch_size], dtype=np.int64)
        new_x_in[:self._batch_size] = self._x_in
        self._x_in = new_x_in

        new_y_cands = np.zeros([batch_size, self._y_cands.shape[1]], dtype=np.int64)
        new_y_cands[:self._batch_size] = self._y_cands
        self._y_cands = new_y_cands

        new_y_in = np.zeros([batch_size], dtype=np.int64)
        new_y_in[:self._batch_size] = self._y_in
        self._y_in = new_y_in

        new_rel_in = np.zeros([batch_size, self._rel_in.shape[1]], dtype=np.int64)
        new_rel_in[:self._batch_size] = self._rel_in
        self._rel_in = new_rel_in

        self._batch_size = batch_size

    def _start_adding_triples(self):
        pass

    def _add_triple_and_negs_to_input(self, triple, neg_candidates, batch_idx, is_inv):
        if batch_idx >= self._batch_size:
            self._change_batch_size(max(self._batch_size*2, batch_idx))
        (rel, x, y) = triple
        self._rel_in[batch_idx] = self._kb.get_id(rel, 0)
        self._x_in[batch_idx] = self.arg_vocab[y] if is_inv else self.arg_vocab[x]
        if len(neg_candidates)+1 != self._y_cands.shape[1]:
            self._y_cands = np.zeros([self._batch_size, len(neg_candidates)+1], dtype=np.int64)
        self._y_cands[batch_idx] = [self.arg_vocab[x] if is_inv else self.arg_vocab[y]] + \
                                     [self.arg_vocab[neg] for neg in neg_candidates]

    def _add_triple_to_input(self, triple, batch_idx, is_inv):
        if batch_idx >= self._batch_size:
            self._change_batch_size(max(self._batch_size*2, batch_idx))
        (rel, x, y) = triple
        self._rel_in[batch_idx] = self._kb.get_id(rel, 0)
        self._x_in[batch_idx] = self.arg_vocab[y] if is_inv else self.arg_vocab[x]
        self._y_in[batch_idx] = self.arg_vocab[x] if is_inv else self.arg_vocab[y]

    def _finish_adding_triples(self, batch_size, is_inv):
        if batch_size < self._batch_size:
            self._feed_dict[self._x_input] = self._x_in[:batch_size]
            self._feed_dict[self._y_candidates] = self._y_cands[:batch_size]
            self._feed_dict[self._rel_input] = self._rel_in[:batch_size]
            self._feed_dict[self._y_input] = self._y_in[:batch_size]
        else:
            self._feed_dict[self._x_input] = self._x_in
            self._feed_dict[self._y_candidates] = self._y_cands
            self._feed_dict[self._rel_input] = self._rel_in
            self._feed_dict[self._y_input] = self._y_in
        self._feed_dict[self._is_inv] = is_inv

    def _get_feed_dict(self):
        return self._feed_dict

    def score_triples(self, sess, triples, is_inv):
        i = 0
        result = np.zeros([len(triples)])
        while i < len(triples):
            batch_size = min(self._batch_size, len(triples)-i)
            self._start_adding_triples()
            for batch_idx in range(batch_size):
                self._add_triple_to_input(triples[i + batch_idx], batch_idx, is_inv)
            self._finish_adding_triples(batch_size, is_inv)

            result[i:i+batch_size] = sess.run(self._score, feed_dict=self._get_feed_dict())
            i += batch_size

        return result

    def score_triples_with_negs(self, sess, triples, neg_examples, is_inv):
        i = 0
        result = np.zeros([len(triples), len(neg_examples[0])+1])
        while i < len(triples):
            batch_size = min(self._batch_size, len(triples)-i)
            self._start_adding_triples()
            batch_idx = 0
            for pos, negs in zip(triples[i:i+batch_size], neg_examples[i:i+batch_size]):
                self._add_triple_and_negs_to_input(pos, negs, batch_idx, is_inv)
                batch_idx += 1
            self._finish_adding_triples(batch_size, is_inv)
            result[i:i+batch_size] = sess.run(self._scores_with_negs, feed_dict=self._get_feed_dict())
            i += batch_size

        return result

    def step(self, sess, pos_triples, neg_examples, is_inv, mode="update"):
        '''
        :param sess: tf session
        :param pos_triples: list of positive triple
        :param neg_examples: list of (lists of) negative triples
        :param is_inv: relations are inverted
        :param mode: default(train)|loss|accumulate(used for batch training)
        :return:
        '''
        assert self._is_train, "model has to be created in training mode!"

        assert len(pos_triples) == self._batch_size, \
            "batch_size and provided batch do not fit"

        batch_idx = 0
        self._start_adding_triples()
        for pos, negs in zip(pos_triples, neg_examples):
            self._add_triple_and_negs_to_input(pos, negs, batch_idx, is_inv)
            batch_idx += 1

        self._finish_adding_triples(batch_idx, is_inv)

        if mode == "loss":
            return sess.run(self._loss, feed_dict=self._get_feed_dict())
        else:
            return sess.run([self._loss, self._update], feed_dict=self._get_feed_dict())[0]


class DistMult(AbstractKBPredictionModel):

    def _comp_f(self):
        E_candidate = tf.get_variable("E_candidate", [len(self.arg_vocab), self._size])
        E_rel = tf.get_variable("E_rel", [len(self._kb.get_symbols(0)), self._size])

        e_arg = tf.tanh(tf.nn.embedding_lookup(E_candidate, self._x_input))
        e_rel = tf.nn.embedding_lookup(E_rel, self._rel_input)
        tf.get_variable_scope().reuse_variables()  #  reuse E_candidate
        return e_arg * e_rel


class ModelE(AbstractKBPredictionModel):

    def _comp_f_fw(self):
        E_rel = tf.get_variable("E_rel_fw", [len(self._kb.get_symbols(0)), self._size])
        e_rel = tf.nn.embedding_lookup(E_rel, self._rel_input)
        return e_rel

    def _comp_f_bw(self):
        E_rel = tf.get_variable("E_rel_bw", [len(self._kb.get_symbols(0)), self._size])
        e_rel = tf.nn.embedding_lookup(E_rel, self._rel_input)
        return e_rel

