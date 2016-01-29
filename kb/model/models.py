
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.seq2seq import *
import tf_util
import rprop

def _clip_by_value(gradients, min_value, max_value):
    # clipping would break IndexedSlices and therefore sparse updates, because they get converted to tensors
    return [tf.clip_by_value(g, min_value, max_value) if isinstance(g, ops.IndexedSlices) else g for g in gradients]

class AbstractKBScoringModel:

    def __init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, max_grad=5, l2_lambda=0.0,
                 is_batch_training=False, which_sets={"train", "train_text"}):

        self.tuple_ids = dict()
        for (rel, subj, obj), _, typ in kb.get_all_facts():
            if typ in which_sets:
                tup = (subj, obj)
                if tup not in self.tuple_ids:
                    self.tuple_ids[tup] = len(self.tuple_ids)

        self._kb = kb
        self._size = size
        self._batch_size = batch_size
        self._is_batch_training = is_batch_training
        self._is_train = is_train

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
        self.global_step = tf.Variable(0, trainable=False, name="step")
        with tf.device("/cpu:0"):
            if is_batch_training:
                self.opt = rprop.RPropOptimizer()  # tf.train.GradientDescentOptimizer(self.learning_rate)
            else:
                self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0)
        init = tf.random_uniform_initializer(-0.1, 0.1)

        self.rel_input = tf.placeholder(tf.int64, shape=[None], name="rel")
        self.subj_input = tf.placeholder(tf.int64, shape=[None], name="subj")
        self.obj_input = tf.placeholder(tf.int64, shape=[None], name="obj")
        self.tuple_input = tf.placeholder(tf.int64, shape=[None], name="tuple")

        self._feed_dict = {self.rel_input: np.zeros([batch_size], dtype=np.int64),
                           self.subj_input: np.zeros([batch_size], dtype=np.int64),
                           self.obj_input: np.zeros([batch_size], dtype=np.int64),
                           self.tuple_input: np.zeros([batch_size], dtype=np.int64)}

        with vs.variable_scope("score", initializer=init):
            self._scores = self._scoring_f(self.rel_input, self.subj_input, self.obj_input, self.tuple_input)

        if is_train or is_batch_training:
            assert batch_size % (num_neg+1) == 0, "Batch size must be multiple of num_neg+1 during training"
            #with vs.variable_scope("score", initializer=init):
            #    tf.get_variable_scope().reuse_variables()
            #    for i in xrange(num_neg):
            #        self.triple_inputs.append((tf.placeholder(tf.int64, shape=[None], name="rel_%d" % (i+1)),
            #                                   tf.placeholder(tf.int64, shape=[None], name="subj_%d" % (i+1)),
            #                                   tf.placeholder(tf.int64, shape=[None], name="obj_%d" % (i+1))))
            #        self.scores.append(
            #            self._scoring_f(self.triple_inputs[i+1][0], self.triple_inputs[i+1][1], self.triple_inputs[i+1][2]))

            scores = tf.reshape(self._scores, [-1, num_neg + 1])
            num_pos = int(batch_size/(num_neg+1))
            labels = np.zeros([num_pos, num_neg+1], dtype=np.float32)
            labels[:, 0] = 1
            labels = tf.constant(labels, name="labels_constant", dtype=tf.float32)
            loss = math_ops.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(scores, labels)) \
                / math_ops.cast(num_pos, dtypes.float32)

            train_params = tf.trainable_variables()

            self.training_weight = tf.Variable(float(learning_rate), trainable=False, name="training_weight")
            self._feed_dict[self.training_weight] = np.array([1.0], dtype=np.float32)
            with tf.device("/cpu:0"):
                self._grads = tf.gradients(loss, train_params, self.training_weight)
                #clipped_gradients = _clip_by_value(self.grads, -max_grad, max_grad)
                if is_batch_training:
                    with vs.variable_scope("batch_gradient", initializer=init):
                        self._acc_gradients = map(lambda param: tf.get_variable(param.name.split(":")[0],
                                                                                param.get_shape(), param.dtype,
                                                                                tf.constant_initializer(0.0), False),
                                                  train_params)
                    self._loss = tf.get_variable("acc_loss", (), tf.float32, tf.constant_initializer(0.0), False)
                    acc_opt = tf.train.GradientDescentOptimizer(1.0)
                    self._accumulate_gradients = acc_opt.apply_gradients(zip(self._grads, self._acc_gradients))
                    self._acc_loss = acc_opt.apply_gradients([(loss, self._loss)])

                    self._update = self.opt.apply_gradients(
                        zip(map(lambda v: v.value(), self._acc_gradients), train_params), global_step=self.global_step)
                    self._reset = map(lambda param: param.initializer, self._acc_gradients)
                    self._reset.append(self._loss.initializer)
                else:
                    self._loss = loss
                    self._update = self.opt.apply_gradients(zip(self._grads, train_params), global_step=self.global_step)

            if l2_lambda > 0.0:
                l2 = tf.reduce_sum(array_ops.pack([tf.nn.l2_loss(t) for t in train_params]))
                l2_loss = l2_lambda * l2
                if is_batch_training:
                    l2_grads = tf.gradients(l2_loss, train_params, self.training_weight)
                    self._l2_accumulate_gradients = acc_opt.apply_gradients(zip(l2_grads, self._acc_gradients))
                    self._l2_acc_loss = acc_opt.apply_gradients([(l2_loss, self._loss)])
                else:
                    self._l2_update = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(l2_loss, var_list=train_params)

        self.saver = tf.train.Saver(tf.all_variables())


    def _scoring_f(self, rel, subj, obj, tup):
        """
        :param rel indices of relation(s)
        :param subj indices of subjects(s)
        :param obj indices of object(s)
        :return: a batch_size x 1 tensor of scores
        """
        return tf.constant(np.ones([self._batch_size], dtype=np.float32), name="dummy_score", dtype=tf.float32)

    def score_triples(self, sess, triples):
        i = 0
        subj_in = self._feed_dict[self.subj_input]
        obj_in = self._feed_dict[self.obj_input]
        rel_in = self._feed_dict[self.rel_input]
        tuple_in = self._feed_dict[self.tuple_input]
        feed_dict = self._feed_dict
        subj_in *= 0
        obj_in *= 0
        rel_in *= 0

        result = np.zeros([len(triples)])
        while i < len(triples):
            if self._batch_size > len(triples)-i:
                subj_in = np.zeros(len(triples)-i, dtype=np.int64)
                obj_in = np.zeros(len(triples)-i, dtype=np.int64)
                rel_in = np.zeros(len(triples)-i, dtype=np.int64)
                feed_dict = {self.subj_input: subj_in, self.obj_input: obj_in, self.rel_input: rel_in}

            for j in xrange(min(self._batch_size, len(triples)-i)):
                (rel, subj, obj) = triples[i+j]
                rel_in[j] = self._kb.get_id(rel, 0)
                subj_in[j] = self._kb.get_id(subj, 1)
                obj_in[j] = self._kb.get_id(obj, 2)
                tuple_in[j] = self.tuple_ids.get((subj, obj), 0)

            if i+self._batch_size < len(triples):
                result[i:i+self._batch_size] = sess.run(self._scores, feed_dict=feed_dict)
            else:
                result[i:len(triples)] = sess.run(self._scores, feed_dict=feed_dict)

            i += self._batch_size

        return result

    def step(self, sess, pos_triples, neg_triples, mode="update"):
        '''
        :param sess: tf session
        :param pos_triples: list of positive triple
        :param neg_triples: list of (lists of) negative triples
        :param mode: default(train)|loss|accumulate(used for batch training)
        :return:
        '''
        assert self._update, "model has to be created in training mode!"

        assert len(pos_triples) + reduce(lambda acc, x: acc+len(x), neg_triples, 0) == self._batch_size, \
            "batch_size and provided batch do not fit"

        subj_in = self._feed_dict[self.subj_input]
        obj_in = self._feed_dict[self.obj_input]
        rel_in = self._feed_dict[self.rel_input]
        subj_in *= 0
        obj_in *= 0
        rel_in *= 0

        def add_triple(t, j):
            (rel, subj, obj) = t
            rel_in[j] = self._kb.get_id(rel, 0)
            subj_in[j] = self._kb.get_id(subj, 1)
            obj_in[j] = self._kb.get_id(obj, 2)

        j = 0
        for pos, negs in zip(pos_triples, neg_triples):
            add_triple(pos, j)
            j += 1
            for neg in negs:
                add_triple(neg, j)
                j += 1
        if mode == "loss":
            return sess.run(self._loss, feed_dict=self._feed_dict)
        elif mode == "accumulate":
            assert self._is_batch_training, "accumulate only possible during batch training."
            sess.run([self._accumulate_gradients, self._acc_loss], feed_dict=self._feed_dict)
            return 0.0
        else:
            assert self._is_train or self._is_batch_training, "training only possible in training state."
            return sess.run([self._loss, self._update], feed_dict=self._feed_dict)[0]

    def acc_l2_gradients(self, sess):
        assert self._is_batch_training, "acc_l2_gradients only possible during batch training."
        if self._l2_accumulate_gradients:
            return sess.run([self._l2_accumulate_gradients, self._l2_acc_loss])

    def reset_gradients_and_loss(self, sess):
        assert self._is_batch_training, "reset_gradients_and_loss only possible during batch training."
        return sess.run(self._reset)

    def update(self, sess):
        assert self._is_batch_training, "update only possible during batch training."
        return sess.run([self._loss, self._update])[0]


class DistMult(AbstractKBScoringModel):

    def _scoring_f(self, rel, subj, obj, tup):
        with tf.device("/cpu:0"):
            E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._size])
            E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._size])
            E_rels = tf.get_variable("E_r", [len(self._kb.get_symbols(0)), self._size])

        self.e_subj = tf.tanh(tf.nn.embedding_lookup(E_subjs, subj))
        self.e_obj = tf.tanh(tf.nn.embedding_lookup(E_objs, obj))
        self.e_rel = tf.tanh(tf.nn.embedding_lookup(E_rels, rel))
        s_o_prod = self.e_obj * self.e_subj

        score = tf_util.dot(self.e_rel, s_o_prod)

        return score

    def __create_embeddings(self, prefix, num, max_partitions):
        partition_size = max(num / max_partitions, 1)
        if max_partitions % num != 0:
            partition_size += 1
        embeddings = np.random.uniform(-0.1, 0.1, size=(max_partitions*partition_size, self._size)).astype(np.float32)
        E = [tf.Variable(embeddings[i*partition_size:(i+1)*partition_size], name="%s_%d" % (prefix, i))
             for i in xrange(max_partitions)]

        return E


class ModelE(AbstractKBScoringModel):

    def _scoring_f(self, rel, subj, obj, tup):
        E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._size])
        E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._size])
        E_rels_s = tf.get_variable("E_r_s", [len(self._kb.get_symbols(0)), self._size])
        E_rels_o = tf.get_variable("E_r_o", [len(self._kb.get_symbols(0)), self._size])
        self.e_subj = tf.nn.embedding_lookup(E_subjs, subj)
        self.e_obj = tf.nn.embedding_lookup(E_objs, obj)
        self.e_rel_s = tf.nn.embedding_lookup(E_rels_s, rel)
        self.e_rel_o = tf.nn.embedding_lookup(E_rels_o, rel)

        score = tf_util.dot(self.e_rel_s, self.e_subj) + tf_util.dot(self.e_rel_o, self.e_obj)

        return score


class ObservedModel(AbstractKBScoringModel):

    def _scoring_f(self, rel, subj, obj, tup):
        with tf.device("/cpu:0"):
           E_rels = tf.get_variable("E_r", [len(self._kb.get_symbols(0)), self._size])

        rels = [[] for _ in self.tuple_ids.iterkeys()]
        for (rel, subj, obj), _, typ in self._kb.get_all_facts():
            tup_id = self.tuple_ids.get(subj, obj)
            rels[tup_id].append(self._kb.get_id(rel, 2))


