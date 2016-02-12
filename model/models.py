
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.seq2seq import *
import tf_util
import rprop
import model


class AbstractKBScoringModel:

    def __init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False):
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
        self._init = model.default_init()

        self._init_inputs()

        with vs.variable_scope("score", initializer=self._init):
            self._scores = self._scoring_f()

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

            num_pos = int(batch_size/(num_neg+1))
            scores = tf.reshape(self._scores, [num_pos, num_neg + 1])
            labels = np.zeros([num_pos, num_neg+1], dtype=np.float32)
            labels[:, 0] = 1
            labels = tf.constant(labels, name="labels_constant", dtype=tf.float32)
            loss = math_ops.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(scores, labels))

            train_params = filter(lambda v: self.name() in v.name, tf.trainable_variables())

            self.training_weight = tf.Variable(float(learning_rate), trainable=False, name="training_weight")
            self._feed_dict[self.training_weight] = np.array([1.0], dtype=np.float32)
            with tf.device("/cpu:0"):
                #clipped_gradients = _clip_by_value(self.grads, -max_grad, max_grad)
                if is_batch_training:
                    self._grads = tf.gradients(loss, train_params, self.training_weight)
                    with vs.variable_scope("batch_gradient", initializer=self._init):
                        self._acc_gradients = map(lambda param: tf.get_variable(param.name.split(":")[0],
                                                                                param.get_shape(), param.dtype,
                                                                                tf.constant_initializer(0.0), False),
                                                  train_params)
                    self._loss = tf.get_variable("acc_loss", (), tf.float32, tf.constant_initializer(0.0), False)
                    # We abuse the gradient descent optimizer for accumulating gradients and loss (summing)
                    acc_opt = tf.train.GradientDescentOptimizer(-1.0)
                    self._accumulate_gradients = acc_opt.apply_gradients(zip(self._grads, self._acc_gradients))
                    self._acc_loss = acc_opt.apply_gradients([(loss, self._loss)])

                    self._update = self.opt.apply_gradients(
                        zip(map(lambda v: v.value(), self._acc_gradients), train_params), global_step=self.global_step)
                    self._reset = map(lambda param: param.initializer, self._acc_gradients)
                    self._reset.append(self._loss.initializer)
                else:
                    self._loss = loss / math_ops.cast(num_pos, dtypes.float32)
                    in_params = self._input_params()
                    if not in_params:
                        self._grads = tf.gradients(self._loss, train_params, self.training_weight)
                    else:
                        self._grads = tf.gradients(self._loss, train_params + in_params, self.training_weight)
                        self._input_grads = self._grads[len(train_params):]
                    if len(train_params) > 0:
                        self._update = self.opt.apply_gradients(zip(self._grads[:len(train_params)], train_params),
                                                                global_step=self.global_step)

            if l2_lambda > 0.0:
                l2 = tf.reduce_sum(array_ops.pack([tf.nn.l2_loss(t) for t in train_params]))
                l2_loss = l2_lambda * l2
                if is_batch_training:
                    l2_grads = tf.gradients(l2_loss, train_params)
                    self._l2_accumulate_gradients = acc_opt.apply_gradients(zip(l2_grads, self._acc_gradients))
                    self._l2_acc_loss = acc_opt.apply_gradients([(l2_loss, self._loss)])
                else:
                    self._l2_update = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(l2_loss, var_list=train_params)

        self.saver = tf.train.Saver(tf.all_variables())

    def _input_params(self):
        return None

    def name(self):
        return self.__class__.__name__

    def _scoring_f(self):
        """
        :return: a batch_size tensor of scores
        """
        return tf.constant(np.ones([self._batch_size], dtype=np.float32), name="dummy_score", dtype=tf.float32)

    def _init_inputs(self):
        self._rel_input = tf.placeholder(tf.int64, shape=[None], name="rel")
        self._subj_input = tf.placeholder(tf.int64, shape=[None], name="subj")
        self._obj_input = tf.placeholder(tf.int64, shape=[None], name="obj")
        self._subj_in = np.zeros([self._batch_size], dtype=np.int64)
        self._obj_in = np.zeros([self._batch_size], dtype=np.int64)
        self._rel_in = np.zeros([self._batch_size], dtype=np.int64)
        self._feed_dict = {}

    def _start_adding_triples(self):
        pass

    def _add_triple_to_input(self, t, j):
        (rel, subj, obj) = t

        self._rel_in[j] = self._kb.get_id(rel, 0)
        self._subj_in[j] = self._kb.get_id(subj, 1)
        self._obj_in[j] = self._kb.get_id(obj, 2)

    def _finish_adding_triples(self, batch_size):
        if batch_size < self._batch_size:
            self._feed_dict[self._subj_input] = self._subj_in[:batch_size]
            self._feed_dict[self._obj_input] = self._obj_in[:batch_size]
            self._feed_dict[self._rel_input] = self._rel_in[:batch_size]
        else:
            self._feed_dict[self._subj_input] = self._subj_in
            self._feed_dict[self._obj_input] = self._obj_in
            self._feed_dict[self._rel_input] = self._rel_in

    def _get_feed_dict(self):
        return self._feed_dict

    def score_triples(self, sess, triples):
        i = 0
        result = np.zeros([len(triples)])
        while i < len(triples):
            batch_size = min(self._batch_size, len(triples)-i)
            self._start_adding_triples()
            for j in xrange(batch_size):
                self._add_triple_to_input(triples[i+j], j)
            self._finish_adding_triples(batch_size)

            result[i:i+batch_size] = sess.run(self._scores, feed_dict=self._get_feed_dict())
            i += batch_size

        return result

    def step(self, sess, pos_triples, neg_triples, mode="update"):
        '''
        :param sess: tf session
        :param pos_triples: list of positive triple
        :param neg_triples: list of (lists of) negative triples
        :param mode: default(train)|loss|accumulate(used for batch training)
        :return:
        '''
        assert self._is_train or self._is_batch_training, "model has to be created in training mode!"

        assert len(pos_triples) + reduce(lambda acc, x: acc+len(x), neg_triples, 0) == self._batch_size, \
            "batch_size and provided batch do not fit"

        j = 0
        self._start_adding_triples()
        for pos, negs in zip(pos_triples, neg_triples):
            self._add_triple_to_input(pos, j)
            j += 1
            for neg in negs:
                self._add_triple_to_input(neg, j)
                j += 1

        self._finish_adding_triples(j)

        if mode == "loss":
            return sess.run(self._loss, feed_dict=self._get_feed_dict())
        elif mode == "accumulate":
            assert self._is_batch_training, "accumulate only possible during batch training."
            sess.run([self._accumulate_gradients, self._acc_loss], feed_dict=self._get_feed_dict())
            return 0.0
        else:
            return sess.run([self._loss, self._update], feed_dict=self._get_feed_dict())[0]

    def acc_l2_gradients(self, sess):
        assert self._is_batch_training, "acc_l2_gradients only possible during batch training."
        if hasattr(self, "_l2_accumulate_gradients"):
            return sess.run([self._l2_accumulate_gradients, self._l2_acc_loss])

    def reset_gradients_and_loss(self, sess):
        assert self._is_batch_training, "reset_gradients_and_loss only possible during batch training."
        return sess.run(self._reset)

    def update(self, sess):
        assert self._is_batch_training, "update only possible during batch training."
        return sess.run([self._loss, self._update])[0]


class DistMult(AbstractKBScoringModel):

    def _scoring_f(self):
        with tf.device("/cpu:0"):
            E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._size])
            E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._size])
            E_rels = tf.get_variable("E_r", [len(self._kb.get_symbols(0)), self._size])

        self.e_subj = tf.tanh(tf.nn.embedding_lookup(E_subjs, self._subj_input))
        self.e_obj = tf.tanh(tf.nn.embedding_lookup(E_objs, self._obj_input))
        self.e_rel = tf.sigmoid(tf.nn.embedding_lookup(E_rels, self._rel_input))
        s_o_prod = self.e_obj * self.e_subj

        score = tf_util.dot(self.e_rel, s_o_prod)

        return score


class ModelE(AbstractKBScoringModel):

    def _scoring_f(self):
        with tf.device("/cpu:0"):
            E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._size])
            E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._size])
            E_rels_s = tf.get_variable("E_r_s", [len(self._kb.get_symbols(0)), self._size])
            E_rels_o = tf.get_variable("E_r_o", [len(self._kb.get_symbols(0)), self._size])

        self.e_subj = tf.tanh(tf.nn.embedding_lookup(E_subjs, self._subj_input))
        self.e_obj = tf.tanh(tf.nn.embedding_lookup(E_objs, self._obj_input))
        self.e_rel_s = tf.tanh(tf.nn.embedding_lookup(E_rels_s, self._rel_input))
        self.e_rel_o = tf.tanh(tf.nn.embedding_lookup(E_rels_o, self._rel_input))

        score = tf_util.dot(self.e_rel_s, self.e_subj) + tf_util.dot(self.e_rel_o, self.e_obj)

        return score


class ModelO(AbstractKBScoringModel):

    def __init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False, which_sets=["train_text"]):
        self._which_sets = set(which_sets)
        AbstractKBScoringModel.__init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2,
                                        l2_lambda=0.0, is_batch_training=False)

    def _init_inputs(self):
        self._rel_ids = dict()
        len(self._kb.get_symbols(0))
        # create tuple to rel lookup
        self._tuple_rels_lookup = dict()
        for (rel, subj, obj), _, typ in self._kb.get_all_facts():
            if typ in self._which_sets:
                if rel not in self._rel_ids:
                    self._rel_ids[rel] = len(self._rel_ids)
                s_i = self._kb.get_id(subj, 1)
                o_i = self._kb.get_id(obj, 2)
                r_i = self._rel_ids[rel]
                t = (s_i, o_i)
                if t not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t] = [r_i]
                else:
                    self._tuple_rels_lookup[t].append(r_i)

        self.__num_relations = len(self._rel_ids)

        #also add inverse relations to tuples
        for (rel, subj, obj), _, typ in self._kb.get_all_facts():
            if typ in self._which_sets:
                if rel not in self._rel_ids:
                    self._rel_ids[rel] = len(self._rel_ids)
                s_i = self._kb.get_id(subj, 1)
                o_i = self._kb.get_id(obj, 2)
                r_i = self._rel_ids[rel] + self.__num_relations
                t = (o_i, s_i)
                if t not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t] = [r_i]
                else:
                    self._tuple_rels_lookup[t].append(r_i)

        self._rel_input = tf.placeholder(tf.int64, shape=[None], name="rel")
        self._rel_in = np.zeros([self._batch_size], dtype=np.int64)
        self._sparse_indices_input = tf.placeholder(tf.int64, name="sparse_indices")
        self._sparse_values_input = tf.placeholder(tf.int64, name="sparse_values")
        self._shape_input = tf.placeholder(tf.int64, name="shape")
        self._feed_dict = {}

    def _start_adding_triples(self):
        self._sparse_indices = []
        self._sparse_values = []
        self._max_cols = 1

    def _add_triple_to_input(self, t, j):
        (rel, subj, obj) = t
        r_i = self._kb.get_id(rel, 0)
        self._rel_in[j] = r_i
        s_i = self._kb.get_id(subj, 1)
        o_i = self._kb.get_id(obj, 2)

        rels = self._tuple_rels_lookup.get((s_i, o_i))
        if rels:
            for i in xrange(len(rels)):
                if rels[i] != r_i:
                    self._sparse_indices.append([j, i])
                    self._sparse_values.append(rels[i])
            self._max_cols = max(self._max_cols, len(rels) + 1)
            # default relation
            self._sparse_indices.append([j, len(rels)])
        else:
            self._sparse_indices.append([j, 0])
        self._sparse_values.append(2 * self.__num_relations)

    def _finish_adding_triples(self, batch_size):
        self._feed_dict[self._sparse_indices_input] = self._sparse_indices
        self._feed_dict[self._sparse_values_input] = self._sparse_values
        self._feed_dict[self._shape_input] = [batch_size, self._max_cols]
        if batch_size < self._batch_size:
            self._feed_dict[self._rel_input] = self._rel_in[:batch_size]
        else:
            self._feed_dict[self._rel_input] = self._rel_in

    def _scoring_f(self):
        with tf.device("/cpu:0"):
           E_rels = tf.get_variable("E_r", [len(self._kb.get_symbols(0)), self._size])
           E_tup_rels = tf.get_variable("E_tup_r", [2*self.__num_relations+1, self._size])  # rels + inv rels + default rel

        self.e_rel = tf.tanh(tf.nn.embedding_lookup(E_rels, self._rel_input))
        # weighted sum of tuple rel embeddings
        sparse_tensor = tf.SparseTensor(self._sparse_indices_input, self._sparse_values_input, self._shape_input)
        # mean embedding
        self.e_tuple_rels = tf.tanh(tf.nn.embedding_lookup_sparse(E_tup_rels, sparse_tensor, None))

        return tf_util.dot(self.e_rel, self.e_tuple_rels)


class ModelN(ModelO):

    def _init_inputs(self):
        ModelO._init_inputs(self)
        self._rel_cooc_lookup = dict()
        for tup, rels in self._tuple_rels_lookup.iteritems():
            for rel in rels:
                for rel2 in rels:
                    if rel != rel2:
                        rel_cooc = (rel, rel2)
                        if rel_cooc not in self._rel_cooc_lookup:
                            self._rel_cooc_lookup[rel_cooc] = len(self._rel_cooc_lookup)

    def _add_triple_to_input(self, t, j):
        (rel, subj, obj) = t
        r_i = self._kb.get_id(rel, 0)
        self._rel_in[j] = r_i
        s_i = self._kb.get_id(subj, 1)
        o_i = self._kb.get_id(obj, 2)

        rels = self._tuple_rels_lookup.get((s_i, o_i))
        if rels:
            for i in xrange(len(rels)):
                if rels[i] != r_i:
                    cooc_id = self._rel_cooc_lookup.get((r_i, rels[i]))
                    if cooc_id:
                        self._sparse_indices.append([j, i])
                        self._sparse_values.append(cooc_id)
            self._max_cols = max(self._max_cols, len(rels) + 1)
            # default relation
            self._sparse_indices.append([j, len(rels)])
        else:
            self._sparse_indices.append([j, 0])
        self._sparse_values.append(len(self._rel_cooc_lookup))

    def _scoring_f(self):
        with tf.device("/cpu:0"):
           E_link_feat_weights = tf.get_variable("E_tup_r", [len(self._rel_cooc_lookup) + 1, 1])

        # weighted sum of tuple rel embeddings
        sparse_tensor = tf.SparseTensor(self._sparse_indices_input, self._sparse_values_input, self._shape_input)
        score = tf.reshape(tf.nn.embedding_lookup_sparse(E_link_feat_weights, sparse_tensor, None), [-1])

        return score


class ModelF(AbstractKBScoringModel):

    def _init_inputs(self):
        # create tuple to rel lookup
        self.__tuple_lookup = dict()
        for (rel, subj, obj), _, typ in self._kb.get_all_facts():
            if typ.startswith("train"):
                s_i = self._kb.get_id(subj, 1)
                o_i = self._kb.get_id(obj, 2)
                t = (s_i, o_i)
                if t not in self.__tuple_lookup:
                    self.__tuple_lookup[t] = len(self.__tuple_lookup)

        self._rel_input = tf.placeholder(tf.int64, shape=[None], name="rel")
        self._rel_in = np.zeros([self._batch_size], dtype=np.int64)
        self._tuple_input = tf.placeholder(tf.int64, shape=[None], name="tuple")
        self._tuple_in = np.zeros([self._batch_size], dtype=np.int64)
        self._feed_dict = {}

    def _add_triple_to_input(self, t, j):
        (rel, subj, obj) = t
        r_i = self._kb.get_id(rel, 0)
        self._rel_in[j] = r_i
        s_i = self._kb.get_id(subj, 1)
        o_i = self._kb.get_id(obj, 2)
        self._tuple_in[j] = self.__tuple_lookup[(s_i, o_i)]

    def _finish_adding_triples(self, batch_size):
        if batch_size < self._batch_size:
            self._feed_dict[self._rel_input] = self._rel_in[:batch_size]
            self._feed_dict[self._tuple_input] = self._tuple_in[:batch_size]
        else:
            self._feed_dict[self._rel_input] = self._rel_in
            self._feed_dict[self._tuple_input] = self._tuple_in

    def _scoring_f(self):
        with tf.device("/cpu:0"):
           E_rels = tf.get_variable("E_r", [len(self._kb.get_symbols(0)), self._size])
           E_tups = tf.get_variable("E_t", [len(self.__tuple_lookup), self._size])

        self.e_rel = tf.tanh(tf.nn.embedding_lookup(E_rels, self._rel_input))
        self.e_tup = tf.tanh(tf.nn.embedding_lookup(E_tups, self._tuple_input))

        return tf_util.dot(self.e_rel, self.e_tup)


class CombinedModel(AbstractKBScoringModel):

    def __init__(self, models, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False, composition=None, share_vars=False):
        self._models = []
        self.__name = '_'.join(models)
        if composition:
            self.__name = composition + "__" + self.__name
        with vs.variable_scope(self.name()):
            for m in models:
                self._models.append(model.create_model(kb, size, batch_size, False, num_neg, learning_rate,
                                                       l2_lambda, False, composition=composition, type=m))

        AbstractKBScoringModel.__init__(self, kb, size, batch_size, is_train, num_neg, learning_rate,
                                        l2_lambda, is_batch_training)

    def name(self):
        return self.__name

    def _scoring_f(self):
        weights = map(lambda _: tf.Variable(float(1)), xrange(len(self._models)-1))
        scores = [self._models[0]._scores]
        for i in xrange(len(self._models)-1):
            scores.append(self._models[i+1]._scores * weights[i])
        return tf.reduce_sum(tf.pack(scores), 0)

    def _add_triple_to_input(self, t, j):
        for m in self._models:
            m._add_triple_to_input(t, j)

    def _finish_adding_triples(self, batch_size):
        for m in self._models:
            m._finish_adding_triples(batch_size)
        for m in self._models:
            self._feed_dict.update(m._get_feed_dict())

    def _start_adding_triples(self):
        for m in self._models:
            m._start_adding_triples()
        self._feed_dict = dict()

    def _input_params(self):
        ips = []
        for m in self._models:
            ips.extend(m._input_params())