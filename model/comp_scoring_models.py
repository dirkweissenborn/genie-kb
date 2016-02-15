from model.models import *
import tensorflow as tf
import model
from tensorflow.models.rnn.rnn_cell import *


class CompositionalKBScoringModel(AbstractKBScoringModel):
    def __init__(self, kb, size, batch_size, comp_model, is_train=True, num_neg=200, learning_rate=1e-2):
        self._comp_model = comp_model
        AbstractKBScoringModel.__init__(self, kb, size, batch_size, is_train, num_neg, learning_rate, 0.0, False)

    def _input_params(self):
        return [self._rel_input]

    def _init_inputs(self):
        self._rel_input = tf.placeholder(tf.float32, shape=[None, self._size], name="rel")
        self._subj_input = tf.placeholder(tf.int64, shape=[None], name="subj")
        self._obj_input = tf.placeholder(tf.int64, shape=[None], name="obj")
        self._subj_in = np.zeros([self._batch_size], dtype=np.int64)
        self._obj_in = np.zeros([self._batch_size], dtype=np.int64)
        self._rel_in = np.zeros([self._batch_size, self._size], dtype=np.float32)
        self._feed_dict = {}

    def _start_adding_triples(self):
        self._rels = []

    def _add_triple_to_input(self, t, j):
        (rel, subj, obj) = t
        self._subj_in[j] = self._kb.get_id(subj, 1)
        self._obj_in[j] = self._kb.get_id(obj, 2)
        self._rels.append(rel)

    def _finish_adding_triples(self, batch_size):
        if batch_size < self._batch_size:
            self._feed_dict[self._subj_input] = self._subj_in[:batch_size]
            self._feed_dict[self._obj_input] = self._obj_in[:batch_size]
            self._feed_dict[self._rel_input] = self._rel_in[:batch_size]
        else:
            self._feed_dict[self._subj_input] = self._subj_in
            self._feed_dict[self._obj_input] = self._obj_in
            self._feed_dict[self._rel_input] = self._rel_in

    def name(self):
        return self.__class__.__name__ + "__" + self._comp_model.name()

    def _get_feed_dict(self):
        return self._feed_dict

    def _composition_forward(self, sess):
        rel_embeddings = self._comp_model.forward(sess, self._rels)
        for b in xrange(len(rel_embeddings)):
            self._rel_in[b] = rel_embeddings[b]

    def _composition_backward(self, sess, grads):
        grad_list = [grads[0][b] for b in xrange(grads[0].shape[0])]
        self._comp_model.backward(sess, grad_list)

    def score_triples(self, sess, triples):
        i = 0
        result = np.zeros([len(triples)])
        while i < len(triples):
            batch_size = min(self._batch_size, len(triples)-i)
            self._start_adding_triples()
            for j in xrange(batch_size):
                self._add_triple_to_input(triples[i+j], j)
            self._finish_adding_triples(batch_size)
            self._composition_forward(sess)
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
        assert self._is_train, "model has to be created in training mode!"

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
        self._composition_forward(sess)

        if mode == "loss":
            return sess.run(self._loss, feed_dict=self._get_feed_dict())
        else:
            assert self._is_train, "training only possible in training state."
            if hasattr(self, "_update"):
                res = sess.run([self._loss, self._update] + self._input_grads, feed_dict=self._get_feed_dict())
                self._composition_backward(sess, res[2:])
            else:
                res = sess.run([self._loss] + self._input_grads, feed_dict=self._get_feed_dict())
                self._composition_backward(sess, res[1:])
            return res[0]


class CompDistMult(CompositionalKBScoringModel):
    def _scoring_f(self):
        with tf.device("/cpu:0"):
            E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._size])
            E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._size])

        self.e_subj = tf.tanh(tf.nn.embedding_lookup(E_subjs, self._subj_input))
        self.e_obj = tf.tanh(tf.nn.embedding_lookup(E_objs, self._obj_input))
        self.e_rel = self._rel_input  # relation is already embedded by composition function
        s_o_prod = self.e_obj * self.e_subj

        score = tf_util.batch_dot(self.e_rel, s_o_prod)

        return score


class CompModelE(CompositionalKBScoringModel):
    def _init_inputs(self):
        self._rel_input = tf.placeholder(tf.float32, shape=[None, 2*self._size], name="rel")
        self._subj_input = tf.placeholder(tf.int64, shape=[None], name="subj")
        self._obj_input = tf.placeholder(tf.int64, shape=[None], name="obj")
        self._subj_in = np.zeros([self._batch_size], dtype=np.int64)
        self._obj_in = np.zeros([self._batch_size], dtype=np.int64)
        self._rel_in = np.zeros([self._batch_size, 2*self._size], dtype=np.float32)
        self._feed_dict = {}

    def _scoring_f(self):
        with tf.device("/cpu:0"):
            E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._size])
            E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._size])

        self.e_rel_s, self.e_rel_o = tf.split(1, 2, self._rel_input)

        self.e_subj = tf.tanh(tf.nn.embedding_lookup(E_subjs, self._subj_input))
        self.e_obj = tf.tanh(tf.nn.embedding_lookup(E_objs, self._obj_input))

        score = tf_util.batch_dot(self.e_rel_s, self.e_subj) + tf_util.batch_dot(self.e_rel_o, self.e_obj)

        return score


class CompModelO(CompositionalKBScoringModel):

    def __init__(self, kb, size, batch_size, comp_model, is_train=True, num_neg=200, learning_rate=1e-2,
                 which_sets=["train_text"]):
        self._which_sets = set(which_sets)
        CompositionalKBScoringModel.__init__(self, kb, size, batch_size, comp_model, is_train=True, num_neg=200,
                                             learning_rate=1e-2)

    def _init_inputs(self):
        self._rel_ids = dict()
        len(self._kb.get_symbols(0))
        # create tuple to rel lookup
        self._tuple_rels_lookup = dict()
        for (rel, subj, obj), _, typ in self._kb.get_all_facts():
            if typ in self._which_sets:
                s_i = self._kb.get_id(subj, 1)
                o_i = self._kb.get_id(obj, 2)
                t = (s_i, o_i)
                if t not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t] = [rel]
                else:
                    self._tuple_rels_lookup[t].append(rel)
                t = (o_i, s_i)
                if t not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t] = [rel+"_inv"]
                else:
                    self._tuple_rels_lookup[t].append(rel+"_inv")

        self._rel_input = tf.placeholder(tf.float32, shape=[None, self._size], name="rel")
        self._rel_in = np.zeros([self._batch_size, self._size], dtype=np.float32)
        self._observed_input = tf.placeholder(tf.float32, shape=[None, self._size], name="observed")
        self._observed_in = np.zeros([self._batch_size, self._size], dtype=np.float32)
        self._feed_dict = {}

    def _start_adding_triples(self):
        self._rels = []
        self.__offsets = []

    def _add_triple_to_input(self, t, j):
        self.__offsets.append(len(self._rels))
        (rel, subj, obj) = t
        self._rels.append(rel)
        s_i = self._kb.get_id(subj, 1)
        o_i = self._kb.get_id(obj, 2)
        rels = self._tuple_rels_lookup.get((s_i, o_i))
        if rels:
            for i in xrange(len(rels)):
                if rels[i] != rel:
                    self._rels.append(rels[i])

    def _finish_adding_triples(self, batch_size):
        if batch_size < self._batch_size:
            self._feed_dict[self._rel_input] = self._rel_in[:batch_size]
            self._feed_dict[self._observed_input] = self._observed_in[:batch_size]
        else:
            self._feed_dict[self._rel_input] = self._rel_in
            self._feed_dict[self._observed_input] = self._observed_in

    def _scoring_f(self):
        return tf_util.batch_dot(self._rel_input, self._observed_input)

    def _input_params(self):
        return [self._rel_input, self._observed_input]

    def _composition_forward(self, sess):
        rel_embeddings = self._comp_model.forward(sess, self._rels)
        for b, off in enumerate(self.__offsets):
            self._rel_in[b] = rel_embeddings[off]
            end = self.__offsets[b+1] if len(self.__offsets) > (b+1) else len(self._rels)
            self._observed_in[b] *= 0.0
            for i in xrange(off+1, end):
                self._observed_in[b] += rel_embeddings[i]
            if (end-off-1) > 0:
                self._observed_in[b] /= (end-off-1)

    def _composition_backward(self, sess, grads):
        grad_list = []
        for b, off in enumerate(self.__offsets):
            grad_list.append(grads[0][b])
            end = self.__offsets[b+1] if len(self.__offsets) > (b+1) else len(self._rels)
            observed_grad = grads[1][b]
            if (end-off-1) > 0:
                observed_grad /= (end-off-1)
            for i in xrange(off+1, end):
                grad_list.append(observed_grad)
        self._comp_model.backward(sess, grad_list)

class CompWeightedModelO(CompModelO):

    def _init_inputs(self):
        self._rel_ids = dict()
        len(self._kb.get_symbols(0))
        # create tuple to rel lookup
        self._tuple_rels_lookup = dict()
        for (rel, subj, obj), _, typ in self._kb.get_all_facts():
            if typ in self._which_sets:
                s_i = self._kb.get_id(subj, 1)
                o_i = self._kb.get_id(obj, 2)
                t = (s_i, o_i)
                if t not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t] = [rel]
                else:
                    self._tuple_rels_lookup[t].append(rel)
                t = (o_i, s_i)
                if t not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t] = [rel+"_inv"]
                else:
                    self._tuple_rels_lookup[t].append(rel+"_inv")

        self._rel_input = tf.placeholder(tf.float32, shape=[None, self._size], name="rel")
        self._sparse_indices_input = tf.placeholder(tf.int64, name="sparse_indices")
        self._shape_input = tf.placeholder(tf.int64, name="shape")
        self._observed_input = tf.placeholder(tf.float32, [None, self._size], name="observation_inputs")
        self._feed_dict = {}

    def _start_adding_triples(self):
        self._sparse_indices = []
        self._rel_in = []
        self._rels = []
        self._obs_in = []
        self.__offsets = []
        self._max_cols = 1

    def _add_triple_to_input(self, t, b):
        self.__offsets.append(len(self._rels))
        (rel, subj, obj) = t
        s_i = self._kb.get_id(subj, 1)
        o_i = self._kb.get_id(obj, 2)

        rels = self._tuple_rels_lookup.get((s_i, o_i))
        if rels and any(rel_i != rel for rel_i in rels):
            self._rels.append(rel)
            for i in xrange(len(rels)):
                if rels[i] != rel:
                    self._sparse_indices.append([b, i])
                    self._rels.append(rels[i])
            self._max_cols = max(self._max_cols, len(rels))
        else:
            self._sparse_indices.append([b, 0])

    def _finish_adding_triples(self, batch_size):
        self._feed_dict[self._sparse_indices_input] = self._sparse_indices
        self._feed_dict[self._shape_input] = [batch_size, self._max_cols]
        self._feed_dict[self._observed_input] = self._obs_in
        self._feed_dict[self._rel_input] = self._rel_in

    def _scoring_f(self):
        scores_flat = tf_util.batch_dot(self._rel_input, self._observed_input)
        # for softmax set empty cells to something very small, so weight becomes practically zero
        scores = tf.sparse_to_dense(self._sparse_indices_input, self._shape_input,
                                    scores_flat, default_value=-1e-3)
        softmax = tf.nn.softmax(scores)
        weighted_scores = tf.reduce_sum(scores * softmax, reduction_indices=[1], keep_dims=False)

        return weighted_scores

    def _composition_forward(self, sess):
        rel_embeddings = self._comp_model.forward(sess, self._rels)
        zero_v = np.zeros([self._size], dtype=np.float32)
        for b, off in enumerate(self.__offsets):
            end = self.__offsets[b+1] if len(self.__offsets) > (b+1) else len(self._rels)
            if end == off:
                self._rel_in.append(zero_v)  # default relation if none was observed
                self._obs_in.append(zero_v)
            else:
                for i in xrange(end - (off+1)):
                    self._rel_in.append(rel_embeddings[off])
                self._obs_in.extend(rel_embeddings[off+1:end])

    def _composition_backward(self, sess, grads):
        grad_list = []
        skip = 0
        for b, off in enumerate(self.__offsets):
            end = self.__offsets[b+1] if len(self.__offsets) > (b+1) else len(self._rels)
            if end > off:
                off = off - b + skip
                end = end - b + skip
                rel_grad = grads[0][off]
                for i in xrange(off+1, end):
                    rel_grad += grads[0][i]
                grad_list.append(rel_grad)

                observed_grads = grads[1][off:end]
                grad_list.extend(observed_grads)
            else:
                skip += 1
        self._comp_model.backward(sess, grad_list)


class CompCombinedModel(CompositionalKBScoringModel):

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
        self._rels = []
        for m in self._models:
            m._finish_adding_triples(batch_size)
            self._feed_dict.update(m._get_feed_dict())
            if m._rels:
                self._rels.extend(m._rels)

    def _start_adding_triples(self):
        for m in self._models:
            m._start_adding_triples()

    def _input_params(self):
        ips = []
        for m in self._models:
            ips.extend(m._input_params())
        return ips

    def _composition_forward(self, sess):
        for m in self._models:
            m._composition_forward(sess)

    def _composition_backward(self, sess, grads):
        i = 0
        for m in self._models:
            inp_params = m._input_params()
            if inp_params:
                j = len(inp_params)
                m._composition_backward(sess, grads[i:i+j])
                i += j


