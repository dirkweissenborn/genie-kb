from model.models import *
import tensorflow as tf
import model
from tensorflow.models.rnn.seq2seq import rnn_decoder
from tensorflow.models.rnn.rnn_cell import *


class CompositionModel:

    def __init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        self._kb = kb
        self._size = size
        self._batch_size = batch_size
        self._rel2seq = rel2seq
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0)
        l_count = dict()
        total = 0
        max_l = 0
        self._vocab = {"#PADDING#": 0}
        for (rel, _, _), _, typ in kb.get_all_facts():
            s = self._rel2seq(rel)
            l = len(s)
            for word in s:
                if word not in self._vocab:
                    self._vocab[word] = len(self._vocab)
            max_l = max(max_l, l)
            if l not in l_count:
                l_count[l] = 0
            l_count[l] += 1
            total += 1
        self._seq_inputs = [tf.placeholder(tf.int64, shape=[None], name="seq_input%d" % i)
                            for i in xrange(max_l)]
        with vs.variable_scope("composition", initializer=model.default_init()):
            seq_outputs = self._comp_f()
        self._bucket_outputs = []
        ct = 0
        self._buckets = []
        for l in xrange(max_l):
            c = l_count.get(l)
            if c:
                ct += c
                if ct % (total / num_buckets) < c:
                    self._bucket_outputs.append(seq_outputs[l])
                    self._buckets.append(l)
        if len(self._buckets) >= num_buckets:
            self._buckets[-1] = max_l
            self._bucket_outputs[-1] = seq_outputs[-1]
        else:
            self._buckets.append(max_l)
            self._bucket_outputs.append(seq_outputs[-1])

        self._input = [[0]*self._batch_size for _ in xrange(max_l)]  # fill input with padding
        self._feed_dict = dict()
        train_params = filter(lambda v: "composition" in v.name, tf.trainable_variables())
        self._grad = tf.placeholder(tf.float32, shape=[None, self._size], name="rel_grad")
        self._grad_in = np.zeros((self._batch_size, self._size), dtype=np.float32)
        self._grads = [tf.gradients(o, train_params, self._grad) for o in self._bucket_outputs]
        self._bucket_update = [self.opt.apply_gradients(zip(grads, train_params))
                               for o, grads in zip(self._bucket_outputs, self._grads)]

    def _comp_f(self):
        # defaults to bag of words
        with tf.device("/cpu:0"):
            # word embedding matrix
            self.__E_ws = tf.get_variable("E_ws", [len(self._vocab), self._size])
            self.embeddings = map(lambda inp: tf.nn.embedding_lookup(self.__E_ws, inp), self._seq_inputs)
            out = [self.embeddings[0]]
            for i in xrange(1, len(self.embeddings)):
                out.append(out[-1] + self.embeddings[i])
        return map(tf.sigmoid, out)

    def name(self):
        return "BoW"

    def _rel2word_ids(self, rel):
        return [self._vocab[w] for w in self._rel2seq(rel)]

    def _finish_batch(self, batch_size, batch_length):
        if batch_size < self._batch_size:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp[:batch_size]
        else:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp

    def _add_input(self, b, j, inp):
        self._input[j][b] = inp

    def forward(self, sess, rels):
        self._last_rel_groups = dict()
        self._last_rels = []
        for i, rel in enumerate(rels):
            if rel in self._last_rel_groups:
                self._last_rel_groups[rel].append(i)
            else:
                self._last_rel_groups[rel] = [i]
                self._last_rels.append((rel, self._rel2word_ids(rel)))
        self._last_sorted = np.argsort(np.array(map(lambda x: len(x[1]), self._last_rels)))

        compositions = [None] * len(rels)
        i = 0
        while i < len(self._last_rels):
            batch_size = min(self._batch_size, len(self._last_rels)-i)
            batch_length = len(self._last_rels[self._last_sorted[i+batch_size-1]][1])
            bucket_id = 0
            for idx, bucket_length in enumerate(self._buckets):
                if bucket_length >= batch_length:
                    batch_length = bucket_length
                    bucket_id = idx
                    break
            for b in xrange(batch_size):
                _, rel_symbols = self._last_rels[self._last_sorted[i+b]]
                offset = batch_length-len(rel_symbols)
                for j in xrange(offset):
                    self._add_input(b, j, 0)  # padding
                for j, w_id in enumerate(rel_symbols):
                    self._add_input(b, j+offset, w_id)

            self._finish_batch(batch_size, batch_length)
            out = sess.run(self._bucket_outputs[bucket_id], feed_dict=self._feed_dict)
            for b in xrange(batch_size):
                rel_idx = self._last_sorted[i+b]
                rel, _ = self._last_rels[rel_idx]
                for j in self._last_rel_groups[rel]:
                    compositions[j] = out[b]

            i += batch_size
        return compositions

    #TODO: optimization
    def backward(self, sess, grads):
        # ineffective because forward pass is run here again
        i = 0
        self._feed_dict[self._grad] = self._grad_in
        while i < len(self._last_rels):
            batch_size = min(len(self._last_rels)-i, self._batch_size)
            batch_length = len(self._last_rels[self._last_sorted[i+batch_size-1]][1])
            bucket_id = 0
            for idx, bucket_length in enumerate(self._buckets):
                if bucket_length >= batch_length:
                    batch_length = bucket_length
                    bucket_id = idx
                    break

            self._grad_in *= 0.0  # zero grads

            for b in xrange(batch_size):
                rel, rel_symbols = self._last_rels[self._last_sorted[i+b]]
                offset = batch_length-len(rel_symbols)
                for j in xrange(offset):
                    self._add_input(b, j, 0)  # padding
                for j, w_id in enumerate(rel_symbols):
                    self._add_input(b, j+offset, w_id)

                for j in self._last_rel_groups[rel]:
                    self._grad_in[b] += grads[j]

            self._finish_batch(batch_size, batch_length)

            sess.run(self._bucket_update[bucket_id], feed_dict=self._feed_dict)
            i += batch_size


class RNNCompModel(CompositionModel):

    def __init__(self, cell, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        assert cell.output_size == size, "cell size must equal size for RNNs"
        self._cell = cell
        CompositionModel.__init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate)

    def _comp_f(self):
        self._init_state = tf.get_variable("init_state", [self._cell.state_size])
        shape = tf.shape(self._seq_inputs[0])  # current_batch_size x 1
        init = tf.tile(self._init_state, shape)
        init = tf.reshape(init, [-1, self._cell.state_size])
        out = embedding_rnn_decoder(self._seq_inputs, init, self._cell, len(self._vocab))[0]
        return out

    def name(self):
        return "RNN_" + self._cell.__class__.__name__


class LSTMCompModel(RNNCompModel):
    def __init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        RNNCompModel.__init__(self, BasicLSTMCell(size), kb, size, num_buckets, rel2seq, batch_size, learning_rate)

    def name(self):
        return "LSTM"


class TanhRNNCompModel(RNNCompModel):
    def __init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        RNNCompModel.__init__(self, BasicRNNCell(size), kb, size, num_buckets, rel2seq, batch_size, learning_rate)

    def name(self):
        return "RNN"


class GRUCompModel(RNNCompModel):
    def __init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        RNNCompModel.__init__(self, GRUCell(size), kb, size, num_buckets, rel2seq, batch_size, learning_rate)

    def name(self):
        return "GRU"


class BiRNNCompModel(CompositionModel):
    def __init__(self, cell, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        assert cell.output_size == size/2, "cell size must be size / 2 for BiRNNs"
        self._cell = cell
        CompositionModel.__init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate)
        self._rev_input = [[0]*self._batch_size for _ in xrange(len(self._input))]

    def _finish_batch(self, batch_size, batch_length):
        self._rev_input[:batch_length] = self._input[(batch_length-1)::-1]
        if batch_size < self._batch_size:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp[:batch_size]
            for seq_in, inp in zip(self._rev_seq_inputs, self._rev_input):
                self._feed_dict[seq_in] = inp[:batch_size]
        else:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp
            for seq_in, inp in zip(self._rev_seq_inputs, self._rev_input):
                self._feed_dict[seq_in] = inp

    def _comp_f(self):
        self._rev_seq_inputs = [tf.placeholder(tf.int64, shape=[None], name="seq_input%d" % i)
                                for i in xrange(len(self._seq_inputs))]
        self._init_state = tf.get_variable("init_state", [self._cell.state_size * 2])
        shape = tf.shape(self._seq_inputs[0])  # current_batch_size x 1
        init = tf.tile(self._init_state, shape)
        init = tf.reshape(init, [-1, self._cell.state_size * 2])

        init_fw, init_bw = tf.split(1, 2, init)

        with vs.variable_scope("forward_rnn"):
            out_fw = embedding_rnn_decoder(self._seq_inputs, init_fw, self._cell, len(self._vocab))[0]
        with vs.variable_scope("backward_rnn"):
            out_bw = embedding_rnn_decoder(self._rev_seq_inputs, init_bw, self._cell, len(self._vocab))[0]
        out = map(lambda (o_f, o_b): tf.concat(1, [o_f, o_b]), zip(out_fw, out_bw))
        return out

    def name(self):
        return "BiRNN_"+ self._cell.__class__.__name__


class BiLSTMCompModel(BiRNNCompModel):
    def __init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        BiRNNCompModel.__init__(self, BasicLSTMCell(size/2), kb, size, num_buckets, rel2seq, batch_size, learning_rate)

    def name(self):
        return "BiLSTM"


class BiTanhRNNCompModel(BiRNNCompModel):
    def __init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        BiRNNCompModel.__init__(self, BasicRNNCell(size/2), kb, size, num_buckets, rel2seq, batch_size, learning_rate)

    def name(self):
        return "BiRNN"


class BiGRUCompModel(BiRNNCompModel):
    def __init__(self, kb, size, num_buckets, rel2seq, batch_size, learning_rate=1e-2):
        BiRNNCompModel.__init__(self, GRUCell(size/2), kb, size, num_buckets, rel2seq, batch_size, learning_rate)

    def name(self):
        return "BiGRU"


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
        assert self._update, "model has to be created in training mode!"

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
            res = sess.run([self._loss, self._update] + self._input_grads, feed_dict=self._get_feed_dict())
            self._composition_backward(sess, res[2:])
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

        score = tf_util.dot(self.e_rel, s_o_prod)

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

        score = tf_util.dot(self.e_rel_s, self.e_subj) + tf_util.dot(self.e_rel_o, self.e_obj)

        return score


class CompModelO(CompositionalKBScoringModel):

    def __init__(self, kb, size, batch_size, comp_model, is_train=True, num_neg=200, learning_rate=1e-2,
                 which_sets=["train_text"]):
        self._which_sets = set(which_sets)
        CompositionalKBScoringModel.__init__(self, kb, size, batch_size, comp_model, is_train=True, num_neg=200, learning_rate=1e-2)

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

        self._rel_input = tf.placeholder(tf.int64, shape=[None, self._size], name="rel")
        self._rel_in = np.zeros([self._batch_size, self._size], dtype=np.float32)
        self._observed_input = tf.placeholder(tf.int64, shape=[None, self._size], name="observed")
        self._observed_in = np.zeros([self._batch_size, self._size], dtype=np.float32)
        self._feed_dict = {}

    def _start_adding_triples(self):
        self._max_cols = 1
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
        rels = self._tuple_inv_rels_lookup.get((s_i, o_i))
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
        return tf_util.dot(self._rel_input, self._observed_input)

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