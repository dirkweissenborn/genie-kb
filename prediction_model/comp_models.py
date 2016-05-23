from prediction_model.models import *
from data import load_fb15k237
import my_rnn

class CompositionUtil:
    """Holds information on decomposing relations, word vocabulary, etc."""

    def __init__(self, kb, rel2seq, max_size, include_args=False):
        self._kb = kb
        self._rel2seq = rel2seq

        # Creation of vocabulary
        self._vocab = {"#PADDING#": 0}
        self._vocab["#UNK#"] = 1
        self.max_length = 0
        l_count = {}
        total = 0
        must_contain = set()

        vocab = {}
        keys = []
        values = []
        if include_args:
            must_contain = must_contain.union(self._kb.get_symbols(1)).union(self._kb.get_symbols(2))
        for (rel, _, _), _, typ in kb.get_all_facts():
            s = rel2seq(rel)
            for word in s:
                if typ == "train":  # as opposed to train_text for example
                    must_contain.add(word)
                elif word not in vocab:
                    vocab[word] = len(vocab)
                    keys.append(word)
                    values.append(-1)
                else:
                    values[vocab[word]] -= 1
            l = len(s)
            self.max_length = max(self.max_length, l)
            if l not in l_count:
                l_count[l] = 0
            l_count[l] += 1
            total += 1

        if max_size <= 0:
            for k in keys:
                self._vocab[k] = len(self._vocab)
        else:
            keys, values = list(), list()
            if len(values) > 0:
                most_frequent = np.argsort(np.array(values))
                max_size = min(max_size-len(must_contain)-1, len(most_frequent))
                print("Total words: %d" % (len(keys)))
                print("Min word occurrence: %d" % -values[most_frequent[max_size-1]])
                for i in range(max_size):
                    k = keys[most_frequent[i]]
                    self._vocab[k] = len(self._vocab)
        for w in must_contain:
            self._vocab[w] = len(self._vocab)

    def rel2word_ids(self, rel):
        return [self._vocab.get(w, 1) for w in self._rel2seq(rel)]  # use unknown word if w is not in vocab

    @property
    def vocab(self):
        return self._vocab


class CompositionalKBPredictionModel(AbstractKBPredictionModel):

    def __init__(self, kb, size, batch_size, is_train=True, learning_rate=1e-2, comp_util=None, composition="GRU"):
        if comp_util is None:
            comp_util = CompositionUtil(kb, load_fb15k237.split_relations, 30000)
        self._comp_util = comp_util
        self._composition = composition
        AbstractKBPredictionModel.__init__(self, kb, size, batch_size, is_train, learning_rate)


    def _composition_function(self, inputs, length, init_state=None):
        if self._composition == "GRU":
            cell = GRUCell(self._size)
            return my_rnn.rnn(cell, inputs, sequence_length=length, initial_state=init_state, dtype=tf.float32)[1]
        elif self._composition == "LSTM":
            cell = BasicLSTMCell(self._size)
            out = my_rnn.rnn(cell, inputs, sequence_length=length, initial_state=init_state, dtype=tf.float32)[1]
            return tf.slice(out, [0, cell.state_size-cell.output_size],[-1,-1])
        else:
            raise NotImplementedError("Other compositions not implemented yet.")

    def _init_inputs(self):
        self._rel_input = tf.placeholder(tf.int64, shape=[None, self._comp_util.max_length], name="rel")
        self._rel_length = tf.placeholder(tf.int64, shape=[None], name="rel_length")
        self._x_input = tf.placeholder(tf.int64, shape=[None], name="x")
        self._y_candidates = tf.placeholder(tf.int64, shape=[None, None], name="candidates")
        self._y_input = tf.placeholder(tf.int64, shape=[None], name="y")
        self._is_inv = tf.placeholder(tf.bool, shape=(), name="invert")

        self._y_cands = np.zeros([self._batch_size, 2], dtype=np.int64)
        self._x_in = np.zeros([self._batch_size], dtype=np.int64)
        self._y_cands = np.zeros([self._batch_size, self._y_cands.shape[1]], dtype=np.int64)
        self._y_in = np.zeros([self._batch_size], dtype=np.int64)
        self._rel_in = np.zeros([self._batch_size, self._comp_util.max_length], dtype=np.int64)
        self._rel_l = np.zeros([self._batch_size], dtype=np.int64)

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

        new_rel_l = np.zeros([batch_size], dtype=np.int64)
        new_rel_l[:self._batch_size] = self._rel_l
        self._rel_l = new_rel_l

        self._batch_size = batch_size

    def _add_triple_and_negs_to_input(self, triple, neg_candidates, batch_idx, is_inv):
        if batch_idx >= self._batch_size:
            self._change_batch_size(max(self._batch_size*2, batch_idx))
        (rel, x, y) = triple
        w_ids = self._comp_util.rel2word_ids(rel)
        self._rel_in[batch_idx, :len(w_ids)] = w_ids
        self._rel_l[batch_idx] = len(w_ids)
        self._x_in[batch_idx] = self.arg_vocab[y] if is_inv else self.arg_vocab[x]
        if len(neg_candidates)+1 != self._y_cands.shape[1]:
            self._y_cands = np.zeros([self._batch_size, len(neg_candidates)+1], dtype=np.int64)

        self._y_cands[batch_idx] = [self.arg_vocab[x] if is_inv else self.arg_vocab[y]] + \
                                     [self.arg_vocab[neg] for neg in neg_candidates]

    def _add_triple_to_input(self, triple, batch_idx, is_inv):
        if batch_idx >= self._batch_size:
            self._change_batch_size(max(self._batch_size*2, batch_idx))
        (rel, x, y) = triple
        w_ids = self._comp_util.rel2word_ids(rel)
        self._rel_in[batch_idx, :len(w_ids)] = w_ids
        self._rel_l[batch_idx] = len(w_ids)
        self._x_in[batch_idx] = self.arg_vocab[y] if is_inv else self.arg_vocab[x]
        self._y_in[batch_idx] = self.arg_vocab[x] if is_inv else self.arg_vocab[y]

    def _finish_adding_triples(self, batch_size, is_inv):
        if batch_size < self._batch_size:
            self._feed_dict[self._x_input] = self._x_in[:batch_size]
            self._feed_dict[self._y_candidates] = self._y_cands[:batch_size]
            self._feed_dict[self._rel_input] = self._rel_in[:batch_size]
            self._feed_dict[self._y_input] = self._y_in[:batch_size]
            self._feed_dict[self._rel_length] = self._rel_l[:batch_size]
        else:
            self._feed_dict[self._x_input] = self._x_in
            self._feed_dict[self._y_candidates] = self._y_cands
            self._feed_dict[self._rel_input] = self._rel_in
            self._feed_dict[self._y_input] = self._y_in
            self._feed_dict[self._rel_length] = self._rel_l
        self._feed_dict[self._is_inv] = is_inv

    def name(self):
        return self.__class__.__name__ + "_" + self._composition


class CompDistMult(CompositionalKBPredictionModel):

    def _comp_f_fw(self):
        E_candidate = tf.get_variable("E_candidate", [len(self.arg_vocab), self._size])
        E_rel = tf.get_variable("E_rel", [len(self._comp_util.vocab), self._size])

        e_rel = tf.nn.embedding_lookup(E_rel, self._rel_input)
        e_rel_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._comp_util.max_length, e_rel)]
        e_rel = self._composition_function(e_rel_inputs, self._rel_length)

        e_arg = tf.tanh(tf.nn.embedding_lookup(E_candidate, self._x_input))
        tf.get_variable_scope().reuse_variables()
        return e_arg * e_rel

    def _comp_f_bw(self):
        E_candidate = tf.get_variable("E_candidate", [len(self.arg_vocab), self._size])
        E_rel = tf.get_variable("E_rel", [len(self._comp_util.vocab), self._size])

        #r_input = tf.reverse_sequence(self._rel_input, self._rel_length, 1, 0) ## This is the only difference to fw
        e_rel = tf.nn.embedding_lookup(E_rel, self._rel_input)
        e_rel_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._comp_util.max_length, e_rel)]
        e_rel = self._composition_function(e_rel_inputs, self._rel_length)

        e_arg = tf.tanh(tf.nn.embedding_lookup(E_candidate, self._x_input))
        tf.get_variable_scope().reuse_variables()  #  reuse E_candidate
        return e_arg * e_rel


class CompModelE(CompositionalKBPredictionModel):

    def _comp_f_fw(self):
        E_rel = tf.get_variable("E_rel_fw", [len(self._kb.get_symbols(0)), self._size])
        e_rel = tf.nn.embedding_lookup(E_rel, self._rel_input)
        e_rel_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._comp_util.max_length, e_rel)]
        e_rel = self._composition_function(e_rel_inputs, self._rel_length)
        return e_rel

    def _comp_f_bw(self):
        E_rel = tf.get_variable("E_rel_bw", [len(self._kb.get_symbols(0)), self._size])
        r_input = tf.reverse_sequence(self._rel_input, self._rel_length, 1, 0) ## This is the only difference to fw
        e_rel = tf.nn.embedding_lookup(E_rel, r_input)
        e_rel_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._comp_util.max_length, e_rel)]
        e_rel = self._composition_function(e_rel_inputs, self._rel_length)
        return e_rel


class CompModel(CompositionalKBPredictionModel):

    def _comp_f_fw(self):
        E_candidate = tf.get_variable("E_candidate", [len(self.arg_vocab), self._size])
        E_rel = tf.get_variable("E_rel", [len(self._comp_util.vocab), self._size])

        e_arg = tf.tanh(tf.nn.embedding_lookup(E_candidate, self._x_input))
        e = tf.nn.embedding_lookup(E_rel, self._rel_input)
        e_inputs = [e_arg] + [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._comp_util.max_length, e)]
        e = self._composition_function(e_inputs, self._rel_length)
        tf.get_variable_scope().reuse_variables()

        return e

    def _comp_f_bw(self):
        E_candidate = tf.get_variable("E_candidate", [len(self.arg_vocab), self._size])
        E_rel = tf.get_variable("E_rel", [len(self._comp_util.vocab), self._size])

        e_arg = tf.tanh(tf.nn.embedding_lookup(E_candidate, self._x_input))
        r_input = tf.reverse_sequence(self._rel_input, self._rel_length, 1, 0) ## This is the only difference to fw
        e = tf.nn.embedding_lookup(E_rel, r_input)
        e_inputs = [e_arg] + [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._comp_util.max_length, e)]
        e = self._composition_function(e_inputs, self._rel_length)
        tf.get_variable_scope().reuse_variables()

        return e
