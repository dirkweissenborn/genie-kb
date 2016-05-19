from model.comp_models import *
from data.load_fb15k237 import split_relations
import my_rnn

class CompBiRNNModel(CompositionalKBScoringModel):
    def __init__(self, cell, kb, size, batch_size, comp_f, is_train=True, num_neg=200, learning_rate=1e-2):
        assert cell.output_size == size//2, "cell size must be size / 2 for BiRNNs"
        self._cell = cell
        CompositionalKBScoringModel.__init__(self, kb, size, batch_size, comp_f, is_train, num_neg, learning_rate)

    def _scoring_f(self):
        inp = tf.reshape(tf.pack(self._seq_inputs), [self._max_length, -1, 1])
        rev_inp = tf.reverse_sequence(inp, self._lengths, 0, 1)
        rev_inp = [tf.squeeze(x,[0,2]) for x in tf.split(0, self._max_length, rev_inp)]
        self._init_state = tf.get_variable("init_state", [self._cell.state_size * 2])
        batch_size = tf.shape(self._seq_inputs[0])
        init = tf.tile(self._init_state, batch_size)
        init = tf.reshape(init, [-1, self._cell.state_size * 2])
        cell = EmbeddingWrapper(self._cell, len(self.vocab), self._cell.input_size)

        init_fw, init_bw = tf.split(1, 2, init)

        with vs.variable_scope("forward_rnn"):
            out_fw = my_rnn.rnn(cell, self._seq_inputs,init_fw, sequence_length=self._lengths)[1]
        with vs.variable_scope("backward_rnn"):
            out_bw = my_rnn.rnn(cell, rev_inp, init_bw, sequence_length=self._lengths)[1]

        #out_fw = tf.reshape(out_fw, [-1, self._size])
        #out_bw = tf.reshape(out_bw, [-1, self._size])  # L*BxS
        #offsets = tf.cast(tf.lin_space(0.0, tf.cast(batch_size, tf.float32), batch_size), tf.int64)
        #batch_size = tf.cast(batch_size[0], tf.int64)
        #out_fw = tf.gather(out_fw, self._lengths*batch_size + offsets)
        #out_bw = tf.gather(out_bw, self._lengths*batch_size + offsets)

        out_fw = tf.slice(out_fw,[0,cell.state_size-cell.output_size],[-1,-1])
        out_bw = tf.slice(out_bw,[0,cell.state_size-cell.output_size],[-1,-1])

        out = tf.concat(1, [out_fw, out_bw])
        out = tf.contrib.layers.fully_connected(out, self._size, activation_fn=lambda x:tf.maximum(x,0.0),weight_init=None)
        weight = tf.get_variable("score_weight", [self._size, 1])

        return tf.reshape(tf.matmul(out, weight), [-1])

    def _init_inputs(self):
        max_size = 10000
        # Creation of vocabulary
        self._vocab = {"#PADDING#": 0}
        self._vocab["#UNK#"] = 1
        counts = {}
        self._max_length = 0
        l_count = {}
        total = 0
        must_contain = set().union(self._kb.get_symbols(1)).union(self._kb.get_symbols(2))
        for (rel, _, _), _, typ in self._kb.get_all_facts():
            s = split_relations(rel)
            for word in s:
                if typ == "train":  # as opposed to train_text for example
                    must_contain.add(word)
                elif word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
            l = len(s)
            self._max_length = max(self._max_length, l+2)
            if l not in l_count:
                l_count[l] = 0
            l_count[l] += 1
            total += 1

        for w in must_contain:
            self._vocab[w] = len(self._vocab)
        if max_size <= 0:
            for k in counts:
                self._vocab[k] = len(self._vocab)
        else:
            keys, values = list(), list()
            for k, v in counts.items():
                if k not in must_contain:
                    keys.append(k)
                    values.append(-v)
            if values:
                most_frequent = np.argsort(np.array(values))
                print("Total words: %d" % len(counts))
                print("Min word occurrence: %d" % -values[most_frequent[max_size-1]])
                for i in range(max_size):
                    k = keys[most_frequent[i]]
                    self._vocab[k] = len(self._vocab)

        self._seq_inputs = [tf.placeholder(tf.int64, shape=[None], name="seq_input%d" % i)
                            for i in range(self._max_length)]
        self._lengths = tf.placeholder(tf.int64, shape=[None])
        self._feed_dict = {}

    def _start_adding_triples(self):
        self._input = []

    def _add_triple_to_input(self, t, j):
        (rel, subj, obj) = t
        l = [self._vocab.get(w, 1) for w in split_relations(rel)]  # use unknown word if w is not in vocab
        l.insert(0, self._vocab[subj])
        #l.append(self._vocab[obj])
        self._input.append(l)

    def _finish_adding_triples(self, batch_size):
        for i in range(self._max_length):
            l = []
            for inp in self._input:
                if i < len(inp):
                    l.append(inp[i])
                else:
                    l.append(0)
            self._feed_dict[self._seq_inputs[i]] = l

        self._feed_dict[self._lengths] = [len(x) for x in self._input]

    @property
    def vocab(self):
        return self._vocab