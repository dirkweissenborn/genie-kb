from model.models import *
import tensorflow as tf
import model
from tensorflow.models.rnn.rnn_cell import *
from tf_util import get_tensors

class CompositionUtil:
    """Holds information on decomposing relations, word vocabulary, buckets, their sizes etc."""

    def __init__(self, kb, num_buckets, rel2seq, max_size):
        self._kb = kb
        self._rel2seq = rel2seq

        # Creation of vocabulary
        self._vocab = {"#PADDING#": 0}
        self._vocab["#UNK#"] = 1
        counts = {}
        max_l = 0
        l_count = {}
        total = 0
        must_contain = set()
        for (rel, _, _), _, typ in kb.get_all_facts():
            s = rel2seq(rel)
            for word in s:
                if typ == "train":  # as opposed to train_text for example
                    must_contain.add(word)
                elif word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1
            l = len(s)
            max_l = max(max_l, l)
            if l not in l_count:
                l_count[l] = 0
            l_count[l] += 1
            total += 1

        if max_size <= 0:
            for k in counts:
                self._vocab[k] = len(self._vocab)
        else:
            keys, values = list(), list()
            for k, v in counts.iteritems():
                if k not in must_contain:
                    keys.append(k)
                    values.append(-v)

            most_frequent = np.argsort(np.array(values))
            for w in must_contain:
                self._vocab[w] = len(self._vocab)
            max_size = min(max_size-len(must_contain)-1, len(most_frequent))
            for i in xrange(max_size):
                k = keys[most_frequent[i]]
                self._vocab[k] = len(self._vocab)

        ct = 0
        self.buckets = []
        for l in xrange(max_l+1):
            c = l_count.get(l)
            if c:
                ct += c
                if ct % (total / num_buckets) < c:
                    self.buckets.append(l)

        if self.buckets and self.buckets[-1] != max_l:
            if len(self.buckets) >= num_buckets:
                self.buckets[-1] = max_l
            else:
                self.buckets.append(max_l)

    def rel2word_ids(self, rel):
        return [self._vocab.get(w, 1) for w in self._rel2seq(rel)]  # use unknown word if w is not in vocab

    @property
    def vocab(self):
        return self._vocab


class CompositionFunction:

    def __init__(self, size, batch_size, comp_util, learning_rate=1e-2):
        self._comp_util = comp_util
        self._size = size
        self._batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0)
        max_l = comp_util.buckets[-1]  # maximum length
        self._seq_inputs = [tf.placeholder(tf.int64, shape=[None], name="seq_input%d" % i)
                            for i in xrange(max_l)]
        with vs.variable_scope("composition", initializer=model.default_init()):
            seq_outputs = self._comp_f()
        self._bucket_outputs = []

        for l in comp_util.buckets:
            self._bucket_outputs.append(seq_outputs[l-1])

        self._input = [[0]*self._batch_size for _ in xrange(max_l)]  # fill input with padding
        self._feed_dict = dict()
        train_params = filter(lambda v: "composition" in v.name, tf.trainable_variables())
        self._grad = tf.placeholder(tf.float32, shape=[None, self._size], name="rel_grad")
        self._grad_in = np.zeros((self._batch_size, self._size), dtype=np.float32)
        self._grads = [tf.gradients(o, train_params, self._grad) for o in self._bucket_outputs]
        self._bucket_update = [self.opt.apply_gradients(zip(grads, train_params))
                               for o, grads in zip(self._bucket_outputs, self._grads)]
        #self._state_tensors = [[o] + list(get_tensors([o], self._seq_inputs, False)) for o in self._bucket_outputs]

    def _comp_f(self):
        pass

    def name(self):
        return "Composition"

    def _finish_batch(self, batch_size, batch_length):
        if batch_size < self._batch_size:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp[:batch_size]
            self._feed_dict[self._grad] = self._grad_in[:batch_size]
        else:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp
            self._feed_dict[self._grad] = self._grad_in

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
                self._last_rels.append((rel, self._comp_util.rel2word_ids(rel)))
        self._last_sorted = np.argsort(np.array(map(lambda x: len(x[1]), self._last_rels)))

        compositions = [None] * len(rels)
        i = 0
        self._states = []
        while i < len(self._last_rels):
            batch_size = min(self._batch_size, len(self._last_rels)-i)
            batch_length = len(self._last_rels[self._last_sorted[i+batch_size-1]][1])
            bucket_id = 0
            for idx, bucket_length in enumerate(self._comp_util.buckets):
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
            #if i > 0:
            out = sess.run(self._bucket_outputs[bucket_id], feed_dict=self._feed_dict)
           # else:
            #state = sess.run(self._state_tensors[bucket_id], feed_dict=self._feed_dict)
            #out = state[0]
            #self._states.append(state)
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
        c = 0
        while i < len(self._last_rels):
            batch_size = min(len(self._last_rels)-i, self._batch_size)
            batch_length = len(self._last_rels[self._last_sorted[i+batch_size-1]][1])
            bucket_id = 0
            for idx, bucket_length in enumerate(self._comp_util.buckets):
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

            #if i > 0:
            sess.run(self._bucket_update[bucket_id], feed_dict=self._feed_dict)
            #else:
            #feed = dict(zip(self._state_tensors[bucket_id], self._states[c]))
            #feed.update(self._feed_dict)
            #sess.run(self._bucket_update[bucket_id], feed_dict=feed)
            #c += 1

            i += batch_size
        # free states
        #self._states = None


class BoWCompF(CompositionFunction):
    def _comp_f(self):
        with tf.device("/cpu:0"):
            # word embedding matrix
            self.__E_ws = tf.get_variable("E_ws", [len(self._comp_util.vocab), self._size])
            self.embeddings = map(lambda inp: tf.nn.embedding_lookup(self.__E_ws, inp), self._seq_inputs)
        out = [self.embeddings[0]]
        for i in xrange(1, len(self.embeddings)):
            out.append(tf.add(out[i-1], self.embeddings[i]))
        return map(tf.tanh, out)

    def name(self):
        return "BoW"


class RNNCompF(CompositionFunction):

    def __init__(self, cell, size, batch_size, comp_util, learning_rate=1e-2):
        assert cell.output_size == size, "cell size must equal size for RNNs"
        self._cell = cell
        CompositionFunction.__init__(self, size, batch_size, comp_util, learning_rate)
        #self._state_tensors = [[o] + list(get_tensors([o], self._seq_inputs + [self._init_state], False))
        #                       for o in self._bucket_outputs]

    def _comp_f(self):
        self._init_state = tf.get_variable("init_state", [self._cell.state_size])
        shape = tf.shape(self._seq_inputs[0])  # current_batch_size x 1
        init = tf.tile(self._init_state, shape)
        init = tf.reshape(init, [-1, self._cell.state_size])
        out = embedding_rnn_decoder(self._seq_inputs, init, self._cell, len(self._comp_util.vocab))[0]
        return out

    def name(self):
        return "RNN_" + self._cell.__class__.__name__


class LSTMCompF(RNNCompF):
    def __init__(self, size, batch_size, comp_util, learning_rate=1e-2):
        RNNCompF.__init__(self, BasicLSTMCell(size), size, batch_size, comp_util, learning_rate)

    def name(self):
        return "LSTM"


class TanhRNNCompF(RNNCompF):
    def __init__(self, size, batch_size, comp_util, learning_rate=1e-2):
        RNNCompF.__init__(self, BasicRNNCell(size), size, batch_size, comp_util, learning_rate)

    def name(self):
        return "RNN"


class GRUCompF(RNNCompF):
    def __init__(self, size, batch_size, comp_util, learning_rate=1e-2):
        RNNCompF.__init__(self, GRUCell(size), size, batch_size, comp_util, learning_rate)

    def name(self):
        return "GRU"


class BiRNNCompF(CompositionFunction):
    def __init__(self, cell, size, batch_size, comp_util, learning_rate=1e-2):
        assert cell.output_size == size/2, "cell size must be size / 2 for BiRNNs"
        self._cell = cell
        CompositionFunction.__init__(self, size, batch_size, comp_util, learning_rate)
        self._rev_input = [[0]*self._batch_size for _ in xrange(len(self._input))]
        #self._state_tensors = [[o] + list(get_tensors([o], self._seq_inputs + [self._init_state], False))
        #                       for o in self._bucket_outputs]

    def _finish_batch(self, batch_size, batch_length):
        self._rev_input[:batch_length] = self._input[(batch_length-1)::-1]
        if batch_size < self._batch_size:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp[:batch_size]
            for seq_in, inp in zip(self._rev_seq_inputs, self._rev_input):
                self._feed_dict[seq_in] = inp[:batch_size]
            self._feed_dict[self._grad] = self._grad_in[:batch_size]
        else:
            for seq_in, inp in zip(self._seq_inputs, self._input):
                self._feed_dict[seq_in] = inp
            for seq_in, inp in zip(self._rev_seq_inputs, self._rev_input):
                self._feed_dict[seq_in] = inp
            self._feed_dict[self._grad] = self._grad_in

    def _comp_f(self):
        self._rev_seq_inputs = [tf.placeholder(tf.int64, shape=[None], name="seq_input%d" % i)
                                for i in xrange(len(self._seq_inputs))]
        self._init_state = tf.get_variable("init_state", [self._cell.state_size * 2])
        shape = tf.shape(self._seq_inputs[0])  # current_batch_size
        init = tf.tile(self._init_state, shape)
        init = tf.reshape(init, [-1, self._cell.state_size * 2])

        init_fw, init_bw = tf.split(1, 2, init)

        with vs.variable_scope("forward_rnn"):
            out_fw = embedding_rnn_decoder(self._seq_inputs, init_fw, self._cell, len(self._comp_util.vocab))[0]
        with vs.variable_scope("backward_rnn"):
            out_bw = embedding_rnn_decoder(self._rev_seq_inputs, init_bw, self._cell, len(self._comp_util.vocab))[0]
        out = map(lambda (o_f, o_b): tf.concat(1, [o_f, o_b]), zip(out_fw, out_bw))
        return out

    def name(self):
        return "BiRNN_"+ self._cell.__class__.__name__


class BiLSTMCompF(BiRNNCompF):
    def __init__(self, size, batch_size, comp_util, learning_rate=1e-2):
        BiRNNCompF.__init__(self, BasicLSTMCell(size / 2), size, batch_size, comp_util, learning_rate)

    def name(self):
        return "BiLSTM"


class BiTanhRNNCompF(BiRNNCompF):
    def __init__(self, size, batch_size, comp_util, learning_rate=1e-2):
        BiRNNCompF.__init__(self, BasicRNNCell(size / 2), size, batch_size, comp_util, learning_rate)

    def name(self):
        return "BiRNN"


class BiGRUCompF(BiRNNCompF):
    def __init__(self, size, batch_size, comp_util, learning_rate=1e-2):
        BiRNNCompF.__init__(self, GRUCell(size / 2), size, batch_size, comp_util, learning_rate)

    def name(self):
        return "BiGRU"


class ConvCompF(CompositionFunction):
    def __init__(self, width, size, batch_size, comp_util, learning_rate=1e-2):
        self._width = width
        CompositionFunction.__init__(self, size, batch_size, comp_util, learning_rate)
        #self._state_tensors = [[o] + list(get_tensors([o], self._seq_inputs + [self._subj, self._obj], False))
        #                       for o in self._bucket_outputs]

    def _comp_f(self):
        conv_kernels = {j:tf.get_variable("W_%d" % j,[self._size, self._size]) for j in xrange(-self._width, self._width+1)}
        shape = tf.shape(self._seq_inputs[0])  # current_batch_size x 1
        self._subj = tf.get_variable("subject", [1, self._size])
        self._obj = tf.get_variable("object", [1, self._size])
        with tf.device("/cpu:0"):
            # word embedding matrix
            self.__E_ws = tf.get_variable("E_ws", [len(self._comp_util.vocab), self._size])
            embeddings = map(lambda inp: tf.nn.embedding_lookup(self.__E_ws, inp), self._seq_inputs)

        subj_h = tf.reshape(tf.matmul(self._subj, conv_kernels[-self._width]), [-1])
        obj_h = tf.reshape(tf.matmul(self._obj, conv_kernels[self._width]), [-1])
        subj_h = tf.reshape(tf.tile(subj_h, shape), [-1, self._size])
        obj_h = tf.reshape(tf.tile(obj_h, shape), [-1, self._size])
        bias = tf.reshape(tf.tile(tf.get_variable("bias", [self._size]), shape), [-1, self._size])
        convs = []
        out = []
        for i in xrange(len(embeddings)):
            sum = bias
            if i < 2*self._width - 2:
                # for these sequence lengths (i) there is no full convolution
                last_center = max(i + 1 - self._width, 0)
                sum = sum + subj_h
                for j in xrange(max(0, last_center-self._width), min(last_center + self._width, len(embeddings))):
                    w = conv_kernels[j-last_center]
                    sum = sum + tf.matmul(embeddings[j], w)
                # no pooling because there is only one convolution for sequences of length i
                out.append(tf.tanh(sum + obj_h))
            else:
                # for sequence length of this i, we have at least two full convolution
                last_center = i + 1 - self._width
                for j in xrange(-self._width, self._width):
                    position = last_center + j
                    if position == -1:
                        sum = sum + subj_h
                    else:
                        w = conv_kernels[j]
                        sum = sum + tf.matmul(embeddings[position], w)

                if len(convs) > 0:
                    # add obj to last convolution and pack with previous convolution
                    h = tf.pack(convs + [tf.tanh(sum + obj_h)])
                    # max pooling of all convolutions
                    out.append(tf.reduce_max(h, [0]))
                else:
                    # no previous convs so output is this conv
                    out.append(tf.tanh(sum + obj_h))

                if i+1 < len(embeddings):
                    #add next conv to convs
                    convs.append(tf.tanh(sum + tf.matmul(embeddings[i+1], conv_kernels[self._width])))

        return out

    def name(self):
        return "Conv"
