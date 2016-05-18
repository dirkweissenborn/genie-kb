from model.models import *

class RNNModel(AbstractKBScoringModel):
    def __init__(self, cell, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False, which_sets=["train_text"]):
        self._cell = cell
        AbstractKBScoringModel.__init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2,
                                        l2_lambda=0.0, is_batch_training=False)

    def _scoring_f(self):
        with tf.device("/cpu:0"):
            E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._cell.input_size])
            E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._cell.input_size])
            E_rels = tf.get_variable("E_r", [len(self._kb.get_symbols(0)), self._cell.input_size])

        self.e_subj = tf.nn.embedding_lookup(E_subjs, self._subj_input)
        self.e_obj = tf.nn.embedding_lookup(E_objs, self._obj_input)
        self.e_rel = tf.nn.embedding_lookup(E_rels, self._rel_input)

        self._init_state = tf.get_variable("init_state", [self._cell.state_size])
        shape = tf.shape(self._subj_input)  # current_batch_size x 1
        init = tf.tile(self._init_state, shape)
        init = tf.reshape(init, [-1, self._cell.state_size])

        out = rnn_decoder([self.e_subj, self.e_obj, self.e_rel], init, self._cell)[0][-1]
        weight = tf.get_variable("score_weight", [self._cell.output_size, 1])

        return tf.reshape(tf.matmul(out, weight), [-1])


class GRUModel(RNNModel):
    def __init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False, which_sets=["train_text"]):
        RNNModel.__init__(self, GRUCell(size), kb, size, batch_size, is_train, num_neg, learning_rate,
                          l2_lambda, is_batch_training)


class BiRNNModel(AbstractKBScoringModel):
    def __init__(self, cell, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False, which_sets=["train_text"]):
        assert cell.output_size == size/2, "cell size must be size / 2 for BiRNNs"
        self._cell = cell
        AbstractKBScoringModel.__init__(self, kb, size, batch_size, is_train, num_neg, learning_rate,
                                        l2_lambda, is_batch_training)

    def _scoring_f(self):
        with tf.device("/cpu:0"):
            E_subjs = tf.get_variable("E_s", [len(self._kb.get_symbols(1)), self._cell.input_size*2])
            E_objs = tf.get_variable("E_o", [len(self._kb.get_symbols(2)), self._cell.input_size*2])
            E_rels = tf.get_variable("E_r", [len(self._kb.get_symbols(0)), self._cell.input_size*2])

        self.e_subj = tf.nn.embedding_lookup(E_subjs, self._subj_input)
        self.e_obj = tf.nn.embedding_lookup(E_objs, self._obj_input)
        self.e_rel = tf.nn.embedding_lookup(E_rels, self._rel_input)

        self._init_state = tf.get_variable("init_state", [self._cell.state_size*2])
        shape = tf.shape(self._subj_input)  # current_batch_size x 1
        init = tf.tile(self._init_state, shape)
        init = tf.reshape(init, [-1, self._cell.state_size*2])
        init_fw, init_bw = tf.split(1, 2, init)

        e_subj_f, e_subj_b = tf.split(1, 2, self.e_subj)
        e_obj_f, e_obj_b = tf.split(1, 2, self.e_obj)
        e_rel_f, e_rel_b = tf.split(1, 2, self.e_rel)

        with vs.variable_scope("forward_rnn"):
            out_f = rnn_decoder([e_subj_f, e_rel_f, e_obj_f], init_fw, self._cell)[0][-1]
        with vs.variable_scope("backward_rnn"):
            out_b = rnn_decoder([e_obj_b, e_rel_b, e_subj_b], init_bw, self._cell)[0][-1]

        out = tf.concat(1, [out_f, out_b])
        out = tf.contrib.layers.fully_connected(out, self._size, activation_fn=tf.tanh,weight_init=None)
        weight = tf.get_variable("score_weight", [self._size, 1])

        return tf.reshape(tf.matmul(out, weight), [-1])


class BiGRUModel(BiRNNModel):
    def __init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False, which_sets=["train_text"]):
        BiRNNModel.__init__(self, GRUCell(int(size / 2)), kb, size, batch_size, is_train, num_neg, learning_rate,
                            l2_lambda, is_batch_training)


class BiLSTMModel(BiRNNModel):
    def __init__(self, kb, size, batch_size, is_train=True, num_neg=200, learning_rate=1e-2, l2_lambda=0.0,
                 is_batch_training=False, which_sets=["train_text"]):
        BiRNNModel.__init__(self, BasicLSTMCell(int(size / 2), 0.0), kb, size, batch_size, is_train, num_neg, learning_rate,
                            l2_lambda, is_batch_training)

class SpecializedLSTMCell(RNNCell):
    def __init__(self, num_units, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = array_ops.split(1, 2, state)
            concat = linear([inputs, h], 3 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, m = array_ops.split(1, 3, concat)

            new_c = c * sigmoid(m) * tanh(j) + (1-sigmoid(m)) * (c + sigmoid(i) * tanh(j))
            new_h = tanh(new_c)

        return new_h, array_ops.concat(1, [new_c, new_h])


class SpecializedLSTMCell2(RNNCell):
    def __init__(self, num_units, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = array_ops.split(1, 2, state)
            concat = tf.concat(1, [inputs, h])
            m = tf.contrib.layers.fully_connected(concat, 4 * self._num_units + self.input_size, weight_init=None)
            m1 = tf.expand_dims(array_ops.slice(m, [0, 0], [-1, self._num_units+self.input_size]), 2)
            m2 = tf.expand_dims(array_ops.slice(m, [0, self._num_units+self.input_size], [-1, 3 * self._num_units]), 1)

            m = tf.batch_matmul(m1, m2)
            bias = tf.get_variable("bias", (3 * self._num_units,))

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f = array_ops.split(1, 3, tf.squeeze(tf.batch_matmul(tf.expand_dims(concat, 1), m), [1]) + bias)

            new_c = c * sigmoid(f) + sigmoid(i) * tanh(j)
            new_h = tanh(new_c)

        return new_h, array_ops.concat(1, [new_c, new_h])



