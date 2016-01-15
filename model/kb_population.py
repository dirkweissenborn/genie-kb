
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.models.rnn import *
from tensorflow.models.rnn import seq2seq
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.linalg_ops import *
from tensorflow.models.rnn import rnn


class KBPopulation(object):

    def __init__(self, size, vocab, concept_vocab, max_facts, left_fact_buckets, right_fact_buckets,
                 left_query_buckets, right_query_buckets, batch_size, num_samples=200, max_grad_norm=5,
                 dropout_prob=0, learning_rate=1e-3, num_layers=1, learning_rate_decay_factor=1, forward_only=True):
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
        self.learning_rate_decay_op = self.learning_rate.assign(learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False, name="step")

        self.vocab = vocab
        self.dropout_prob = dropout_prob
        self.max_facts = max_facts
        self.left_fact_buckets = left_fact_buckets
        self.right_fact_buckets = right_fact_buckets
        self.left_query_buckets = left_query_buckets
        self.right_query_buckets = right_query_buckets
        self.max_grad_norm = max_grad_norm

        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        if not forward_only:
            init = tf.random_uniform_initializer(-0.1, 0.1)

        fact_max_left = self.left_fact_buckets[-1]
        fact_max_right = self.right_fact_buckets[-1]

        query_max_left = self.left_query_buckets[-1]
        query_max_right = self.right_query_buckets[-1]

        vocab_size = len(self.vocab)
    
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if not forward_only and dropout_prob > 0:
            lstm_cell = rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=1-dropout_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
        cell = rnn_cell.EmbeddingWrapper(cell, len(self.vocab))

        def var_in_scope(v):
            return v.name.startswith(tf.get_variable_scope().name)

        #  Computation graphs for facts
        self.fact_left_inputs = []  # fact_num -> bucket_id
        self.fact_right_inputs = []
        self.fact_left_outputs = []
        self.fact_right_outputs = []

        with vs.variable_scope("fact", initializer=init):
            for j in xrange(max_facts):
                with vs.variable_scope("left", initializer=init):
                    fl_inputs = []
                    self.fact_left_inputs.append(fl_inputs)
                    for i in xrange(fact_max_left):
                        fl_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                        name="fact_left{0}{1}".format(j, i)))

                    left_outputs = _rnn_with_buckets(fl_inputs, left_fact_buckets, cell, reuse=j > 0)
                    self.fact_left_outputs.append(left_outputs)

                with vs.variable_scope("right", initializer=init):
                    fr_inputs = []
                    self.fact_right_inputs.append(fr_inputs)
                    for i in xrange(fact_max_right):
                        fr_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                        name="fact_right{0}{1}".format(j, i)))
                    right_outputs = _rnn_with_buckets(fr_inputs, right_fact_buckets, cell, reuse=j > 0)
                    # output for each bucket for each fact
                    self.fact_right_outputs.append(right_outputs)

        if not forward_only:
            self.fact_left_grad = []
            self.fact_right_grad = []

            self.fact_left_updates = []
            self.fact_right_updates = []

            for j in xrange(max_facts):
                # Left updates
                fl_grad = tf.placeholder(tf.float32, shape=[None, size], name="fact_left_grad{0}".format(j))
                self.fact_left_grad.append(fl_grad)
                fl_updates = []
                self.fact_left_updates.append(fl_updates)
                params = filter(lambda v: var_in_scope(v), tf.trainable_variables())
                for o in self.fact_left_outputs[j]:  # create update ops for each bucket
                    fl_param_grads = tf.gradients(o, params, fl_grad)
                    fl_updates.append(self._train_ops(params, fl_param_grads))

                # Right updates
                fr_grad = tf.placeholder(tf.float32, shape=[None, size], name="fact_right_grad{0}".format(j))
                self.fact_right_grad.append(fr_grad)
                fr_updates = []
                self.fact_right_updates.append(fr_updates)
                params = filter(lambda v: var_in_scope(v), tf.trainable_variables())
                for o in self.fact_right_outputs[j]:  # create updates for each bucket
                    fr_param_grads = tf.gradients(o, params, fr_grad)
                    fr_updates.append(self._train_ops(params, fr_param_grads))

        #  Computation graph for query
        self.query_left_inputs = []
        self.query_right_inputs = []

        with vs.variable_scope("query", initializer=init):
            with vs.variable_scope("left", initializer=init):
                for i in xrange(query_max_left):
                    self.query_left_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                  name="query_left{0}".format(i)))
                # output for each bucket
                self.query_left_outputs = _rnn_with_buckets(self.query_left_inputs, left_query_buckets, cell)

            with vs.variable_scope("right", initializer=init):
                for i in xrange(query_max_right):
                    self.query_right_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                   name="query_right{0}".format(i)))
                # output for each bucket
                self.query_right_outputs = _rnn_with_buckets(self.query_right_inputs, right_query_buckets, cell)

        if not forward_only:
            self.query_left_updates = []  # updates for each bucket
            self.query_left_grad = tf.placeholder(tf.float32, shape=[None, size], name="query_left_grad")

            params = filter(lambda v: var_in_scope(v), tf.trainable_variables())
            for o in self.query_left_outputs:  # create updates for each bucket
                ql_param_grads = tf.gradients(o, params, self.query_left_grad)
                self.query_left_updates.append(self._train_ops(params, ql_param_grads))

            self.query_right_updates = []  # updates for each bucket
            self.query_right_grad = tf.placeholder(tf.float32, shape=[None, size], name="query_right_grad")

            params = filter(lambda v: var_in_scope(v), tf.trainable_variables())
            for o in self.query_right_outputs:  # create updates for each bucket
                ql_param_grads = tf.gradients(o, params, self.query_right_grad)
                self.query_right_updates.append(self._train_ops(params, ql_param_grads))

        #  attention over facts given query vector + query vector -> softmax over concepts
        self.query_input = tf.placeholder(tf.float32, shape=[None, 2*size], name="query_input")
        self.attention_states_input = []
        attention_states = []
        attention_vec_size = 2 * size
        for j in xrange(max_facts):
            s = tf.placeholder(tf.float32, shape=[None, attention_vec_size], name="attention_state{0}".format(j))
            attention_states.append(tf.expand_dims(s, 1))
            self.attention_states_input.append(s)

        attention_states = tf.concat(1, attention_states)  # batch x length x att_size

        # candidates (i.e., their ids) are given by user, first candidate is assumed to be correct
        self.neg_candidates = tf.placeholder(tf.int64, shape=[None, num_samples], name="neg_candidates")
        # labels
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")

        # sampling prob for each candidate
        self.sampling_prob = tf.placeholder(tf.float32, shape=[None, num_samples], name="sampling_prob")

        with vs.variable_scope("classification", initializer=init):
            with vs.variable_scope("attention", initializer=init):
                # calc interaction vectors for all relations between relation and loast state
                query_interact = tf.matmul(self.query_input, tf.get_variable("w_proj", [attention_vec_size, size])) + \
                                 tf.get_variable("b_proj", [size])  # batch_size x size

                query_interact = tf.expand_dims(query_interact, 1)  # batch_size x 1 x size
                query_interact = tf.tile(query_interact, [1, max_facts, 1])  # batch_size x facts x size

                w_att = tf.get_variable("w_att", [1, attention_vec_size, size]) # 1 x att_size x size
                w_att = tf.tile(w_att, [batch_size, 1, 1])  # batch_size x att_size x size

                att_interact = tf.batch_matmul(attention_states, w_att)  # batch_size x length x size

                interact = tanh(query_interact + att_interact)  # batch_size x facts x size

                # calc scores for all interactions
                w_score = tf.get_variable("w_score", [1, size, 1])
                w_score = tf.tile(w_score, [batch_size, 1, 1])  # batch_size x size x 1
                scores = tf.squeeze(tf.batch_matmul(interact, w_score), [2])  # batch_size x facts

                # calculate actual attention with kb_bias
                self.attention_weights = tf.nn.softmax(scores)  # batch_size x facts
                attention_weights = tf.expand_dims(self.attention_weights, 1)  # batch_size x 1 x facts
                # start_state of decoder are attention weighted relation embeddings
                attention = tf.batch_matmul(attention_weights, attention_states)  # batch_size x 1 x att_size
                attention = tf.squeeze(attention, [1])  # batch_size x att_size

            proj = tf.matmul(self.query_input, tf.get_variable("w_q_inter", [attention_vec_size, size])) + \
                   tf.matmul(attention, tf.get_variable("w_a_inter", [attention_vec_size, size])) + \
                   tf.get_variable("b_inter", [size])
            proj = tf.nn.tanh(proj)

            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [len(concept_vocab), size])
                b = tf.get_variable("proj_b", [len(concept_vocab)])
                labels = tf.reshape(self.labels, [-1, 1])
                #  chances for sampling those true labels for this input should be one
                true_label_prob = np.ones([batch_size, 1], dtype=np.float32)
                if not forward_only:
                    flat_cands = tf.reshape(self.neg_candidates, [-1])
                    self.loss = \
                        tf.nn.sampled_softmax_loss(w, b, proj, labels, num_samples, len(concept_vocab),
                                                   sampled_values=(flat_cands, true_label_prob, self.sampling_prob))
                else:
                    # get weights for candidats
                    w_cands = tf.nn.embedding_lookup(w, self.neg_candidates)  # batch_size x cands x size
                    b_cands = tf.nn.embedding_lookup(b, self.neg_candidates)  # batch_size x cands
                    proj = tf.expand_dims(proj, 2)  # batch_size x size x 1
                    # calc scores for all interactions
                    scores = tf.squeeze(tf.batch_matmul(w_cands, proj), [2]) + b_cands  # batch_size x cands
                    self.output = tf.nn.softmax(scores)

        # Gradients and SGD update operation for training the model.
        params = [self.query_input] + self.attention_states_input + tf.trainable_variables()
        if not forward_only:
            train_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            self.attention_states_grads = gradients[1:-len(train_params)]
            self.query_input_grad = gradients[0]
            train_grads = gradients[-len(train_params):]
            self.classification_update = self._train_ops(train_params, train_grads, self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())

    def _train_ops(self, train_params, gradients, step=None):
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        return self.opt.apply_gradients(zip(clipped_gradients, train_params), global_step=step)


def _rnn_with_buckets(inputs, buckets, cell, reuse=False, name=None):
    if len(inputs) < buckets[-1]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(inputs), buckets[-1]))

    outputs = []
    with ops.op_scope(inputs, name, "rnn_with_buckets"):
        for j in xrange(len(buckets)):
            if j > 0 or reuse:
                vs.get_variable_scope().reuse_variables()
            bucket_encoder_inputs = [inputs[i] for i in xrange(buckets[j])]
            bucket_outputs, _ = rnn.rnn(cell, bucket_encoder_inputs, dtype=tf.float32)
            outputs.append(bucket_outputs[-1])
    return outputs

#size, vocab, concept_vocab, max_facts, fact_max, query_max, batch_size,
#num_samples=200, dropout_prob=0, learning_rate=1e-3, num_layers=1, learning_rate_decay_factor=1, forward_only=True
def load_model(sess, path, batch_size=1, forward_only=True):
    with open(os.path.join(path, "config.pkl"), "r") as f:
        c = pickle.load(f)  # configuration
        model = KBPopulation(c["size"], c["vocab"], c["concept_vocab"], c["max_facts"], c["fact_max"], c["query_max"],
                             batch_size, num_samples=c["num_samples"], dropout_prob=c["dropout_prob"],
                             num_layers=c["num_layers"], forward_only=forward_only)
        model.saver.restore(sess, os.path.join(path, "model.tf"))
    return model


def save_model(sess, path, model):
    configuration = {"source_vocab": model.source_vocab,
                     "target_vocab": model.target_vocab,
                     "relations": model.relations,
                     "tuple_vocab": model.tuple_vocab,
                     "kb_shape": model.kb_shape,
                     "buckets": model.buckets,
                     "size": model.size,
                     "num_layers": model.num_layers}
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path,"config.pkl"), "w") as f:
        pickle.dump(configuration, f)
    model.saver.save(sess, os.path.join(path, "model.tf"))

#size, vocab, concept_vocab, max_facts, left_fact_buckets, right_fact_buckets,
#left_query_buckets, right_query_buckets, batch_size, num_samples=200, max_grad_norm=5,
#dropout_prob=0, learning_rate=1e-3, num_layers=1, learning_rate_decay_factor=1, forward_only=True
def sample_model(sess, batch_size=1):
    size = 2
    vocab = {"a":0, "b":1}
    concept_vocab = {"c1":0, "c2":1}
    max_facts = 3
    left_fact_buckets = [1,3]
    right_fact_buckets = [1,2]
    left_query_buckets = [1,2]
    right_query_buckets = [1,2]
    model = KBPopulation(size, vocab, concept_vocab, max_facts, left_fact_buckets, right_fact_buckets,
                         left_query_buckets, right_query_buckets, batch_size, num_samples=1, forward_only=False)
    sess.run(tf.initialize_all_variables())
    return model



def test_forward_backward(sess, model):
    s = tf.Session()
    batch_size = 2
    model = sample_model(s, batch_size=batch_size)
    s.run(tf.initialize_all_variables())
    ##### FORWARD ######
    # facts
    left_fact_bucket_ids = [0, 0]
    right_fact_bucket_ids = [0, 0]

    fact_inputs = {}
    fact_outputs = []
    for i in xrange(model.max_facts):
        fact_outputs.append(model.fact_left_outputs[i][left_fact_bucket_ids[i]])
        fact_outputs.append(model.fact_right_outputs[i][left_fact_bucket_ids[i]])
        fact_inputs[model.fact_left_inputs[i][left_fact_bucket_ids[i]]] = np.zeros([batch_size, 1])  # Example
        fact_inputs[model.fact_right_inputs[i][right_fact_bucket_ids[i]]] = np.ones([batch_size, 1])  # Example

    fact_embeddings = s.run(fact_outputs, fact_inputs)
    # concatenate left and right fact embeddings for each fact
    concat_facts = []
    for i in xrange(model.max_facts):
        concat_facts.append(np.concatenate((fact_embeddings[2*i], fact_embeddings[2*i+1]), 1))

    # Query
    left_query_bucket_id = 0
    right_query_bucket_id = 0

    query_inputs = {}
    query_outputs = {}
    fact_inputs[model.query_left_inputs[left_query_bucket_id]] = np.zeros([batch_size, 1])  # Example
    fact_inputs[model.query_right_inputs[right_query_bucket_id]] = np.ones([batch_size, 1])  # Example

    fact_outputs.append(model.fact_left_outputs[i][left_fact_bucket_ids[i]])
    fact_outputs.append(model.fact_right_outputs[i][left_fact_bucket_ids[i]])
    fact_inputs[model.fact_left_inputs[i][left_fact_bucket_ids[i]]] = np.zeros([batch_size, 1])  # Example
    fact_inputs[model.fact_right_inputs[i][right_fact_bucket_ids[i]]] = np.ones([batch_size, 1])  # Example
    tf.assign()