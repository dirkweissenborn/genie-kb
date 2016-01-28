
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.models.rnn import *
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.models.rnn import rnn


class KBPopulation(object):

    def __init__(self, size, vocab, concept_vocab, max_facts, fact_left_buckets, fact_right_buckets,
                 query_left_buckets, query_right_buckets, batch_size, num_samples=200, max_grad=5,
                 dropout_prob=0, learning_rate=1e-3, num_layers=1, learning_rate_decay_factor=1, forward_only=True):
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
        self.learning_rate_decay_op = self.learning_rate.assign(learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False, name="step")
        self.max_facts = max_facts
        self.vocab = vocab
        self.concept_vocab = concept_vocab
        self.dropout_prob = dropout_prob
        self.max_grad = max_grad
        self.size = size
        self.num_layers = num_layers
        self.num_samples = num_samples

        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        init = tf.random_uniform_initializer(-0.1, 0.1)

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if not forward_only and dropout_prob > 0:
            lstm_cell = rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1-dropout_prob)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)
        cell = rnn_cell.EmbeddingWrapper(cell, len(self.vocab))

        #  Computation graphs for facts
        self.fact_left_rnns = []  # fact_num -> bucket_id
        self.fact_right_rnns = []

        for j in xrange(max_facts):
            self.fact_left_rnns.append(RNNWithBuckets(size, cell, fact_left_buckets, "%d" % j, "fact/left", j>0,
                                                      self.opt, self.max_grad, init=init, forward_only=forward_only))

            self.fact_right_rnns.append(RNNWithBuckets(size, cell, fact_right_buckets, "%d" % j, "fact/right", j>0,
                                                       self.opt, self.max_grad, init=init, forward_only=forward_only))

        #  Computation graphs for query
        self.query_left_inputs = []
        self.query_right_inputs = []

        self.query_left_rnn = RNNWithBuckets(size, cell, query_left_buckets, "", "query/left", False,
                                             self.opt, self.max_grad, init=init, forward_only=forward_only)

        self.query_right_rnn = RNNWithBuckets(size, cell, query_right_buckets, "", "query/right", False,
                                              self.opt, self.max_grad, init=init, forward_only=forward_only)

        # classification graph
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
        self.labels = tf.placeholder(tf.int64, shape=[None], name="labels")

        self.fact_concepts = tf.placeholder(tf.int64, shape=[None, num_samples], name="fact_concepts")

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

                self.scores = tf.segment_sum(scores, self.fact_concepts, name="attention_scores")

                # calculate actual attention with kb_bias
                #self.attention_weights = tf.nn.softmax(scores)  # batch_size x facts
                # attention_weights = tf.expand_dims(self.attention_weights, 1)  # batch_size x 1 x facts
                # start_state of decoder are attention weighted relation embeddings
                # attention = tf.batch_matmul(attention_weights, attention_states)  # batch_size x 1 x att_size
                # attention = tf.squeeze(attention, [1])  # batch_size x att_size

            proj = tf.matmul(self.query_input, tf.get_variable("w_q_inter", [attention_vec_size, size])) + \
                   tf.get_variable("b_inter", [size])
                  # tf.matmul(attention, tf.get_variable("w_a_inter", [attention_vec_size, size])) + \
            proj = tf.nn.tanh(proj)

            with tf.device("/cpu:0"):
                w = tf.get_variable("proj_w", [len(concept_vocab), size])
                b = tf.get_variable("proj_b", [len(concept_vocab)])
                labels = tf.reshape(self.labels, [-1, 1])
                #  chances for sampling those true labels for this input should be one
                true_label_prob = np.ones([batch_size, 1], dtype=np.float32)
                if not forward_only:
                    flat_cands = tf.reshape(self.neg_candidates, [-1])
                    flat_sampling_prob = tf.reshape(self.sampling_prob, [-1])
                    self.loss = \
                        tf.nn.sampled_softmax_loss(w, b, proj, labels, num_samples, len(concept_vocab),
                                                   remove_accidental_hits=False,
                                                   sampled_values=(flat_cands, true_label_prob, flat_sampling_prob))
                else:
                    self.cands = tf.concat(1,[tf.reshape(self.labels, [-1, 1]), self.neg_candidates])
                    # get weights for candidates
                    w_t = tf.transpose(w)
                    #w_cands = tf.nn.embedding_lookup(w, self.cands)  # batch_size x cands x size
                    #b_cands = tf.nn.embedding_lookup(b, self.cands)  # batch_size x cands
                    scores = tf.matmul(proj, w_t) + b
                    #proj = tf.expand_dims(proj, 2)  # batch_size x size x 1
                    # calc scores for all interactions
                    #scores = tf.squeeze(tf.batch_matmul(w_cands, proj), [2]) + b_cands  # batch_size x cands
                    self.output = tf.nn.softmax(scores)

        # Gradients and SGD update operation for training the model.
        if not forward_only:
            with vs.variable_scope("classification", initializer=init):
                train_params = filter(lambda v: _var_in_scope(v), tf.trainable_variables())
            params = [self.query_input] + self.attention_states_input + train_params
            gradients = tf.gradients(self.loss, params)
            self.attention_states_grads = gradients[1:-len(train_params)]
            self.query_input_grad = gradients[0]
            self.grads = gradients[-len(train_params):]
            clipped_gradients = _clip_by_value(self.grads, -self.max_grad, self.max_grad)
            self.update = self.opt.apply_gradients(zip(clipped_gradients, train_params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())


class RNNWithBuckets(object):

    def __init__(self, size, cell, buckets, name, scope, reuse, opt, max_grad, init=None, forward_only=True):
        self.buckets = buckets
        self.inputs = []
        self.outputs = []
        with vs.variable_scope(scope, initializer=init):
            if reuse:
                vs.get_variable_scope().reuse_variables()
            with vs.variable_scope(name):
                for i in xrange(buckets[-1]):
                    self.inputs.append(tf.placeholder(tf.int32, shape=[None], name="input{0}".format(i)))

            outputs, _ = rnn.rnn(cell, self.inputs, dtype=tf.float32)
            for i in buckets:
                self.outputs.append(outputs[i-1])

        if not forward_only:
            self.updates = []
            self.grads = []
            with vs.variable_scope(scope):
                with vs.variable_scope(name):
                    self.out_grad = tf.placeholder(tf.float32, shape=[None, size], name="grad")
                params = filter(lambda v: _var_in_scope(v), tf.trainable_variables())
            for o in self.outputs:  # create update ops for each bucket
                grads = tf.gradients(o, params, self.out_grad)
                self.grads.append(grads)
                clipped_grads = _clip_by_value(grads, -max_grad, max_grad)
                self.updates.append(opt.apply_gradients(zip(clipped_grads, params)))


def _clip_by_value(gradients, min_value, max_value):
    return [tf.clip_by_value(g, min_value, max_value) for g in gradients]


def get_dep_tensors(output_tensors, input_tensors, current=None):
    res = set()
    for o in output_tensors:
        if o not in input_tensors:  # we do not want to add placeholders, for example
            current_new = set()
            current_new.add(o)
            if current:
                current_new = current_new.union(current)
            res = res.union(get_dep_tensors(o.op.inputs, input_tensors, current_new))
        else:
            res = res.union(current)
        # return current set if we reached input tensor other wise discard current
    return res


def _var_in_scope(v):
    return v and v.name.startswith(tf.get_variable_scope().name)


def load_model(sess, path, batch_size=1, forward_only=True):
    with open(os.path.join(path, "config.pkl"), "r") as f:
        c = pickle.load(f)  # configuration
        model = KBPopulation(c["size"], c["vocab"], c["concept_vocab"], c["max_facts"], c["fact_left_buckets"],
                             c["fact_right_buckets"], c["query_left_buckets"], c["query_right_buckets"],
                             batch_size, num_samples=c["num_samples"], dropout_prob=c["dropout_prob"],
                             num_layers=c["num_layers"], max_grad=c["max_grad"], forward_only=forward_only)
        model.saver.restore(sess, os.path.join(path, "model.tf"))
    return model


def save_model(sess, path, model):
    configuration = {"size": model.size,
                     "vocab": model.vocab,
                     "concept_vocab": model.concept_vocab,
                     "max_facts": model.max_facts,
                     "fact_left_buckets": model.fact_left_rnns[0].buckets,
                     "fact_right_buckets": model.fact_right_rnns[0].buckets,
                     "query_left_buckets": model.query_left_rnn.buckets,
                     "query_right_buckets": model.query_right_rnn.buckets,
                     "num_layers": model.num_layers,
                     "num_samples": model.num_samples,
                     "max_grad": model.max_grad,
                     "dropout_prob": model.dropout_prob}
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "config.pkl"), "w") as f:
        pickle.dump(configuration, f)
    model.saver.save(sess, os.path.join(path, "model.tf"))


def sample_model(sess, batch_size=1, forward_only=False):
    size = 2
    vocab = {"a": 0, "b": 1}
    concept_vocab = {"c1": 0, "c2": 1}
    max_facts = 3
    fact_left_buckets = [1, 3]
    fact_right_buckets = [1, 2]
    query_left_buckets = [1, 2]
    query_right_buckets = [1, 2]
    model = KBPopulation(size, vocab, concept_vocab, max_facts, fact_left_buckets, fact_right_buckets,
                         query_left_buckets, query_right_buckets, batch_size, num_samples=1, forward_only=forward_only)
    sess.run(tf.initialize_all_variables())
    return model


def test_forward_backward():
    s = tf.Session()

    batch_size = 2
    model = sample_model(s, batch_size=batch_size)
    s.run(tf.initialize_all_variables())

    # ----- FORWARD ------
    # facts
    fact_left_bucket_ids = [0, 0, 0]
    fact_right_bucket_ids = [0, 0, 0]

    fact_inputs = {}
    fact_outputs = []
    for i in xrange(model.max_facts):
        fact_outputs.append(model.fact_left_rnns[i].outputs[fact_left_bucket_ids[i]])
        fact_outputs.append(model.fact_right_rnns[i].outputs[fact_right_bucket_ids[i]])
        fact_inputs[model.fact_left_rnns[i].inputs[0]] = np.zeros([batch_size])  # Example
        fact_inputs[model.fact_right_rnns[i].inputs[0]] = np.ones([batch_size])  # Example

    dep_fact_tensors = list(get_dep_tensors(fact_outputs, fact_inputs.keys()))
    dep_fact_tensors = fact_outputs + filter(lambda t: t not in fact_outputs, dep_fact_tensors)
    fact_state = s.run(dep_fact_tensors, fact_inputs)

    # concatenate left and right fact embeddings for each fact
    concat_facts = []
    for i in xrange(model.max_facts):
        concat_facts.append(np.concatenate((fact_state[2*i], fact_state[2*i+1]), 1))

    # Query
    query_left_bucket_id = 0
    query_right_bucket_id = 0

    query_inputs = {}
    query_outputs = [model.query_left_rnn.outputs[query_left_bucket_id],
                     model.query_right_rnn.outputs[query_right_bucket_id]]

    query_inputs[model.query_left_rnn.inputs[0]] = np.zeros([batch_size])  # Example
    query_inputs[model.query_right_rnn.inputs[0]] = np.ones([batch_size])  # Example

    dep_query_tensors = list(get_dep_tensors(query_outputs, query_inputs.keys()))
    dep_query_tensors = query_outputs + filter(lambda t: t not in query_outputs, dep_query_tensors)
    query_state = s.run(dep_query_tensors, query_inputs)

    concat_query = np.concatenate((query_state[0], query_state[1]), 1)

    # Classification
    classification_input = dict(zip(model.attention_states_input, concat_facts))
    classification_input[model.query_input] = concat_query
    classification_input[model.neg_candidates] = np.array([[1], [0]])
    classification_input[model.labels] = np.array([0,1])
    classification_input[model.sampling_prob] = np.array([[1.0], [1.0]])

    grads = s.run([model.update, model.loss, model.query_input_grad] + model.attention_states_grads,
                  feed_dict=classification_input)

    loss = grads[1]
    query_grad = grads[2]
    fact_grads = grads[3:]

    # --- BACKWARD ---
    # facts
    fact_updates = []
    fact_inputs_u = dict(fact_inputs)
    for i in xrange(model.max_facts):
        g = fact_grads[i]
        fact_inputs_u[model.fact_left_rnns[i].out_grad] = g[:, :g.shape[1]/2]
        fact_inputs_u[model.fact_right_rnns[i].out_grad] = g[:, g.shape[1]/2:]
        fact_updates.append(model.fact_left_rnns[i].updates[fact_left_bucket_ids[i]])
        fact_updates.append(model.fact_right_rnns[i].updates[fact_right_bucket_ids[i]])

    for i in xrange(len(dep_fact_tensors)):
        fact_inputs_u[dep_fact_tensors[i]] = fact_state[i]

    s.run(fact_updates, fact_inputs_u)
    
    # query
    query_updates = []
    query_updates.append(model.query_left_rnn.updates[query_left_bucket_id])
    query_updates.append(model.query_right_rnn.updates[query_right_bucket_id])
    query_inputs_u = dict(query_inputs)
    query_inputs_u[model.query_left_rnn.out_grad] = query_grad[:, :query_grad.shape[1]/2]
    query_inputs_u[model.query_right_rnn.out_grad] = query_grad[:, query_grad.shape[1]/2:]

    for i in xrange(len(dep_query_tensors)):
        query_inputs_u[dep_query_tensors[i]] = query_state[i]

    s.run(query_updates, query_inputs_u)
