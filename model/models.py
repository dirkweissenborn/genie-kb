
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.seq2seq import *
import tf_util
import model
from tensorflow.models.rnn.rnn_cell import *
import my_rnn
import functools
from model.query import *


class QAModel:

    def __init__(self, size, batch_size, vocab_size, max_length, is_train=True, learning_rate=1e-2,
                 composition="GRU", max_queries=0, devices=None):
        self._vocab_size = vocab_size
        self._max_length = max_length
        self._size = size
        self._batch_size = batch_size
        self._is_train = is_train
        self._init = model.default_init()
        self._composition = composition
        self._max_queries = max_queries
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._device2 = devices[2 % len(devices)] if devices is not None else "/cpu:0"
        self._device3 = devices[3 % len(devices)] if devices is not None else "/cpu:0"
        with tf.device(self._device0):
            with vs.variable_scope(self.name(), initializer=self._init):
                self._init_inputs()
                with tf.device("/cpu:0"):
                    self.candidates = tf.get_variable("E_candidate", [vocab_size, self._size])
                    self.embeddings = tf.get_variable("E_words", [vocab_size, self._size])
                    answer, _ = tf.dynamic_partition(self._answer_input, self._query_partition, 2)
                    lookup_individual = tf.nn.embedding_lookup(self.candidates, answer)
                    cands,_ = tf.dynamic_partition(self._answer_candidates, self._query_partition, 2)
                    lookup = tf.nn.embedding_lookup(self.candidates, cands)
                self.num_queries = tf.Variable(self._max_queries, trainable=False, name="num_queries")
                self.query = self._comp_f()
                self.query = self._supporting_evidence(self.query)
                self._score = tf_util.batch_dot(lookup_individual, self.query)
                self._scores_with_negs = tf.squeeze(tf.batch_matmul(lookup, tf.expand_dims(self.query, [2])), [2])
                self._scores_with_negs += self._candidate_mask  # number of negative candidates can vary for each example

                if is_train:
                    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.9)
                    self.global_step = tf.Variable(0, trainable=False, name="step")

                    self.opt = tf.train.AdamOptimizer(self.learning_rate) #, beta1=0.0)

                    current_batch_size = tf.gather(tf.shape(self._scores_with_negs), [0])
                    labels = tf.constant([0], tf.int64)
                    labels = tf.tile(labels, current_batch_size)
                    loss = math_ops.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self._scores_with_negs, labels))

                    train_params = tf.trainable_variables()
                    self.training_weight = tf.Variable(1.0, trainable=False, name="training_weight")

                    self._loss = loss / math_ops.cast(current_batch_size, dtypes.float32)
                    self._grads = tf.gradients(self._loss, train_params, self.training_weight, colocate_gradients_with_ops=True)

                    if len(train_params) > 0:
                        self._update = self.opt.apply_gradients(zip(self._grads[:len(train_params)], train_params),
                                                                global_step=self.global_step)
                    else:
                        self._update = tf.assign_add(self.global_step, 1)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

    def __l2_normalize(self, x, factor=1.0, epsilon=1e-12, name=None):
        with ops.op_scope([x], name, "l2_normalize") as name:
            x = ops.convert_to_tensor(x, name="x")
            square_sum = tf.reduce_sum(tf.square(x), [1], keep_dims=True)
            # we change this to min (1, 1/norm)
            x_inv_norm = tf.rsqrt(math_ops.maximum(square_sum, epsilon))
            if factor != 1.0:
                x_inv_norm = x_inv_norm * factor
            x_inv_norm = tf.minimum(1.0, x_inv_norm)
            return tf.mul(x, x_inv_norm, name=name)

    def _composition_function(self, inputs, length, init_state=None):
        if self._composition == "GRU":
            cell = GRUCell(self._size)
            return my_rnn.rnn(cell, inputs, sequence_length=length,
                              initial_state=init_state, dtype=tf.float32)[0]
        elif self._composition == "LSTM":
            cell = BasicLSTMCell(self._size)
            init_state = tf.concat(1, [tf.zeros_like(init_state, tf.float32), init_state]) if init_state else None
            outs = my_rnn.rnn(cell, inputs, sequence_length=length, initial_state=init_state, dtype=tf.float32)[0]
            return tf.slice(outs, [0, cell.state_size-cell.output_size],[-1,-1])
        elif self._composition == "BiGRU":
            cell = GRUCell(self._size // 2, self._size)
            init_state_fw, init_state_bw = tf.split(1, 2, init_state) if init_state else (None, None)
            with vs.variable_scope("forward"):
                fw_outs = my_rnn.rnn(cell, inputs, sequence_length=length, initial_state=init_state_fw, dtype=tf.float32)[0]
            with vs.variable_scope("backward"):
                rev_inputs = tf.reverse_sequence(tf.pack(inputs), length, 0, 1)
                rev_inputs = [tf.reshape(x, [-1, self._size]) for x in tf.split(0, len(inputs), rev_inputs)]
                bw_outs = my_rnn.rnn(cell, rev_inputs, sequence_length=length, initial_state=init_state_bw, dtype=tf.float32)[0]
                bw_outs = tf.reverse_sequence(tf.pack(bw_outs), length, 0, 1)
                bw_outs = [tf.reshape(x, [-1, self._size]) for x in tf.split(0, len(inputs), bw_outs)]
            return [tf.concat(1, [fw_out, bw_out]) for fw_out, bw_out in zip(fw_outs, bw_outs)]
        else:
            raise NotImplementedError("Other compositions not implemented yet.")

    def name(self):
        return self.__class__.__name__

    def _comp_f(self):
        with tf.device("/cpu:0"):
            embedded = tf.nn.embedding_lookup(self.embeddings, self._context)
            batch_size = tf.gather(tf.shape(self._context), [0])

        with tf.device(self._device1):
            #use other device for backward rnn
            with vs.variable_scope("backward"):
                min_start = tf.segment_min(self._starts, self._span_context)
                init_state = tf.get_variable("init_state", [self._size])
                init_state = tf.reshape(tf.tile(init_state, batch_size), [-1, self._size])
                rev_embedded = tf.reverse_sequence(embedded, self._length, 1, 0)
                rev_e_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._max_length, rev_embedded)]
                outs_bw = self._composition_function(rev_e_inputs, self._length - min_start, init_state)
                outs_bw.insert(0, init_state)
                # reshape to all possible queries for all sequences. Dim[0]=batch_size*max_length+1.
                # "+1" because we include the initial state
                outs_bw = tf.reshape(tf.concat(1, outs_bw), [-1, self._size])
                # gather respective queries via their lengths-start (because reversed sequence)
                #  (with offset of context_index*(max_length+1))
                lengths_aligned = tf.gather(self._length, self._span_context)
                out_bw = tf.gather(outs_bw, (lengths_aligned - self._starts) + self._span_context * (self._max_length + 1))

        with tf.device(self._device2):
            with vs.variable_scope("forward"):
                e_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._max_length, embedded)]
                max_end = tf.segment_max(self._ends, self._span_context)
                init_state = tf.get_variable("init_state", [self._size])
                init_state = tf.reshape(tf.tile(init_state, batch_size), [-1,self._size])
                outs_fw = self._composition_function(e_inputs, max_end, init_state)
                outs_fw.insert(0, init_state)
                # reshape to all possible queries for all sequences. Dim[0]=batch_size*max_length+1.
                # "+1" because we include the initial state
                outs_fw = tf.reshape(tf.concat(1, outs_fw), [-1, self._size])
                # gather respective queries via their positions (with offset of context_index*(max_length+1))
                out_fw = tf.gather(outs_fw, self._ends + self._span_context * (self._max_length + 1))
            # form query from forward and backward compositions
            #query = tf.contrib.layers.fully_connected(tf.concat(1, [out_fw, out_bw]), self._size,
            #                                          activation_fn=None, weight_init=None)
            query = out_fw + out_bw #tf.tanh(query)#, float(self._size))
        # add supporting evidence to this query
        return query

    def _supporting_evidence(self, query):
        if self._max_queries == 0:
            return query
        else:
            with vs.variable_scope("supporting"):
                query, supp_queries = tf.dynamic_partition(query, self._query_partition, 2)
                num_queries = tf.shape(query)[0]
                
                with tf.device("/cpu:0"):
                    _, supp_answer_ids = tf.dynamic_partition(self._answer_input, self._query_partition, 2)
                    supp_answers = tf.nn.embedding_lookup(self.candidates, supp_answer_ids)
                    aligned_supp_answers = tf.gather(supp_answers, self._support_ids)  # and with respective answers

                    if self._max_queries > 1:
                        # used in multihop
                        answer_words = tf.nn.embedding_lookup(self.embeddings, supp_answer_ids)
                        aligned_answer_words = tf.gather(answer_words, self._support_ids)

                self.evidence_weights = []
                current_answers = [query]
                current_query = query

                aligned_support = tf.gather(supp_queries, self._support_ids)  # align supp_queries with queries

                for i in range(self._max_queries):
                    with vs.variable_scope("evidence"):
                        vs.get_variable_scope()._reuse = \
                                any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())

                        aligned_queries = tf.gather(current_query, self._query_ids)  # align queries

                        scores = tf_util.batch_dot(aligned_queries, aligned_support)
                        self.evidence_weights.append(scores)
                        e_scores = tf.exp(scores - tf.reduce_max(scores, [0], keep_dims=True))
                        norm = tf.unsorted_segment_sum(e_scores, self._query_ids, num_queries) + 0.00001 # for zero norms
                        norm = tf.reshape(norm, [-1, 1])
                        # this is basically the dot product between query and weighted supp_queries
                        summed_scores = tf.unsorted_segment_sum(tf.reshape(e_scores * scores, [-1,1]),
                                                                self._query_ids, num_queries) / norm
                        norm = tf.tile(norm, [1, self._size])
                        #scores = tf.tile(tf.reshape(scores, [-1, 1]), [1, self._size])
                        e_scores = tf.tile(tf.reshape(e_scores, [-1, 1]), [1, self._size])
                        weighted_answers = tf.unsorted_segment_sum(e_scores * aligned_supp_answers,
                                                                   self._query_ids, num_queries) / norm
                        weighted_answers = tf.tanh(weighted_answers)
                        #answer_weight = tf.contrib.layers.fully_connected(tf.concat(1, [weighted_queries,query]), self._size,
                        #                                                  activation_fn=tf.nn.relu, weight_init=None,
                        #                                                  bias_init=None)
                        answer_weight = tf.contrib.layers.fully_connected(summed_scores, 1,
                                                                          activation_fn=tf.nn.sigmoid,
                                                                          weight_init=tf.constant_initializer(0.0),
                                                                          bias_init=tf.constant_initializer(0.0))

                        new_answer = weighted_answers * answer_weight + (1.0-answer_weight) * current_answers[-1]
                        current_answers.append(tf.cond(tf.greater(self.num_queries, i),
                                                      lambda: new_answer,
                                                      lambda: current_answers[i]))

                        if i < self._max_queries - 1:
                            weighted_answer_words = tf.unsorted_segment_sum(e_scores * aligned_answer_words,
                                                                            self._query_ids, num_queries) / norm

                            weighted_queries = tf.unsorted_segment_sum(e_scores * aligned_support, self._query_ids, num_queries) / norm
                            c = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_answer_words]), self._size,
                                                                  activation_fn=tf.tanh, weight_init=None, bias_init=None)

                            gate = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_queries]),
                                                                     self._size,
                                                                     activation_fn=tf.sigmoid, weight_init=None,
                                                                     bias_init=tf.constant_initializer(1))
                            current_query = gate * current_query + (1-gate) * c

            tf.get_variable_scope().reuse_variables()
            return current_answers[-1]

    def _comp_f_fw(self, input, length):
        e_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._max_length, input)]
        e = self._composition_function(e_inputs, length)
        return e

    def _comp_f_bw(self, input, length):
        e_inputs = [tf.reshape(e, [-1, self._size]) for e in tf.split(1, self._max_length, input)]
        e = self._composition_function(e_inputs, length)
        return e

    def _init_inputs(self):
        #General
        with tf.device("/cpu:0"):
            self._context = tf.placeholder(tf.int64, shape=[None, self._max_length], name="context")
            self._answer_candidates = tf.placeholder(tf.int64, shape=[None, None], name="candidates")
            self._answer_input = tf.placeholder(tf.int64, shape=[None], name="answer")
            self._starts = tf.placeholder(tf.int64, shape=[None], name="span_start")
            self._ends = tf.placeholder(tf.int64, shape=[None], name="span_end")
            self._span_context = tf.placeholder(tf.int64, shape=[None], name="answer_position_context")
            self._candidate_mask = tf.placeholder(tf.float32, shape=[None, None], name="candidate_mask")
            self._length = tf.placeholder(tf.int64, shape=[None], name="context_length")

        self._ctxt = np.zeros([self._batch_size, self._max_length], dtype=np.int64)
        self._len = np.zeros([self._batch_size], dtype=np.int64)

        #Supporting Evidence
        # partition of queries (class 0) and support (class 1)
        self._query_partition = tf.placeholder(tf.int32, [None], "query_partition")
        # aligned support ids with query ids for supporting evidence
        self._support_ids = tf.placeholder(tf.int64, shape=[None], name="supp_ids")
        self._query_ids = tf.placeholder(tf.int64, shape=[None], name="query_ids")

        self._feed_dict = {}

    def _change_batch_size(self, batch_size):
        new_ctxt_in = np.zeros([batch_size, self._max_length], dtype=np.int64)
        new_ctxt_in[:self._batch_size] = self._ctxt
        self._ctxt = new_ctxt_in

        new_length = np.zeros([batch_size], dtype=np.int64)
        new_length[:self._batch_size] = self._len
        self._len = new_length

        self._batch_size = batch_size

    def _start_adding_examples(self):
        self._batch_idx = 0
        self._query_idx = 0
        self._support_idx = 0
        self._answer_cands = []
        self._answer_in = []
        self._s = []
        self._e = []
        self._span_ctxt = []
        # supporting evidence
        self._query_part = []
        self._queries = []
        self._support = []

        self.supporting_qa = []

    def _add_example(self, context_query, is_query=True):
        '''
        :param context: list of symbol indices
        :param starts: list of span starts to answer for
        :param ends: list of span ends
        :param answers: answer for respective aligned spans
        :param neg_candidates: list of negative candidate indices
        :param supporting_evidence: list of (context, positions)
        :return:
        '''
        assert is_query or context_query.supporting_evidence is None, "Supporting evidence cannot have supporting evidence!"
        if self._batch_idx >= self._batch_size:
            self._change_batch_size(max(self._batch_size*2, self._batch_idx))
        self._ctxt[self._batch_idx][:len(context_query.context)] = context_query.context
        self._len[self._batch_idx] = len(context_query.context)

        batch_idx = self._batch_idx
        self._batch_idx += 1
        for i, q in enumerate(context_query.queries):
            self._s.append(q.start)
            self._e.append(q.end)
            self._span_ctxt.append(batch_idx)
            self._answer_in.append(q.answer)
            cands = [q.answer]
            if q.neg_candidates is not None and q.neg_candidates is not None:
                cands.extend(q.neg_candidates)
            self._answer_cands.append(cands)
            self._query_part.append(0 if is_query else 1)

        if is_query:
            ### add query specific supports ###
            for i, q in enumerate(context_query.queries):
                if q.supporting_evidence is not None and self._max_queries > 0:
                    for qs in q.supporting_evidence:
                        start_idx = self._support_idx
                        if qs.context is None:
                            #supporting context is the same as query context, only add corresponding positions
                            for q in qs.queries:
                                self._s.append(q.start)
                                self._e.append(q.end)
                                self._span_ctxt.append(batch_idx)
                                self._answer_in.append(q.answer)
                                self._answer_cands.append([q.answer])
                                self._query_part.append(1)
                                self.supporting_qa.append((q.context, q.start, q.end, q.answer))
                        else:
                            self._add_example(qs, is_query=False)
                        self._support_idx += len(qs.queries)
                        # align queries with support idxs
                        self._support.extend(list(range(start_idx, self._support_idx)))
                        self._queries.extend([self._query_idx] * len(qs.queries))
                self._query_idx += 1

            ### add context specific support to all queries of this context ###
            if context_query.supporting_evidence is not None and self._max_queries > 0:
                for qs in context_query.supporting_evidence:
                    start_idx = self._support_idx
                    if qs.context is None:
                        for q in qs.queries:
                            self._s.append(q.start)
                            self._e.append(q.end)
                            self._span_ctxt.append(batch_idx)
                            self._answer_in.append(q.answer)
                            self._answer_cands.append([q.answer])
                            self._query_part.append(1)
                            self.supporting_qa.append((q.context, q.start, q.end, q.answer))
                    else:
                        self._add_example(qs, is_query=False)
                    self._support_idx += len(qs.queries)
                    # this evidence supports all queries in this context
                    for i, _ in enumerate(context_query.queries):
                        # align queries with support idxs
                        self._support.extend(list(range(start_idx, self._support_idx)))
                        self._queries.extend([self._query_idx - len(context_query.queries) + i] * len(qs.queries))
        else:
            for i, q in enumerate(context_query.queries):
                self.supporting_qa.append((q.context, q.start, q.end, q.answer))


    def _finish_adding_examples(self):
        max_cands = max((len(x) for x in self._answer_cands))
        cand_mask = []
        for i in range(len(self._answer_cands)):
            l = len(self._answer_cands[i])
            if self._query_part[i] == 0: # if this is a query (and not supporting evidence)
                mask = [0] * l
            for _ in range(max_cands - l):
                self._answer_cands[i].append(self._answer_cands[i][0])  # dummy
                if self._query_part[i] == 0:
                    mask.append(-1e6)  # this is added to scores, serves basically as a bias mask to exclude dummy negative candidates
            if self._query_part[i] == 0:
                cand_mask.append(mask)

        if self._batch_idx < self._batch_size:
            self._feed_dict[self._context] = self._ctxt[:self._batch_idx]
            self._feed_dict[self._length] = self._len[:self._batch_idx]
        else:
            self._feed_dict[self._context] = self._ctxt
            self._feed_dict[self._length] = self._len
        self._feed_dict[self._starts] = self._s
        self._feed_dict[self._ends] = self._e
        self._feed_dict[self._span_context] = self._span_ctxt
        self._feed_dict[self._answer_input] = self._answer_in
        self._feed_dict[self._answer_candidates] = self._answer_cands
        self._feed_dict[self._candidate_mask] = cand_mask
        self._feed_dict[self._query_ids] = self._queries
        self._feed_dict[self._support_ids] = self._support
        self._feed_dict[self._query_partition] = self._query_part

    def _get_feed_dict(self):
        return self._feed_dict

    def score_examples(self, sess, queries):
        i = j = 0
        num_queries = functools.reduce(lambda a,x: a+len(x.queries), queries, 0)
        max_neg_candidates = 0
        for context_queries in queries:
            for q in context_queries.queries:
                max_neg_candidates = max(max_neg_candidates, len(q.neg_candidates))
        result = np.zeros([num_queries, max_neg_candidates+1])
        while i < len(queries):
            batch_size = min(self._batch_size, len(queries)-i)
            self._start_adding_examples()
            num_batch_queries = 0
            for batch_idx in range(batch_size):
                context_query = queries[batch_idx + i]
                num_batch_queries += len(context_query.queries)
                self._add_example(context_query)
            self._finish_adding_examples()
            num_cands = len(self._answer_cands[0])
            result[j:j+num_batch_queries, 0:num_cands] = sess.run(self._scores_with_negs, feed_dict=self._get_feed_dict())
            i += batch_size
            j += num_batch_queries

        return result

    def step(self, sess, queries, mode="update"):
        '''
        :param sess:
        :param contexts: list of contexts, a context itself is a list of symbol indices
        :param positions: list of list of positions aligned with contexts, where a position determines a point
        of interest or query within a context
        :param neg_candidates: negative candidates for each position (list of list of list)
        :param supporting_evidence: (context, positions) tuples for each context =>
        list of (list of contexts, list of positions) or list of None, if no supporting evidence
        :param mode:
        :return:
        '''
        assert self._is_train, "model has to be created in training mode!"
        i = 0
        while i < len(queries):
            batch_size = min(self._batch_size, len(queries)-i)
            self._start_adding_examples()
            num_batch_queries = 0
            for batch_idx in range(batch_size):
                context_query = queries[batch_idx + i]
                num_batch_queries += len(context_query.queries)
                self._add_example(context_query)
            self._finish_adding_examples()
            i += batch_size
            if mode == "loss":
                return sess.run(self._loss, feed_dict=self._get_feed_dict())
            else:
                return sess.run([self._loss, self._update], feed_dict=self._get_feed_dict())[0]


def test_model():

    model = QAModel(10, 4, 5, 5, max_queries=2)
    # 3 contexts (of length 3) with queries at 2/1/2 (totaling 5) positions
    # and respective negative candidates for each position
    contexts =       [[0, 1, 2]       , [1, 2, 0], [0, 2, 1]]  # 4 => placeholder for prediction position

    queries = [ContextQueries(contexts[0], [ContextQuery(contexts[0], 0,1,0,[2,1]),
                                            ContextQuery(contexts[0], 2,3,2,[0,1])]),
               ContextQueries(contexts[1], [ContextQuery(contexts[1], 1,2,2,[0,1])]),
               ContextQueries(contexts[2], [ContextQuery(contexts[2], 1,2,2,[0,2]),
                                            ContextQuery(contexts[2], 2,3,1,[0,1])])]

    queries = [ContextQueries(contexts[0], [ContextQuery(contexts[0], 0,1,0,[2,1]),
                                            ContextQuery(contexts[0], 2,3,2,[0,1])], queries),
               ContextQueries(contexts[1], [ContextQuery(contexts[1], 1,2,2,[0,1], queries)]),
               ContextQueries(contexts[2], [ContextQuery(contexts[2], 1,2,1,[0,2], queries),
                                            ContextQuery(contexts[2], 2,3,2,[0,1], queries)])]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(model.num_queries.assign(1))
        print("Test update ...")
        for i in range(10):
            print("Loss: %.3f" %
                  model.step(sess, queries)[0])
        print("Test scoring ...")
        print(model.score_examples(sess, queries))
        print("Done")


if __name__ == '__main__':
    test_model()


"""
Test update ...
Loss: 1.012
Loss: 1.002
Loss: 0.990
Loss: 0.973
Loss: 0.948
Loss: 0.912
Loss: 0.865
Loss: 0.807
Loss: 0.736
Loss: 0.648
Test scoring....
[[  9.91435409e-01  -4.19906378e-01   5.15302122e-02]
 [  1.89884555e+00  -3.02041292e-01  -1.25379610e+00]
 [  3.59776109e-01  -1.48627639e-01  -9.99999625e+05]
 [ -2.41864566e-02  -1.41035974e-01   1.46159694e-01]
 [  7.77729750e-01  -2.62276053e-01  -5.17708778e-01]]
Done
"""
