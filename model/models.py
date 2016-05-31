
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


class QAModel:

    def __init__(self, size, batch_size, vocab_size, max_length, is_train=True, learning_rate=1e-2,
                 composition="GRU", num_consecutive_queries=0, devices=None):
        self._vocab_size = vocab_size
        self._max_length = max_length
        self._size = size
        self._batch_size = batch_size
        self._is_train = is_train
        self._init = model.default_init()
        self._composition = composition
        self._num_queries = num_consecutive_queries
        self._device0 = devices[0] if devices is not None else "/cpu:0"
        self._device1 = devices[1 % len(devices)] if devices is not None else "/cpu:0"
        self._device2 = devices[2 % len(devices)] if devices is not None else "/cpu:0"
        self._device3 = devices[3 % len(devices)] if devices is not None else "/cpu:0"
        with tf.device(self._device0):
            with vs.variable_scope(self.name(), initializer=self._init):
                self._init_inputs()
                with tf.device("/cpu:0"):
                    self.candidates = tf.get_variable("E_candidate", [vocab_size, self._size])
                    answer, _ = tf.dynamic_partition(self._answer_input, self._query_partition, 2)
                    lookup_individual = tf.tanh(tf.nn.embedding_lookup(self.candidates, answer))
                    cands,_ = tf.dynamic_partition(self._answer_candidates, self._query_partition, 2)
                    lookup = tf.tanh(tf.nn.embedding_lookup(self.candidates, cands))

                self.query = self._comp_f()
                self.query = self._supporting_evidence(self.query)
                self._score = tf_util.batch_dot(lookup_individual, self.query)
                self._scores_with_negs = tf.squeeze(tf.batch_matmul(lookup, tf.expand_dims(self.query, [2])), [2])
                self._scores_with_negs += self._candidate_mask  # number of negative candidates can vary for each example

                if is_train:
                    self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.9)
                    self.global_step = tf.Variable(0, trainable=False, name="step")

                    self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0)

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
            embed = tf.get_variable("E_words", [self._vocab_size, self._size])
            embedded = tf.nn.embedding_lookup(embed, self._context)
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
            query = tf.contrib.layers.fully_connected(tf.concat(1, [out_fw, out_bw]), self._size,
                                                      activation_fn=None, weight_init=None)
            query = tf.tanh(query)#, float(self._size))
        # add supporting evidence to this query
        return query

    def _supporting_evidence(self, query):
        if self._num_queries == 0:
            return query
        else:
            with vs.variable_scope("supporting"):
                query, supp_queries = tf.dynamic_partition(query, self._query_partition, 2)
                num_queries = tf.shape(query)[0]
                
                with tf.device("/cpu:0"):
                  _, supp_answers = tf.dynamic_partition(self._answer_input, self._query_partition, 2)
                  supp_answers = tf.tanh(tf.nn.embedding_lookup(self.candidates, supp_answers))

                self.evidence_weights = []
                current_answer = query
                current_query = query
                for i in range(self._num_queries):
                    with vs.variable_scope("evidence"):
                        vs.get_variable_scope()._reuse = \
                                any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())
                        aligned_queries = tf.gather(current_query, self._support_ids)  # align query with respective supp_queries

                        scores = tf_util.batch_dot(aligned_queries, supp_queries)
                        self.evidence_weights.append(scores)
                        e_scores = tf.exp(scores - tf.reduce_max(scores, [0], keep_dims=True))
                        norm = tf.unsorted_segment_sum(e_scores, self._support_ids, num_queries) + 0.00001 # for zero norms
                        norm = tf.reshape(norm, [-1, 1])
                        # this is basically the dot product between query and weighted supp_queries
                        summed_scores = tf.unsorted_segment_sum(tf.reshape(e_scores * scores, [-1,1]),
                                                                self._support_ids, num_queries) / norm
                        norm = tf.tile(norm, [1, self._size])
                        scores = tf.tile(tf.reshape(scores, [-1, 1]), [1, self._size])
                        weighted_answers = tf.unsorted_segment_sum(scores * supp_answers, self._support_ids, num_queries)
                        #answer_weight = tf.contrib.layers.fully_connected(tf.concat(1, [weighted_queries,query]), self._size,
                        #                                                  activation_fn=tf.nn.relu, weight_init=None,
                        #                                                  bias_init=None)
                        answer_weight = tf.contrib.layers.fully_connected(summed_scores, 1,
                                                                          activation_fn=tf.nn.sigmoid,
                                                                          weight_init=tf.constant_initializer(0.0),
                                                                          bias_init=tf.constant_initializer(0.0))

                        current_answer = weighted_answers * answer_weight + (1.0-answer_weight) * current_answer

                        if i < self._num_queries - 1:
                            e_scores = tf.tile(tf.reshape(e_scores, [-1, 1]), [1, self._size])
                            weighted_queries = tf.unsorted_segment_sum(e_scores * supp_queries, self._support_ids, num_queries) / norm

                            c = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_answers]), self._size,
                                                                  activation_fn=tf.tanh, weight_init=None, bias_init=None)

                            gate = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_queries]),
                                                                     self._size,
                                                                     activation_fn=tf.sigmoid, weight_init=None,
                                                                     bias_init=tf.constant_initializer(0))
                            current_query = gate * current_query + (1-gate) * c

            tf.get_variable_scope().reuse_variables()
            return current_answer

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
        # partition of actual queries and supporting evidence
        self._query_partition = tf.placeholder(tf.int32, [None], "query_partition")
        # mapping from supporting evidence to respective queries
        self._support_ids = tf.placeholder(tf.int64, shape=[None], name="supp_evidence")

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
        self._answer_cands = []
        self._answer_in = []
        self._s = []
        self._e = []
        self._span_ctxt = []
        #supporting evidence
        self._query_part = []
        self._support = []
        self.supporting_qa = []

    def _add_example_and_negs(self, context, starts, ends, answers,
                              neg_candidates, supporting_evidence=None, is_query=True):
        '''
        :param context: list of symbol indices
        :param starts: list of span starts to answer for
        :param ends: list of span ends
        :param answers: answer for respective aligned spans
        :param neg_candidates: list of negative candidate indices
        :param supporting_evidence: list of (context, positions)
        :return:
        '''
        assert is_query or supporting_evidence is None, "Supporting evidence cannot have supporting evidence!"
        if self._batch_idx >= self._batch_size:
            self._change_batch_size(max(self._batch_size*2, self._batch_idx))
        self._ctxt[self._batch_idx][:len(context)] = context
        self._len[self._batch_idx] = len(context)
        self._s.extend(starts)
        self._e.extend(ends)

        for i, pos in enumerate(starts):
            self._span_ctxt.append(self._batch_idx)
            self._answer_in.append(answers[i])
            cands = [answers[i]]
            if neg_candidates is not None and i < len(neg_candidates) and neg_candidates[i] is not None:
                cands.extend(neg_candidates[i])
            self._answer_cands.append(cands)
            self._query_part.append(0 if is_query else 1)

        query_batch_idx = self._batch_idx
        self._batch_idx += 1

        if supporting_evidence is not None and self._num_queries > 0:
            for supp_context, supp_start, supp_end, supp_answers in supporting_evidence:
                if supp_context is None:
                    #supporting context is the same as query context, only add corresponding positions
                    self._s.extend(supp_start)
                    self._e.extend(supp_end)
                    for i, start in enumerate(supp_start):
                        self._span_ctxt.append(query_batch_idx)
                        self._answer_in.append(supp_answers[i])
                        self._answer_cands.append([supp_answers[i]])
                        self._query_part.append(1)
                        self.supporting_qa.append((context, start, supp_end[i], supp_answers[i]))
                else:
                    self._add_example_and_negs(supp_context, supp_start, supp_end, supp_answers, None, is_query=False)
                self._support.extend([self._query_idx] * len(supp_start))

        if is_query:
            self._query_idx += 1
        else:
            for i, start in enumerate(starts):
                self.supporting_qa.append((context, start, ends[i], answers[i]))

    def _add_example(self, context, starts, ends, answers, supporting_evidence=None, is_query=True):
        self._add_example_and_negs(context, starts, ends, answers, None, supporting_evidence, is_query)

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
        self._feed_dict[self._support_ids] = self._support
        self._feed_dict[self._query_partition] = self._query_part

    def _get_feed_dict(self):
        return self._feed_dict

    def score_examples(self, sess, contexts, starts, ends, answers, supporting_evidence=None):
        return self.score_examples_with_negs(sess, contexts, starts, ends, answers, None, supporting_evidence=supporting_evidence)

    def score_examples_with_negs(self, sess, contexts, starts, ends, answers, neg_candidates, supporting_evidence=None):
        i = j = 0
        num_queries = functools.reduce(lambda a,x: a+len(x), starts, 0)
        max_neg_candidates = 0
        if neg_candidates:
            for cands in neg_candidates:
                for cs in cands:
                    max_neg_candidates = max(max_neg_candidates, len(cs))
        result = np.zeros([num_queries, max_neg_candidates+1])
        while i < len(contexts):
            batch_size = min(self._batch_size, len(contexts)-i)
            self._start_adding_examples()
            num_batch_queries = 0
            for batch_idx in range(batch_size):
                num_batch_queries += len(starts[i + batch_idx])
                if neg_candidates:
                    self._add_example_and_negs(contexts[i + batch_idx],
                                               starts[i + batch_idx],
                                               ends[i + batch_idx],
                                               answers[i + batch_idx],
                                               neg_candidates[i+batch_idx],
                                               None if supporting_evidence is None else supporting_evidence[i+batch_idx])
                else:
                    self._add_example(contexts[i + batch_idx],
                                      starts[i + batch_idx],
                                      ends[i + batch_idx],
                                      answers[i + batch_idx],
                                      None if supporting_evidence is None else supporting_evidence[i+batch_idx])
            self._finish_adding_examples()
            num_cands = len(self._answer_cands[0])
            result[j:j+num_batch_queries, 0:num_cands] = sess.run(self._scores_with_negs, feed_dict=self._get_feed_dict())
            i += batch_size
            j += num_batch_queries

        return result

    def step(self, sess, contexts, starts, ends, answers, neg_candidates, supporting_evidence=None, mode="update"):
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
        while i < len(contexts):
            batch_size = min(self._batch_size, len(contexts)-i)
            self._start_adding_examples()
            for batch_idx in range(batch_size):
                self._add_example_and_negs(contexts[i + batch_idx],
                                           starts[i + batch_idx],
                                           ends[i + batch_idx],
                                           answers[i + batch_idx],
                                           neg_candidates[i+batch_idx],
                                           None if supporting_evidence is None else supporting_evidence[i+batch_idx])
            self._finish_adding_examples()
            i += batch_size
            if mode == "loss":
                return sess.run(self._loss, feed_dict=self._get_feed_dict())
            else:
                return sess.run([self._loss, self._update], feed_dict=self._get_feed_dict())[0]


def test_model():
    model = QAModel(10, 4, 5, 5, num_consecutive_queries=1)
    # 3 contexts (of length 3) with queries at 2/1/2 (totaling 5) positions
    # and respective negative candidates for each position
    contexts =       [[4, 1, 4]       , [1, 4, 0], [0, 4, 4]]  # 4 => placeholder for prediction position
    starts   =       [[0     , 2]     , [1]      , [1     , 2]]
    ends     =       [[1     , 3]     , [2]      , [2     , 3]]
    answers  =       [[0     , 2]     , [2]      , [1     , 2]]
    neg_candidates = [[[2, 1], [0, 1]], [[0, 1]] , [[0, 2], [0, 1]]]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print("Test update ...")
        for i in range(10):
            print("Loss: %.3f" %
                  model.step(sess, contexts, starts, ends, answers, neg_candidates, [list(zip(contexts, starts, ends, answers))] * 5)[0])
        print("Test scoring (without supporting evidence) ...")
        print(model.score_examples(sess, contexts, starts, ends, answers))
        print("Test scoring with negative examples (and supporting evidence)...")
        print(model.score_examples_with_negs(sess, contexts, starts, ends, answers, neg_candidates, [list(zip(contexts, starts, ends, answers))] * 5))
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
Test scoring (without supporting evidence) ...
[ 0.818488    1.35567915  0.90820128 -0.02418646  0.77772975]
Test scoring with negative examples (and supporting evidence)...
[[  9.91435409e-01  -4.19906378e-01   5.15302122e-02]
 [  1.89884555e+00  -3.02041292e-01  -1.25379610e+00]
 [  3.59776109e-01  -1.48627639e-01  -9.99999625e+05]
 [ -2.41864566e-02  -1.41035974e-01   1.46159694e-01]
 [  7.77729750e-01  -2.62276053e-01  -5.17708778e-01]]
Done
"""
