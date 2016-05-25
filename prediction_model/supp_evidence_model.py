from prediction_model.models import *
import model as m

class SupportingEvidenceModel(AbstractKBPredictionModel):

    def __init__(self, model, learning_rate=1e-2, is_train=True, which_sets=None, num_consecutive_queries=1):
        self._model = model
        self._which_sets = which_sets
        self._num_consecutive_queries = num_consecutive_queries
        self._kb = model._kb
        self._size = model._size
        self._batch_size = model._batch_size
        self._is_train = is_train
        self._init = m.default_init()

        with vs.variable_scope(self.name(), initializer=self._init):
            self._init_inputs()
            self.query = self._comp_f()

            self.candidates = self._model.candidates

            lookup_individual = tf.nn.embedding_lookup(self.candidates, self._y_input)
            self._score = tf_util.batch_dot(lookup_individual, self.query)
            lookup = tf.nn.embedding_lookup(self.candidates, self._y_candidates)
            self._scores_with_negs = tf.squeeze(tf.batch_matmul(lookup, tf.expand_dims(self.query, [2])), [2])

            if is_train:
                self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="lr")
                self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.9)
                self.global_step = tf.Variable(0, trainable=False, name="step")

                self.opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0)

                current_batch_size = tf.gather(tf.shape(self._scores_with_negs), [0])
                labels = tf.constant([0], tf.int64)
                labels = tf.tile(labels, current_batch_size)
                loss = math_ops.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(self._scores_with_negs, labels))

                train_params = tf.trainable_variables() # [v for v in tf.trainable_variables() if "supporting" in v.name]
                self.training_weight = tf.Variable(1.0, trainable=False, name="training_weight")

                self._loss = loss / math_ops.cast(current_batch_size, dtypes.float32)
                in_params = self._input_params()
                if in_params is None:
                    self._grads = tf.gradients(self._loss, train_params, self.training_weight)
                else:
                    self._grads = tf.gradients(self._loss, train_params + in_params, self.training_weight)
                    self._input_grads = self._grads[len(train_params):]
                if len(train_params) > 0:
                    self._update = self.opt.apply_gradients(zip(self._grads[:len(train_params)], train_params),
                                                            global_step=self.global_step)
                else:
                    self._update = tf.assign_add(self.global_step, 1)

        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)

    def _comp_f(self):
        with vs.variable_scope("supporting"):
            query = self._model.query
            query, supp_queries = tf.dynamic_partition(query, self._query_partition, 2)
            num_queries = tf.shape(query)[0]

            _, supp_answers = tf.dynamic_partition(self._model._y_input, self._query_partition, 2)
            supp_answers = tf.tanh(tf.nn.embedding_lookup(self._model.candidates, supp_answers))

            self.evidence_weights = []
            current_answer = query
            current_query = query
            for i in range(self._num_consecutive_queries):
                with vs.variable_scope("evidence_%d"%i):
                    aligned_queries = tf.gather(current_query, self._support_ids)  # align query with respective supp_queries

                    scores = tf_util.batch_dot(aligned_queries, supp_queries)
                    self.evidence_weights.append(scores)
                    e_scores = tf.exp(scores - tf.tile(tf.reduce_max(scores, [0], keep_dims=True), tf.shape(scores)))
                    norm = tf.unsorted_segment_sum(e_scores, self._support_ids, num_queries) + 0.00001 # for zero norms
                    norm = tf.tile(tf.reshape(norm, [-1, 1]), [1, self._size])

                    e_weights = tf.tile(tf.reshape(e_scores, [-1, 1]), [1, self._size])

                    weighted_answers = tf.unsorted_segment_sum(e_weights * supp_answers, self._support_ids, num_queries) / norm
                    weighted_queries = tf.unsorted_segment_sum(e_weights * supp_queries, self._support_ids, num_queries) / norm

                    summed_scores = tf.unsorted_segment_sum(tf.reshape(e_scores * scores, [-1,1]),
                                                            self._support_ids, num_queries) / norm
                    #answer_weight = tf.contrib.layers.fully_connected(tf.concat(1, [weighted_queries,query]), self._size,
                    #                                                  activation_fn=tf.nn.relu, weight_init=None,
                    #                                                  bias_init=None)
                    answer_weight = tf.contrib.layers.fully_connected(summed_scores, 1,
                                                                      activation_fn=tf.nn.sigmoid, weight_init=None,
                                                                      bias_init=tf.constant_initializer(-1.0))

                    current_answer = weighted_answers * answer_weight + current_answer

                    if i < self._num_consecutive_queries - 1:
                        vs.get_variable_scope()._reuse = \
                            any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())

                        c = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_answers]), self._size,
                                                              activation_fn=tf.tanh, weight_init=None, bias_init=None)

                        gate = tf.contrib.layers.fully_connected(tf.concat(1, [current_query, weighted_queries]),
                                                                 self._size,
                                                                 activation_fn=tf.sigmoid, weight_init=None,
                                                                 bias_init=tf.constant_initializer(0))
                        current_query = gate * current_query + (1-gate) * c

            #with vs.variable_scope("selection"):
            #    vs.get_variable_scope()._reuse = \
            #        any(vs.get_variable_scope().name in v.name for v in tf.trainable_variables())

            #    q_inter = tf.contrib.layers.fully_connected(query, self._size,
            #                                                activation_fn=None, weight_init=None, bias_init=None)

            #    with vs.variable_scope("to_select"):
            #        W = tf.get_variable("W_answer_interaction", [self._size, self._size])
            #        inter = []
            #        for other_query in queries:
            #            inter.append(tf.nn.relu(tf.matmul(other_query, W) + q_inter))

             #   inter = tf.reshape(tf.concat(1, inter), [-1, self._size])

            #    scores = tf.contrib.layers.fully_connected(inter, 1)
            #    answer_weights = tf.nn.softmax(tf.reshape(scores, [-1, 1+self._num_consecutive_queries]))
            #    answer_weights = tf.tile(tf.expand_dims(answer_weights, [2]), [1, 1, self._size])

            #    answer_candidates = tf.reshape(tf.concat(1, answers), [-1, 1+self._num_consecutive_queries, self._size])
            #    current_answer = tf.reduce_sum(answer_weights * answer_candidates, [1])

        tf.get_variable_scope().reuse_variables()
        return current_answer

    def _init_inputs(self):
        self.arg_vocab = self._model.arg_vocab
        self._query_partition = tf.placeholder(tf.int32, [None])
        self._support_ids = tf.placeholder(tf.int32, [None])

        query_idx = tf.where(tf.equal(self._query_partition, 0))
        self._y_candidates = tf.squeeze(tf.gather(self._model._y_candidates, query_idx), [1])
        self._y_input = tf.reshape(tf.gather(self._model._y_input, query_idx), [-1])

        self._feed_dict = self._model._feed_dict
        self._tuple_rels_lookup = dict()
        #self._tuple_rels_lookup_inv = dict()
        self._num_relations = len(self._kb.get_symbols(0))

        for (rel, subj, obj), _, typ in self._kb.get_all_facts():
            if self._which_sets is None or typ in self._which_sets:
                s_i = self._model.arg_vocab[subj]
                o_i = self._model.arg_vocab[obj]
                r_i = self._kb.get_id(rel, 0)
                t = (s_i, o_i)
                if t not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t] = [r_i]
                else:
                    self._tuple_rels_lookup[t].append(r_i)
                # also add inverse
                t_inv = (o_i, s_i)
                if t_inv not in self._tuple_rels_lookup:
                    self._tuple_rels_lookup[t_inv] = [r_i]
                else:
                    self._tuple_rels_lookup[t_inv].append(r_i)

    def name(self):
        return self._model.name() + "__supporting_%d" % self._num_consecutive_queries

    def _start_adding_triples(self):
        self._inner_batch_idx = 0
        self._query_part = []
        self._support = []
        self.supporting_triples = []

    def _add_triple_and_negs_to_input(self, triple, neg_candidates, batch_idx, is_inv):
        self._model._add_triple_and_negs_to_input(triple, neg_candidates, self._inner_batch_idx, is_inv)
        self._query_part.append(0)
        self._inner_batch_idx += 1
        (rel, x, y) = triple
        # get supporting evidence
        r_i = self._kb.get_id(rel, 0)
        x_i = self._model.arg_vocab[y] if is_inv else self._model.arg_vocab[x]
        for y2 in [x if is_inv else y] + neg_candidates:
            y_i = self._model.arg_vocab[y2]
            rels = self._tuple_rels_lookup.get((x_i, y_i), [])# + self._tuple_rels_lookup.get((y_i, x_i), [])

            if rels:
                for i in range(len(rels)):
                    if rels[i] != r_i:
                        supporting_triple = (self._kb.get_key(rels[i], 0), y2 if is_inv else x, x if is_inv else y2)
                        self.supporting_triples.append(supporting_triple)
                        self._model._add_triple_to_input(supporting_triple, self._inner_batch_idx, is_inv)
                        self._query_part.append(1)
                        self._support.append(batch_idx)
                        self._inner_batch_idx += 1


    def _add_triple_to_input(self, triple, batch_idx, is_inv):
        self._model._add_triple_to_input(triple, self._inner_batch_idx, is_inv)
        self._query_part.append(0)
        self._inner_batch_idx += 1
        (rel, x, y) = triple
        # get supporting evidence
        r_i = self._kb.get_id(rel, 0)
        x_i = self._model.arg_vocab[y] if is_inv else self._model.arg_vocab[x]
        y_i = self._model.arg_vocab[x] if is_inv else self._model.arg_vocab[y]
        rels = self._tuple_rels_lookup.get((x_i,y_i), [])# + self._tuple_rels_lookup.get((y_i,x_i), [])
        if rels:
            for i in range(len(rels)):
                if rels[i] != r_i:
                    supporting_triple = (self._kb.get_key(rels[i], 0), y if is_inv else x, x if is_inv else y)
                    self.supporting_triples.append(supporting_triple)
                    self._model._add_triple_to_input(supporting_triple, self._inner_batch_idx, is_inv)
                    self._query_part.append(1)
                    self._support.append(batch_idx)
                    self._inner_batch_idx += 1


    def _finish_adding_triples(self, batch_size, is_inv):
        self._model._finish_adding_triples(self._inner_batch_idx, is_inv)
        self._feed_dict[self._query_partition] = self._query_part
        self._feed_dict[self._support_ids] = self._support



