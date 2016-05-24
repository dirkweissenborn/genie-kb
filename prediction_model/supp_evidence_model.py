from prediction_model.models import *


class SupportingEvidenceModel(AbstractKBPredictionModel):

    def __init__(self, model, learning_rate=1e-2, is_train=True, which_sets=None, num_consecutive_queries=1):
        self._model = model
        self._which_sets = which_sets
        self._num_consecutive_queries = num_consecutive_queries
        AbstractKBPredictionModel.__init__(self, model._kb, model._size, model._batch_size, is_train, learning_rate)

    def _comp_f(self):
        with vs.variable_scope("supporting"):
            query = self._model.query
            query, supp_queries = tf.dynamic_partition(query, self._query_partition, 2)
            num_queries = tf.shape(query)[0]

            self.evidence_weights = []
            answer = query
            current_query = query
            for i in range(self._num_consecutive_queries):
                with vs.variable_scope("evidence_%d"%i):
                    aligned_queries = tf.gather(current_query, self._support_ids)  # align query with respective supp_queries
                    e_scores = tf_util.batch_dot(aligned_queries, supp_queries)
                    self.evidence_weights.append(e_scores)
                    e_scores = tf.exp(e_scores -
                                      tf.tile(tf.reduce_max(e_scores, [0], keep_dims=True),
                                              tf.shape(e_scores)))
                    norm = tf.unsorted_segment_sum(e_scores, self._support_ids, num_queries) + 0.00001 # for zero norms

                    e_weights = tf.tile(tf.reshape(e_scores, [-1, 1]), [1, self._size])
                    norm = tf.tile(tf.reshape(norm, [-1, 1]), [1, self._size])

                    _, supp_answers = tf.dynamic_partition(self._model._y_input, self._query_partition, 2)
                    supp_answers = tf.nn.embedding_lookup(self._model.candidates, supp_answers)
                    weighted_answers = tf.unsorted_segment_sum(e_weights * supp_answers, self._support_ids, num_queries) / norm
                    weighted_queries = tf.unsorted_segment_sum(e_weights * supp_queries, self._support_ids, num_queries) / norm

                    answer_weight = tf.contrib.layers.fully_connected(weighted_queries * query, 1,
                                                                      activation_fn=tf.sigmoid, weight_init=None,
                                                                      bias_init=tf.constant_initializer(-1))
                    answer_weight = tf.tile(answer_weight, [1, self._size])

                    answer = weighted_answers * answer_weight + (1-answer_weight) * answer

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
            #    answer = tf.reduce_sum(answer_weights * answer_candidates, [1])

        tf.get_variable_scope().reuse_variables()
        return answer

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
        return self._model.name()

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


