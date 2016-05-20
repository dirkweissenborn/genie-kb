from prediction_model.models import *


class SupportingEvidenceModel(AbstractKBPredictionModel):

    def __init__(self, model, learning_rate=1e-2, is_train=True, which_sets=None):
        self._model = model
        self._which_sets = which_sets
        AbstractKBPredictionModel.__init__(self, model._kb, model._size, model._batch_size, is_train, learning_rate)

    def _comp_f(self):
        tf.get_variable_scope().reuse_variables()
        queries = self._model.query
        queries, supp_evidence = tf.dynamic_partition(queries, self._query_partition, 2)
        num_queries = tf.shape(queries)[0]
        aligned_queries = tf.gather(queries, self._support_ids)  # align queries with respective supp_evidence
        evidence_scores = tf_util.batch_dot(aligned_queries, supp_evidence)
        #TODO maybe stabilize
        e_scores = tf.exp(evidence_scores)
        norm = tf.unsorted_segment_sum(e_scores, self._support_ids, num_queries) + 0.00001 # for zero norms

        evidence_weights = tf.tile(tf.reshape(e_scores, [-1, 1]), [1,self._size])
        norm = tf.tile(tf.reshape(norm, [-1, 1]), [1, self._size])

        _, support_answers = tf.dynamic_partition(self._model._y_input, self._query_partition, 2)
        support_answers = tf.nn.embedding_lookup(self._model.candidates, support_answers)
        weighted_answers = evidence_weights * support_answers
        weighted_answers = tf.unsorted_segment_sum(weighted_answers, self._support_ids, num_queries) / norm

        return queries + weighted_answers

    def _init_inputs(self):
        self.arg_vocab = self._model.arg_vocab
        self._query_partition = tf.placeholder(tf.int32, [None])
        self._support_ids = tf.placeholder(tf.int32, [None])

        query_idx = tf.where(tf.equal(self._query_partition, 0))
        self._y_candidates = tf.squeeze(tf.gather(self._model._y_candidates, query_idx), [1])
        self._y_input = tf.reshape(tf.gather(self._model._y_input, query_idx), [-1])

        self._feed_dict = self._model._feed_dict
        self._tuple_rels_lookup = dict()
        self._tuple_rels_lookup_inv = dict()
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
                if t_inv not in self._tuple_rels_lookup_inv:
                    self._tuple_rels_lookup_inv[t_inv] = [r_i]
                else:
                    self._tuple_rels_lookup_inv[t_inv].append(r_i)

    def name(self):
        return self._model.name()

    def _start_adding_triples(self):
        self._inner_batch_idx = 0
        self._query_part = []
        self._support = []

    def _add_triple_and_negs_to_input(self, triple, neg_candidates, batch_idx, is_inv):
        self._model._add_triple_and_negs_to_input(triple, neg_candidates, self._inner_batch_idx, is_inv)
        self._query_part.append(0)
        self._inner_batch_idx += 1
        (rel, x, y) = triple
        # get supporting evidence
        r_i = self._kb.get_id(rel, 0)
        x_i = self._model.arg_vocab[y] if is_inv else self._model.arg_vocab[x]
        support_size = 0
        for y2 in [x if is_inv else y] + neg_candidates:
            y_i = self._model.arg_vocab[y2]
            rels = self._tuple_rels_lookup.get((x_i, y_i), []) + self._tuple_rels_lookup_inv.get((x_i, y_i), [])

            if rels:
                for i in range(len(rels)):
                    if rels[i] != r_i:
                        supporting_triple = (self._kb.get_key(r_i, 0), y2 if is_inv else x, x if is_inv else y2)
                        self._model._add_triple_to_input(supporting_triple, self._inner_batch_idx, is_inv)
                        self._query_part.append(1)
                        support_size += 1
                        self._inner_batch_idx += 1
        self._support.extend([batch_idx] * support_size)


    def _add_triple_to_input(self, triple, batch_idx, is_inv):
        self._model._add_triple_to_input(triple, self._inner_batch_idx, is_inv)
        self._inner_batch_idx += 1
        (rel, x, y) = triple
        # get supporting evidence
        r_i = self._kb.get_id(rel, 0)
        x_i = self._model.arg_vocab[y] if is_inv else self._model.arg_vocab[x]
        y_i = self._model.arg_vocab[x] if is_inv else self._model.arg_vocab[y]
        rels = self._tuple_rels_lookup.get((x_i,y_i), []) + self._tuple_rels_lookup_inv.get((x_i,y_i), [])
        if rels:
            for i in range(len(rels)):
                if rels[i] != r_i:
                    self._model._add_triple_to_input((self._kb.get_key(r_i, 0), x, y), self._inner_batch_idx, is_inv)
                    self._inner_batch_idx += 1


    def _finish_adding_triples(self, batch_size, is_inv):
        self._model._finish_adding_triples(self._inner_batch_idx, is_inv)
        self._feed_dict[self._query_partition] = self._query_part
        self._feed_dict[self._support_ids] = self._support


