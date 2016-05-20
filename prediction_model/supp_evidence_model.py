from prediction_model.models import *


class SupportingEvidenceModel(AbstractKBPredictionModel):

    def __init__(self, model, kb, size, batch_size, is_train=True, learning_rate=1e-2, which_sets=None):
        self._model = model
        self._which_sets = which_sets
        AbstractKBPredictionModel.__init__(self, kb, size, batch_size, is_train, learning_rate)


    def _init_inputs(self):
        self._query_partition = tf.placeholder(tf.int64, [None])
        self._support_size = tf.placeholder(tf.int64, [None])
        self._query_part = []
        self._support_sz = []

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

    def _start_adding_triples(self):
        self._inner_batch_idx = 0

    def _add_triple_and_negs_to_input(self, triple, neg_candidates, batch_idx, is_inv):
        self._model._add_triple_and_negs_to_input(triple, self._inner_batch_idx, is_inv)
        self._query_part[self._inner_batch_idx] = 0
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
                        self._query_part[self._inner_batch_idx] = 1
                        support_size += 1
                        self._inner_batch_idx += 1
        self._support_sz.append(support_size)

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

    def _comp_f_bw(self):
        super()._comp_f_bw()

    def _comp_f_fw(self):
        super()._comp_f_fw()


