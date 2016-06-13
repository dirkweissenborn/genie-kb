import random
from model.query import *
from cbt import *

class BatchSampler:

    def __init__(self, kb, batch_size, which_set="train", max_contexts=-1, max_vocab=-1):
        self.kb = kb
        self.batch_size = batch_size
        self.which_set = which_set
        self.num_contexts = self.kb.num_contexts(which_set)
        self.max_vocab = max_vocab
        if max_contexts > 0:
            self.num_contexts = min(max_contexts, self.num_contexts)
        self.epoch_size = self.num_contexts // self.batch_size
        self._rng = random.Random(73642)
        self.reset()

    def reset(self):
        self.todo = list(range(self.num_contexts))
        self._rng.shuffle(self.todo)
        if self.num_contexts % self.batch_size != 0:
            self.todo = self.todo[:-(self.num_contexts % self.batch_size)]
        self.count = 0

    def end_of_epoch(self):
        return self.count == self.epoch_size

    def __iter__(self):
        return self

    def next(self):
        if self.count >= self.epoch_size:
            raise StopIteration
        return self.get_batch()

    def get_batch(self):
        '''
        Note, the data in the kb is structured as follows: contexts= supporting_evidence + "||" + query.
        Thus, we have to split contexts.
        :return: list of (contexts, query_positions, negative_candidates, supporting_facts)
        '''
        if self.end_of_epoch():
            print("WARNING: End of epoch reached in sampler. Resetting automatically.")
            self.reset()

        batch_queries = []
        splitter = self.kb.id(answer_sep)
        for i in range(self.batch_size):
            ctxt = self.kb.context(self.todo[i], self.which_set)
            k = ctxt.index(splitter)
            if self.max_vocab > 0:
                ctxt = [min(self.max_vocab, i) for i in ctxt]
            # < k: supporting evidence; >k: query
            # we switch start and end here, because entities are anonymized -> consider only outer context
            ends, starts = self.kb.spans(self.todo[i], self.which_set)
            answers = self.kb.answers(self.todo[i], self.which_set)
            # in cbt all words are potential answers so use word vocab instead of answer vocab
            answers = [min(self.max_vocab, self.kb.answer_id_to_word_id(a))
                       if self.max_vocab > 0 else self.kb.answer_id_to_word_id(a) for a in answers]

            candidates = ctxt[k+1:]
            cand_set = set(candidates)
            ctxt = ctxt[:k]
            supp_queries = []
            for start, c in enumerate(ctxt):
                if c in cand_set:
                    supp_queries.append(ContextQuery(None, start+1, start, c, c, None))
            supp_queries = ContextQueries(None, supp_queries)
            neg_cands = [c for c in candidates if c != answers[-1]]
            query = ContextQuery(ctxt, starts[-1], ends[-1], answers[-1], answers[-1], neg_cands, supporting_evidence=[supp_queries])
            batch_queries.append(ContextQueries(ctxt, [query]))
        self.todo = self.todo[self.batch_size:]
        self.count += 1
        return batch_queries

    def get_epoch(self):
        return self.count / self.epoch_size
