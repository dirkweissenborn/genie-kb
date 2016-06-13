import random
from model.query import *
import math

class BatchSampler:

    def __init__(self, kb, batch_size, which_set="train", max_contexts=-1, max_vocab=-1):
        self.kb = kb
        self.batch_size = batch_size
        self.which_set = which_set
        self.num_contexts = self.kb.num_contexts(which_set)
        self.max_vocab = max_vocab
        if max_contexts > 0:
            self.num_contexts = min(max_contexts, self.num_contexts)
        self.epoch_size = math.ceil(self.num_contexts / self.batch_size)
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
        splitter = self.kb.id("||")
        for i in range(min(self.batch_size, len(self.todo))):
            ctxt = self.kb.context(self.todo[i], self.which_set)
            if self.max_vocab > 0:
                ctxt = [min(self.max_vocab, i) for i in ctxt]
            k = len(ctxt)-1
            while ctxt[k] != splitter:
                k -= 1
            # < k: supporting evidence; >k: query
            # we switch start and end here, because entities are anonymized -> consider only outer context
            ends, starts = self.kb.spans(self.todo[i], self.which_set)
            answers = self.kb.answers(self.todo[i], self.which_set)
            #word vocab and answer vocab differ
            answer_words = [min(self.max_vocab, self.kb.answer_id_to_word_id(a))
                            if self.max_vocab > 0 else self.kb.answer_id_to_word_id(a) for a in answers]
            supp_queries = []
            for i in range(len(starts)):
                if starts[i] < k:
                    supp_queries.append(ContextQuery(None, starts[i], ends[i], answers[i], answer_words[i], None))
            supp_queries = ContextQueries(None, supp_queries)
            neg_cands = list(set((c for c in answers if c != answers[-1])))
            query = ContextQuery(ctxt, starts[-1], ends[-1], answers[-1], answer_words[-1], neg_cands, supporting_evidence=[supp_queries])
            batch_queries.append(ContextQueries(ctxt, [query]))
        self.todo = self.todo[self.batch_size:]
        self.count += 1
        return batch_queries

    def get_epoch(self):
        return self.count / self.epoch_size
