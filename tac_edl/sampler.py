import random
from model.query import *
import functools

class BatchSampler:

    def __init__(self, fact_kb, batch_size, datasets=["TAC_ET/2014_train"]):
        self.fact_kb = fact_kb
        self.kb = fact_kb.kb
        self.batch_size = batch_size
        self.datasets = datasets
        self.num_contexts = functools.reduce(lambda acc, ds: self.kb.num_contexts(ds) + acc, datasets, 0)
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

    def __index_and_dataset(self, i):
        offset = 0
        for ds in self.datasets:
            if offset + self.kb.num_contexts(ds) > i:
                return i-offset, ds
            else:
                offset += self.kb.num_contexts(ds)

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
        for i in range(self.batch_size):
            idx, ds = self.__index_and_dataset(self.todo[i])
            ctxt = self.kb.context(idx, ds)
            starts, ends = self.kb.spans(idx, ds)
            answers = self.kb.answers(idx, ds)
            # use fact_kb entity ids as answers, not the whole vocabulary
            answers = [self.fact_kb.id(self.kb.vocab[x]) for x in answers]
            #supp_queries = []
            #for i in range(len(starts)):
                #supp_queries.append(ContextQuery(None, starts[i], ends[i], answers[i], None))
            #supp_queries = ContextQueries(None, supp_queries)
            neg_cands = [] # list((c for c in self.candidate_searcher.search() if c != answers[-1]))
            queries = []
            for i in range(len(starts)):
                queries.append(ContextQuery(ctxt, starts[i], ends[i], answers[i], neg_cands))#, supporting_evidence=[supp_queries])
            batch_queries.append(ContextQueries(ctxt, queries, source=ds))
        self.todo = self.todo[self.batch_size:]
        self.count += 1

        return batch_queries

    def get_epoch(self):
        return self.count / self.epoch_size
