import random


class BatchSampler:

    def __init__(self, kb, batch_size, which_set="train", max_contexts=-1):
        self.kb = kb
        self.batch_size = batch_size
        self.which_set = which_set
        self.num_contexts = self.kb.num_contexts(which_set)
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
        contexts, starts, ends, answers, neg_candidates, supporting_evidence = [], [], [], [], [], []
        splitter = self.kb.id("||")
        for i in range(self.batch_size):
            ctxt = self.kb.context(self.which_set, self.todo[i])
            k = len(ctxt)-1
            while ctxt[k] != splitter:
                k -= 1
            # < k: supporting evidence; >k: query
            start, end = self.kb.spans(self.which_set, self.todo[i])
            answer = self.kb.answers(self.which_set, self.todo[i])
            contexts.append(ctxt)
            starts.append([start[-1]])  # span start
            ends.append([end[-1]])  # span end
            answers.append([answer[-1]])

            start = [p for p in start if p < k]
            end = end[:len(start)]
            answer = answer[:len(start)]
            neg_cands = list(set(answer))
            # negative candidates are all entities within supporting evidence that are not the answer
            neg_candidates.append([neg_cands])
            # points of interest in supporting evidence
            supporting_evidence.append([(None, start, end, answer)])
        
        self.todo = self.todo[self.batch_size:]
        self.count += 1
        return (contexts, starts, ends, answers, neg_candidates, supporting_evidence)

    def get_epoch(self):
        return self.count / self.epoch_size
