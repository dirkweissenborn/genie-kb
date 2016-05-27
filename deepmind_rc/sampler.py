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
        :return: list of (context, query_positions, negative_candidates, supporting_facts)
        '''
        if self.end_of_epoch():
            print("WARNING: End of epoch reached in sampler. Resetting automatically.")
            self.reset()
        batch = []
        for i in range(self.batch_size):
            context = self.kb.context(self.which_set, self.todo[i])
            positions = self.kb.positions(self.which_set, self.todo[i])
            neg_cands = [context[x] for x in positions if context[x] != context[positions[-1]]]
            batch.append((context, [positions[-1]], neg_cands, [(None, positions[:-1])]))

        return batch

    def get_epoch(self):
        return self.count / self.epoch_size
