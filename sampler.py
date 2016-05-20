import random
from multiprocessing.dummy import Pool


class BatchNegTypeSampler:

    def __init__(self, kb, pos_per_batch, neg_per_pos=200, which_set="train", type_constraint=True):
        self.kb = kb
        self.pos_per_batch = pos_per_batch
        self.neg_per_pos = neg_per_pos
        self.type_constraint = type_constraint
        self.facts = [f[0] for f in self.kb.get_all_facts() if f[2] == which_set]
        self.num_facts = len(self.facts)
        self.epoch_size = self.num_facts // self.pos_per_batch
        self.reset()
        self.__pool = Pool()

        self._rngs = [random.Random(random.randint(0, 1000)) for _ in range(2*pos_per_batch)]

        self._objs = list(self.kb.get_symbols(2))
        self._subjs = list(self.kb.get_symbols(1))

    def init_types(self):
        # add types to concepts
        concept_types = dict()
        self.rel_args = dict()
        self.rel_types = dict()

        for rel, subj, obj in self.facts:
            subj_role = rel + "_s"
            if subj_role not in self.rel_args:
                self.rel_args[subj_role] = set()
            self.rel_args[subj_role].add(subj)

            if subj not in concept_types:
                concept_types[subj] = set()
            concept_types[subj].add(subj_role)

            obj_role = rel + "_o"
            if obj_role not in self.rel_args:
                self.rel_args[obj_role] = set()
            self.rel_args[obj_role].add(subj)

            if obj not in concept_types:
                concept_types[obj] = set()
            concept_types[obj].add(obj_role)

        # count types for positions in relation
        rel_types = dict()
        for rel, subj, obj in self.facts:
            subj_role = rel + "_s"
            obj_role = rel + "_o"
            if subj_role not in rel_types:
                rel_types[subj_role] = dict()
                rel_types[obj_role] = dict()
            subj_ts = rel_types[subj_role]
            obj_ts = rel_types[obj_role]
            for t in concept_types[subj]:
                if t not in subj_ts:
                    subj_ts[t] = 0
                subj_ts[t] += 1
            for t in concept_types[obj]:
                if t not in obj_ts:
                    obj_ts[t] = 0
                obj_ts[t] += 1

        # sort types for relations by count
        for rel_role, types in rel_types.items():
            if rel_role not in self.rel_types:
                self.rel_types[rel_role] = []  # distinction between subj and obj types
            self.rel_types[rel_role].extend(map(lambda x: x[0], sorted(types.items(), key=lambda x:-x[1])))

    # @profile
    def reset(self):
        self.todo_facts = list(range(self.num_facts))
        random.shuffle(self.todo_facts)
        self.todo_facts = self.todo_facts[:-(self.num_facts % self.pos_per_batch)]
        self.count = 0

    def end_of_epoch(self):
        return self.count == self.epoch_size

    def __iter__(self):
        return self

    def next(self):
        if self.count >= self.epoch_size:
            raise StopIteration
        return self.get_batch()

    def __get_neg_examples(self, triple, position, rng):
        (rel, subj, obj) = triple
        dim = 2 if position == "obj" else 1
        #allowed = self.kb.get_symbols(dim)
        disallowed = obj if position == "obj" else subj

        if self.type_constraint:
            #sample by type
            #neg_candidates = set()
            #typs = self.rel_types[rel+"_o"] if position == "obj" else self.rel_types[rel+"_s"]

            # add negative neg_candidates until there are enough negative neg_candidates
            #i = 0
            #while i < len(typs) and len(neg_candidates) < self.neg_per_pos:
            #    typ = typs[i]
            #    i += 1
            #    for c in self.rel_args[typ]:
            #        if c != disallowed and c in allowed:
            #            neg_candidates.add(c)
            neg_candidates = list(self.kb.compatible_args_of(dim, rel))
        else:  # sample from all candidates
            neg_candidates = self._objs if position == "obj" else self._subjs

        neg_examples = list()

        # sampling code is optimized; no use of remove for lists (since it is O(n))
        if position == "obj":
            last = len(neg_candidates)-1  # index of last good candidate
            for _ in range(self.neg_per_pos):
                x = None
                while not x or x == disallowed or self.kb.contains_fact(True, "train", rel, subj, x):
                    i = rng.randint(0, last)
                    x = neg_candidates[i]
                    if neg_candidates is not self._objs:  # do not change self._objs, accidental doubles are very rare
                        # remove candidate efficiently from candidates
                        if i != last:
                            neg_candidates[i] = neg_candidates[last]  # copy last good candidate to position i
                        last -= 1
                        if last == -1:
                            neg_candidates = self._objs  # fallback
                            last = len(neg_candidates) - 1
                neg_examples.append(x)
        else:
            last = len(neg_candidates)-1  # index of last good candidate
            for _ in range(self.neg_per_pos):
                x = None
                while not x or x == disallowed or self.kb.contains_fact(True, "train", rel, x, obj):
                    i = rng.randint(0, last)
                    x = neg_candidates[i]
                    # remove candidate efficiently from candidates
                    if neg_candidates is not self._subjs:  # do not change self._subjs
                        if i != last:
                            neg_candidates[i] = neg_candidates[last]  # copy last good candidate to position i
                        last -= 1
                        if last == -1:
                            neg_candidates = self._subjs # fallback
                            last = len(neg_candidates) - 1
                neg_examples.append(x)

        return neg_examples

    # @profile
    def get_batch(self, position="obj"):
        pos_idx = self.todo_facts[0:self.pos_per_batch]
        self.count += 1
        self.todo_facts = self.todo_facts[self.pos_per_batch::]
        pos = [self.facts[i] for i in pos_idx]
        negs = None
        if position == "subj":
            negs = self.__pool.map(lambda i: self.__get_neg_examples(pos[i], "subj", self._rngs[i]),
                                   (i for i in range(self.pos_per_batch)))

        if position == "obj":
            negs = self.__pool.map(lambda i: self.__get_neg_examples(pos[i], "obj", self._rngs[i]),
                                   (i for i in range(self.pos_per_batch)))

        return pos, negs, position == "subj"

    def get_batch_async(self, position="obj"):
        return self.__pool.apply_async(self.get_batch, (position,))

    def get_epoch(self):
        return self.count / self.epoch_size
