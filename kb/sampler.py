import random
from multiprocessing.dummy import Pool
import dill


class BatchNegTypeSampler:

    def __init__(self, kb, batch_size, neg_per_pos=200, which_set="train", type_constrained=True):
        self.kb = kb
        self.batch_size = batch_size
        self.neg_per_pos = neg_per_pos
        self.pos_per_batch = self.batch_size / (1 + self.neg_per_pos)
        self.type_constrained = type_constrained
        self.facts = [f[0] for f in self.kb.get_all_facts() if f[2] == which_set]
        self.num_facts = len(self.facts)
        self.epoch_size = self.num_facts / self.pos_per_batch
        self.reset()
        self.__pool = Pool(8)

        # we use sampling with type constraints
        if type_constrained:
            self.init_types()

    def init_types(self):
        # add types to concepts
        concept_types = dict()
        self.rel_subjects = dict()
        self.rel_objects = dict()
        self.rel_types = dict()

        for rel, subj, obj in self.facts:
            if rel not in self.rel_subjects:
                self.rel_subjects[rel] = set()
            self.rel_subjects[rel].add(subj)

            if subj not in concept_types:
                concept_types[subj] = set()
            concept_types[subj].add(rel)

            if rel not in self.rel_objects:
                self.rel_objects[rel] = set()
            self.rel_objects[rel].add(obj)

            if obj not in concept_types:
                concept_types[obj] = set()
            concept_types[obj].add(rel)

        # count types for positions in relation
        rel_types = dict()
        for rel, subj, obj in self.facts:
            if rel not in rel_types:
                rel_types[rel] = (dict(), dict())
            (subj_ts, obj_ts) = rel_types[rel]
            for t in concept_types[subj]:
                if t not in subj_ts:
                    subj_ts[t] = 0
                subj_ts[t] += 1
            for t in concept_types[obj]:
                if t not in obj_ts:
                    obj_ts[t] = 0
                obj_ts[t] += 1

        # sort types for relations by count
        for rel, types in rel_types.iteritems():
            if rel not in self.rel_types:
                self.rel_types[rel] = ([], [])  # distinction between subj and obj types
            self.rel_types[rel][0].extend(map(lambda x: x[0], sorted(types[0].items(), key=lambda x:-x[1])))
            self.rel_types[rel][1].extend(map(lambda x: x[0], sorted(types[1].items(), key=lambda x:-x[1])))

    # @profile
    def reset(self):
        self.todo_facts = list(xrange(self.num_facts))
        random.shuffle(self.todo_facts)
        self.todo_facts = self.todo_facts[:-(self.num_facts % self.pos_per_batch)]
        self.count = 0

    def end_of_epoch(self):
        return self.count == self.epoch_size

    def __iter__(self):
        return self

    def next(self):
        if self.count >= self.num_facts:
            self.reset()
            raise StopIteration
        return self.get_batch()

    def __get_neg_examples(self, triple, position):
        (rel, subj, obj) = triple
        allowed = self.kb.get_symbols(2) if position == "obj" else self.kb.get_symbols(1)
        disallowed = obj if position == "obj" else subj
        if self.type_constrained:
            #sample by type
            neg_candidates = set()
            typs = self.rel_types[rel][1 if position == "obj" else 0]

            # add negative neg_candidates until there are enough negative neg_candidates
            i = 0
            type_concepts = self.rel_objects if position == "obj" else self.rel_subjects
            while i < len(typs) and len(neg_candidates) < self.neg_per_pos:
                typ = typs[i]
                i += 1
                for c in type_concepts[typ]:
                    if c in allowed:
                        neg_candidates.add(c)

            while len(neg_candidates) < self.neg_per_pos:  # sample random
                cs = random.sample(allowed, self.neg_per_pos-len(neg_candidates))
                for c in cs:
                    if c in allowed and c != disallowed:
                        neg_candidates.add(c)

        else:  # sample from all candidates
            neg_candidates = allowed
        neg_candidates = list(neg_candidates)

        neg_triples = list()
        if position == "obj":
            for _ in xrange(self.neg_per_pos):
                x = None
                while not x or x == disallowed or self.kb.contains_fact(True, "train", rel, subj, x):
                    x = random.choice(neg_candidates)
                neg_triples.append((rel, subj, x))
        else:
            for _ in xrange(self.neg_per_pos):
                x = None
                while not x or x == disallowed or self.kb.contains_fact(True, "train", rel, x, obj):
                    x = random.choice(neg_candidates)
                neg_triples.append((rel, x, obj))

        return neg_triples

    # @profile
    def get_batch(self, position="obj"):
        if self.end_of_epoch():
            self.reset()
        pos_idx = self.todo_facts[0:self.pos_per_batch]
        self.count += 1
        self.todo_facts = self.todo_facts[self.pos_per_batch::]
        pos = [self.facts[i] for i in pos_idx]

        neg = self.__pool.map(lambda fact: self.__get_neg_examples(fact, position), pos)

        return pos, neg

    def get_epoch(self):
        return self.count / float(self.num_facts)
