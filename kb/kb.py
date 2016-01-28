#  Copy from uclmr

# coding=utf-8
# Knowledge Base Representation
# Tim Rocktaeschel, Guillaume Bouchard

import random
import pandas as pd


class KB:
    """
     KB represents a knowledge base of facts of varying arity
     >>> kb = KB()
     >>> kb.add_train("r1", "e1", "e2")
     >>> kb.is_true("r1", "e1", "e2")
     True
     Anything can be used to represent symbols
     >>> kb.add_train("r2", ("e1", "e3"))
     >>> kb.is_true("r2", ("e1", "e3"))
     True
     >>> kb.add_train("r2", "e1", "e3")
     >>> kb.add_train("r3", "e1", "e2", "e3")
     >>> kb.add_train("r4", ("e4", "e5"), "e6")
     >>> kb.add_train("r5", "e4")
     Any fact can be queried
     >>> kb.is_true("r1", "e1", "e2", "e4", "e5", "e6")
     False
     >>> kb.get_facts("e1", 1)
     [(('r1', 'e1', 'e2'), True, 'train'), (('r2', 'e1', 'e3'), True, 'train'), (('r3', 'e1', 'e2', 'e3'), True, 'train')]
     Adding the same fact twice does not add it
     >>> kb.add_train("r1", "e1", "e2")
     >>> len(kb.get_facts("e1", 1)) == 3
     True
     >>> kb.get_facts("unk_rel", 1)
     []
     >>> kb.get_facts("unk_ent", 6)
     []
     >>> kb.dim_size(0)
     5
     >>> kb.dim_size(1)
     4
     >>> kb.dim_size(3)
     1
    >>> [x for x in kb.get_all_facts_of_arity(1)]
    [(('r2', ('e1', 'e3')), True, 'train'), (('r5', 'e4'), True, 'train')]
    >>> [x for x in kb.get_all_facts_of_arity(2)]
    [(('r1', 'e1', 'e2'), True, 'train'), (('r2', 'e1', 'e3'), True, 'train'), (('r4', ('e4', 'e5'), 'e6'), True, 'train')]
     >>> sorted(list(kb.get_symbols(0)))
     ['r1', 'r2', 'r3', 'r4', 'r5']
     >>> sorted(list(kb.get_symbols(2)))
     ['e2', 'e3', 'e6']
     >>> kb.get_vocab(0)
     ['r1', 'r2', 'r3', 'r4', 'r5']
     >>> kb.get_vocab(1)
     ['e1', ('e1', 'e3'), ('e4', 'e5'), 'e4']
     >>> kb.get_vocab(2)
     ['e2', 'e3', 'e6']
     >>> random.seed(0)
     >>> kb.sample_neg("r1", 0, 2) not in kb.get_all_facts()
     True
     """

    random.seed(0)

    def __init__(self):
        # holds all known facts for every arity
        self.__facts = {}
        # holds all facts independent of arity
        self.__all_facts = set()
        # holds set of all symbols in every dimension
        self.__symbols = list()
        # holds list of all symbols in every dimension
        self.__vocab = list()
        # holds mappings of symbols to indices in every dimension
        self.__ids = list()
        # holds known facts for symbols in every dimension
        self.__maps = list()
        # caches number of dimensions since len(...) is slow
        self.__dims = list()
        # global mapping of symbols to indices independent from dimension
        self.__global_ids = {}

        self.__formulae = {}

    def __add_to_facts(self, fact):
        arity = len(fact[0]) - 1

        if arity not in self.__facts:
            self.__facts[arity] = list()
        self.__facts[arity].append(fact)
        self.__all_facts.add(fact)

    def __add_word(self, word):
        if word not in self.__global_ids:
            self.__global_ids[word] = len(self.__global_ids)

    def __add_to_symbols(self, key, dim):
        if len(self.__symbols) <= dim:
            self.__symbols.append(set())
        self.__symbols[dim].add(key)

        words = key
        if isinstance(words, basestring):
            words = [key]
        for word in words:
            self.__add_word(word)

    def __add_to_vocab(self, key, dim):
        if len(self.__vocab) <= dim:
            self.__vocab.append(list())
            self.__ids.append({})
            self.__dims.append(0)
        if len(self.__symbols) <= dim or key not in self.__symbols[dim]:
            self.__ids[dim][key] = len(self.__vocab[dim])
            self.__vocab[dim].append(key)
            self.__dims[dim] += 1

    def __add_to_maps(self, key, dim, fact):
        if len(self.__maps) <= dim:
            self.__maps.append({key: list()})
        if key in self.__maps[dim]:
            self.__maps[dim][key].append(fact)
        else:
            self.__maps[dim].update({key: [fact]})

    def get_all_facts_of_arity(self, arity, typ="train"):
        if arity not in self.__facts:
            return set()
        else:
            return filter(lambda x: x[2] == typ, self.__facts[arity])

    def get_all_facts(self):
        return self.__all_facts

    def add(self, truth, typ, *keys):
        assert isinstance(truth, bool)
        if not self.contains_fact(truth, typ, *keys):
            fact = (keys, truth, typ)
            self.__add_to_facts(fact)
            for dim in range(len(keys)):
                key = keys[dim]
                self.__add_to_vocab(key, dim)
                self.__add_to_symbols(key, dim)
                self.__add_to_maps(key, dim, fact)

    def contains_fact(self, truth, typ, *keys):
        return (keys, truth, typ) in self.get_all_facts()

    def add_train(self, *keys):
        self.add(True, "train", *keys)

    def get_facts(self, key, dim):
        result = list()
        if len(self.__maps) > dim:
            if key in self.__maps[dim]:
                result = self.__maps[dim][key]
        return result

    def is_true(self, *keys):
        arity = len(keys) - 1
        if arity not in self.__facts:
            return False
        else:
            return (keys, True, "train") in self.__facts[arity]

    def dim_size(self, dim):
        if dim >= len(self.__dims):
            return 0
        else:
            return self.__dims[dim]

    # @profile
    def sample_neg(self, key, dim, arity, tries=100):
        cell = list()
        for i in range(0, arity + 1):
            symbol_ix = random.randint(0, self.dim_size(i) - 1)
            symbol = self.__vocab[i][symbol_ix]
            cell.append(symbol)
        cell[dim] = key
        cell = tuple(cell)

        if tries == 0:
            print "Warning, couldn't sample negative fact for", key, "in dim", dim
            return cell, False, "train"
        elif (cell, True, "train") in self.__facts[arity]:
            return self.sample_neg(key, dim, arity, tries - 1)
        else:
            return cell, False, "train"

    def get_vocab(self, dim):
        return self.__vocab[dim]

    def get_symbols(self, dim):
        return self.__symbols[dim]

    def to_data_frame(self):
        data = {}
        for key1 in self.__vocab[0]:
            row = list()
            for key2 in self.__vocab[1]:
                if ((key1, key2), True, "train") in self.__facts[1]:
                    row.append(1.0)
                else:
                    row.append(0.0)
            data[key1] = row
        df = pd.DataFrame(data, index=self.__vocab[1])
        return df

    def get_id(self, key, dim):
        return self.__ids[dim][key]

    def get_ids(self, *keys):
        ids = list()
        for dim in range(len(keys)):
            ids.append(self.get_id(keys[dim], dim))
        return ids

    def get_global_id(self, symbol):
        return self.__global_ids[symbol]

    def get_global_ids(self, *symbols):
        ids = list()
        for symbol in symbols:
            # fixme
            if not isinstance(symbol, basestring):
                for s in symbol:
                    ids.append(self.get_global_id(s))
            else:
                ids.append(self.get_global_id(symbol))
        return ids

    def num_global_ids(self):
        return len(self.__global_ids)

    def get_key(self, id, dim):
        return self.__vocab[dim][id]

    def get_keys(self, *ids):
        keys = list()
        for dim in range(len(ids)):
            keys.append(self.get_key(ids[dim], dim))
        return keys

    def add_formulae(self, label, formulae):
        self.__formulae[label] = formulae

    def get_formulae(self, label):
        return self.__formulae[label]

    def apply_formulae(self):
        done = False
        while not done:
            done = True

            for (body, head) in self.get_formulae("inv"):
                facts = self.get_facts(body, 0)
                for ((rel, (e1, e2)), target, typ) in facts:
                    contained = self.contains_fact(target, 'train', head, (e2, e1))
                    if target and typ == 'train' and not contained:
                        self.add_train(head, (e2, e1))
                        done = False

            for arity in range(1, 4):
                if arity in self.get_formulae("impl"):
                    for (body, head) in self.get_formulae("impl")[arity]:
                        facts = self.get_facts(body, 0)
                        for ((rel, args), target, typ) in facts:
                            contained = self.contains_fact(target, 'train', head, args)
                            if target and typ == 'train' and not contained:
                                self.add_train(head, args)
                                done = False

                if arity in self.get_formulae("impl_conj"):
                    for (body1, body2, head) in self.get_formulae("impl_conj")[arity]:
                        facts1 = self.get_facts(body1, 0)
                        facts2 = self.get_facts(body2, 0)
                        facts = [x for x in facts1 for y in facts2
                                 if x[0][1] == y[0][1] and x[1] == y[1] and x[2] == y[2]]
                        for ((rel, args), target, typ) in facts:
                            contained = self.contains_fact(target, 'train', head, args)
                            if target and typ == 'train' and not contained:
                                self.add_train(head, args)
                                done = False

            for (body1, body2, head) in self.get_formulae("trans"):
                facts1 = self.get_facts(body1, 0)
                facts2 = self.get_facts(body2, 0)
                facts = [((head, (e1, e4)), typ1, target1) for ((rel1, (e1, e2)), typ1, target1) in facts1 for
                         ((rel2, (e3, e4)), typ2, target2) in facts2
                         if e2 == e3 and typ1 == typ2 and target1 == target2]
                #print facts1
                #print facts2
                #print "trans", facts
                for ((rel, (e1, e2)), target, typ) in facts:
                    contained = self.contains_fact(target, 'train', rel, (e1, e2))
                    if target and typ == 'train' and not contained:
                        self.add_train(rel, (e1, e2))
                        done = False


class SampleKB:
    def __init__(self, num_relations, num_entities,
                 arities=[0.0, 1.0, 0.0],
                 arg_densities=[1.0, 0.1, 0.01],
                 fact_prob=0.2,
                 num_inv=0,
                 num_impl=[0, 0, 0],
                 num_impl_conj=[0, 0, 0],
                 num_trans=0,
                 negated_head_prob=0.0,
                 seed=0):
        """
        :param num_relations:
        :param num_entities: number of distinct entities to generate
        :param arities:
        :param arg_densities: fraction of entity combinations that are observed
        :param fact_prob:
        :param num_inv:
        :param num_impl:
        :param num_impl_conj:
        :param num_trans:
        :param negated_head_prob:
        :param seed:
        :return:
        """
        random.seed(seed)
        self.kb = KB()

        num_relations_per_arity = map(lambda x: int(x * num_relations), arities)
        entities = list(map(lambda x: "e" + str(x), range(1, num_entities)))
        entities_per_arity = {1: entities[0:int(num_entities * arg_densities[0])],
                              2: random.sample([(x, y) for x in entities for y in entities if not x == y],
                                               int(num_entities ** 2 * arg_densities[1])),
                              3: random.sample([(x, y, z) for x in entities for y in entities for z in entities
                                                if not x == y and not y == z and not z == x],
                                               int(num_entities ** 3 * arg_densities[2]))}

        relations_per_arity = {}
        for arity in range(1, len(num_relations_per_arity) + 1):
            for i in range(1, num_relations_per_arity[arity - 1] + 1):
                rel = ""
                if arity == 1:
                    rel = "u"
                elif arity == 2:
                    rel = "b"
                else:
                    rel = "t"
                rel += str(i)

                if not arity in relations_per_arity:
                    relations_per_arity[arity] = list()
                relations_per_arity[arity].append(rel)

                for j in range(0, len(entities)):
                    if fact_prob > random.uniform(0, 1.0):
                        args = random.sample(entities_per_arity[arity], 1)[0]
                        self.kb.add_train(rel, args)

        # sampling reversed: r1(X,Y) => r2(Y,X)
        inverse = random.sample([(x, y) for x in relations_per_arity[2] for y in relations_per_arity[2]
                                 if not x == y], num_inv)
        self.kb.add_formulae("inv", inverse)

        # sampling implications:
        # r1(X) => r2(X)
        # r1(X,Y) => r2(X,Y)
        implications_per_arity = {}
        for arity in range(1, len(num_relations_per_arity) + 1):
            if arity in relations_per_arity:
                implications_per_arity[arity] = \
                    random.sample([(x, y) for x in relations_per_arity[arity] for y in relations_per_arity[arity]
                                   if not x == y], num_impl[arity - 1])
        self.kb.add_formulae("impl", implications_per_arity)

        # sampling implications with conjunction in body:
        # r1(X,Y) ^ r2(X,Y) => r3(X,Y)
        # r1(X) ^ r2(X) => r3(X)
        implications_with_conjunction_per_arity = {}
        for arity in range(1, len(num_relations_per_arity) + 1):
            if arity in relations_per_arity and len(relations_per_arity[arity]) >= 3:
                implications_with_conjunction_per_arity[arity] = \
                    random.sample([(x, y, z) for x in relations_per_arity[arity]
                                   for y in relations_per_arity[arity]
                                   for z in relations_per_arity[arity]
                                   if not x == y and not y == z and not z == x],
                                  num_impl_conj[arity - 1])
        self.kb.add_formulae("impl_conj", implications_with_conjunction_per_arity)

        # sampling transitivities:
        # r1(X,Y) ^ r2(Y,Z) => r3(X,Z)
        transitivities = random.sample([(x, y, z) for x in relations_per_arity[2]
                                        for y in relations_per_arity[2]
                                        for z in relations_per_arity[2]
                                        if not x == y and not y == z and not z == x],
                                       num_trans)
        self.kb.add_formulae("trans", transitivities)

        # todo: sampling negation (also applies to all heads of formulae above):
        # r1 => !r2


    def get_kb(self):
        return self.kb


# implements an iterator over training examples while sampling negative examples
class BatchNegSampler:
    # todo: pass sampling function as argument
    def __init__(self, kb, arity, batch_size):
        self.kb = kb
        self.batch_size = batch_size
        self.facts = list(self.kb.get_all_facts_of_arity(arity))
        self.todo_facts = list(self.facts)
        self.num_facts = len(self.facts)
        self.__reset()

    # @profile
    def __reset(self):
        self.todo_facts = list(self.facts)
        random.shuffle(self.todo_facts)
        self.count = 0

    def __iter__(self):
        return self

    def next(self):
        if self.count >= self.num_facts:
            self.__reset()
            raise StopIteration
        return self.get_batch()

    # todo: generalize this towards sampling in different dimensions
    # @profile
    def get_batch(self, neg_per_pos=1):
        if self.count >= self.num_facts:
            self.__reset()
        num_pos = self.batch_size / (1 + neg_per_pos)
        pos = self.todo_facts[0:num_pos]
        self.count += self.batch_size
        self.todo_facts = self.todo_facts[num_pos::]
        neg = list()
        for fact in pos:
            neg.append(self.kb.sample_neg(fact[0][0], 0, 1))
        return self.__tensorize(pos + neg)

    # @profile
    def __tensorize(self, batch):
        rows = list()
        cols = list()
        targets = list()

        for i in range(len(batch)):
            example = batch[i]
            rows.append(self.kb.get_id(example[0][0], 0))
            cols.append(self.kb.get_id(example[0][1], 1))
            if example[1]:
                targets.append(1)
            else:
                targets.append(0)

        return rows, cols, targets

    def get_epoch(self):
        return self.count / float(self.num_facts)


class Seq2Fact2SeqBatchSampler:
    # todo: pass sampling function as argument
    def __init__(self, kb, arity, batch_size):
        self.kb = kb
        self.batch_size = batch_size
        self.facts = filter(lambda x: not x[0][0][0].startswith("REL$"), list(self.kb.get_all_facts_of_arity(arity)))
        self.todo_facts = list(self.facts)
        self.num_facts = len(self.facts)
        self.__reset()

    # @profile
    def __reset(self):
        self.todo_facts = list(self.facts)
        random.shuffle(self.todo_facts)
        self.count = 0

    def __iter__(self):
        return self

    def next(self):
        if self.count >= self.num_facts:
            self.__reset()
            raise StopIteration
        return self.get_batch()

    # todo: generalize this towards sampling in different dimensions
    # @profile
    def get_batch(self):
        if self.count >= self.num_facts:
            self.__reset()
        pos = self.todo_facts[0:self.batch_size]
        self.count += self.batch_size
        self.todo_facts = self.todo_facts[self.batch_size::]
        return self.__tensorize(pos)

    # @profile
    def __tensorize(self, batch):
        rows = list()
        cols = list()

        for i in range(len(batch)):
            example = batch[i]
            rows.append(self.kb.get_global_ids(example[0][0]))
            cols.append(self.kb.get_id(example[0][1], 1))

        return rows, cols, rows

    def get_epoch(self):
        return self.count / float(self.num_facts)


def test_sampling():
    sampleKB = SampleKB(10, 5, 0.5).get_kb()
    print sampleKB.to_data_frame()

    sampler = BatchNegSampler(sampleKB, 1, 4)
    for i in range(3):
        for batch in sampler:
            print batch
            print sampler.get_epoch()
        print "epoch finished"


def test_ids():
    sampleKB = SampleKB(10, 5, 0.5).get_kb()
    print sampleKB.to_data_frame()
    print sampleKB.get_id("r0", 0)
    print sampleKB.get_id("r1", 0)
    print sampleKB.get_id("e0", 1)
    print sampleKB.get_id("e1", 1)
    print sampleKB.get_ids("r0", "e1")
    print sampleKB.get_keys(0, 4)


def test_train_cells():
    sampleKB = SampleKB(10, 5, 0.5).get_kb()
    print sampleKB.to_data_frame()
    print sampleKB.get_all_facts_of_arity(1)


def test_pairs():
    kb = KB()
    kb.add_train("r4", ("e4", "e5"), "e6")
    kb.add_train("r1", ("e4", "e5"), "e6")
    kb.add_train("r2", ("e4", "e5"), "e6")
    print kb.get_facts(("e4", "e5"), 1)


def test_two_kbs():
    kb1 = KB()
    kb1.add_train("blah", "keks")
    kb2 = KB()
    kb2.add_train("blubs", "hui")
    print kb2.get_all_facts()


def test_global_ids():
    kb = KB()
    kb.add_train("blah", "keks")
    kb.add_train(("e1", "e2"), "r")
    print kb.get_global_id("keks")
    print kb.get_global_id("e1")
    print kb.get_global_id("r")
    print kb.get_global_ids("e1", "e2", "r")

    # test_kb()
    # test_sampling()
    # test_train_cells()
    # test_pairs()
    # test_two_kbs()
    # test_global_ids()


if __name__ == "__main__":
    import doctest

    doctest.testmod()