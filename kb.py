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
        # lists compatible arguments for each arg position foreach relation
        self.__compatible_args = dict()

        self.__formulae = {}

    def add_compatible_arg(self, key, dim, rel_key, rel_dim=0):
        '''
        :param dim: arg dimension
        :param key: arg key
        :param rel_key: key of relation
        :param rel_dim: dim of relation (usually 0)
        :return:
        '''
        if rel_key in self.__symbols[rel_dim] and key in self.__symbols[dim]:
            if dim not in self.__compatible_args:
                self.__compatible_args[dim] = [set() for _ in self.__symbols[rel_dim]]
            args = self.__compatible_args[dim]
            rel_id = self.get_id(rel_key, rel_dim)
            args[rel_id].add(key)

    def compatible_args_of(self, dim, rel_key, rel_dim=0):
        if len(self.__compatible_args) == 0:
            # no constraints, return everything
            return self.__symbols[dim]
        else:
            rel_id = self.get_id(rel_key, rel_dim)
            return self.__compatible_args[dim][rel_id]

    def __add_to_facts(self, fact):
        arity = len(fact[0]) - 1

        if arity not in self.__facts:
            self.__facts[arity] = list()
        self.__facts[arity].append(fact)
        self.__all_facts.add(fact)

    def __add_to_symbols(self, key, dim):
        if len(self.__symbols) <= dim:
            self.__symbols.append(set())
        self.__symbols[dim].add(key)

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


def subsample_kb(kb, num_entities):
    new_kb = KB()
    subj_samples = set(random.sample(kb.get_symbols(1), num_entities))
    for (rel, subj, obj), truth, typ in kb.get_all_facts():
        if subj in subj_samples:
            new_kb.add(truth, typ, rel, subj, obj)
    return new_kb
