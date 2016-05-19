import os
from kb import KB
import numpy as np

def load_fb15k(dir, with_text=True, split_text=False):
    train_file = os.path.join(dir, "train.txt")
    test_file = os.path.join(dir, "test.txt")
    text_file = os.path.join(dir, "text_emnlp.txt")
    valid_file = os.path.join(dir, "valid.txt")

    kb = KB()

    _load_triples(train_file, kb)
    _load_triples(valid_file, kb, typ="valid")
    _load_triples(test_file, kb, typ="test")
    if with_text:
        if split_text:
            _load_dep_paths(text_file, kb, typ="train_text")
        else:
            _load_triples(text_file, kb, typ="train_text")

    return kb


def load_fb15k_type_constraints(kb, dir):
    subj_file = os.path.join(dir, "ecompatiblenesbj.txt")
    obj_file = os.path.join(dir, "ecompatibleneobj.txt")

    load_type_constraints(kb, subj_file, 1)
    load_type_constraints(kb, obj_file, 2)


def load_type_constraints(kb, fn, arg_dim, rel_dim=0):
    with open(fn, 'r') as f:
        for l in f:
            split = l.strip().split("\t")
            for i in range(1, len(split)):
                kb.add_compatible_arg(split[0], arg_dim, split[i], rel_dim)


def _load_triples(fn, kb, typ="train"):
    triples = []
    with open(fn) as f:
        for l in f:
            split = l.strip().split("\t")
            kb.add(True, typ, split[1], split[0], split[2])
    return triples


def split_relations(rel):
    if rel.endswith("_inv"):
        split = split_relations(rel[:-4])
        split.reverse()
        return split
    elif "[XXX]" in rel:
        dep_path_arr = []
        c = 0
        for i in range(len(rel)-2):
            if rel[i:i+3] == ":<-":
                if c > 0:  # do not keep [XXX]
                    dep_path_arr.append(rel[c:i])
                #dep_path_arr.append(":<-")
                c = i+3
            elif rel[i:i+2] == ":<":
                if c > 0:
                    dep_path_arr.append(rel[c:i])
                #dep_path_arr.append(":<")
                c = i+2
            elif rel[i:i+2] == ">:":
                if c > 0:
                    dep_path_arr.append(rel[c:i])
                c = i+2
        return dep_path_arr
    else:
        return [rel]  # rel.split("/")


def _load_dep_paths(fn, kb, typ="train"):
    with open(fn) as f:
        for l in f:
            [id1, dep_path, id2, ct] = l.strip().split("\t")
            dep_path_arr = []
            c = 0
            for i in range(len(dep_path)-2):
                if dep_path[i:i+3] == ":<-":
                    if c > 0:  # do not keep [XXX]
                        dep_path_arr.append(dep_path[c:i])
                    dep_path_arr.append(":<-")
                    c = i+3
                elif dep_path[i:i+2] == ":<":
                    if c > 0:
                        dep_path_arr.append(dep_path[c:i])
                    dep_path_arr.append(":<")
                    c = i+2
                elif dep_path[i:i+2] == ">:":
                    if c > 0:
                        dep_path_arr.append(dep_path[c:i])
                    c = i+2
            kb.add(True, typ, dep_path_arr, id1, id2)


''' deprecated

def _load_triples(fn, vocab, concept_vocab):
    triples = []
    with open(fn) as f:
        for l in f:
            [id1, relation, id2] = l.strip().split("\t")
            _update(id1, concept_vocab)
            _update(relation, vocab)
            _update(id2, concept_vocab)
            triple = (id1, relation, concept_vocab)
            if id1 != id2:
                triples.append(triple)
    return triples



def _load_dep_paths(fn, vocab, concept_vocab):
    triples = []
    with open(fn) as f:
        for l in f:
            [id1, dep_path, id2, ct] = l.strip().split("\t")
            dep_path_arr = []
            c = 0
            for i in range(len(dep_path)-2):
                if dep_path[i:i+3] == ":<-":
                    if c > 0:  # do not keep [XXX]
                        dep_path_arr.append(dep_path[c:i])
                    dep_path_arr.append(":<-")
                    c = i+3
                elif dep_path[i:i+2] == ":<":
                    if c > 0:
                        dep_path_arr.append(dep_path[c:i])
                    dep_path_arr.append(":<")
                    c = i+2
                elif dep_path[i:i+2] == ">:":
                    if c > 0:
                        dep_path_arr.append(dep_path[c:i])
                    c = i+2
            for e in dep_path_arr:
                _update(e, vocab)
            if id1 != id2:
                _update(id1, concept_vocab)
                _update(id2, concept_vocab)
                triple = (id1, dep_path_arr, id2)
                for i in range(int(ct)):
                    triples.append(triple)
    return triples


def _update(entry, vocab):
    v = vocab.get(entry)
    if entry not in vocab:
        v = len(vocab)
        vocab[entry] = v
'''