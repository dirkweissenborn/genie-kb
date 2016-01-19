import os


def load_fb15k(dir, max_vocab=-1):
    train_file = os.path.join(dir, "train.txt")
    test_file = os.path.join(dir, "test.txt")
    text_file = os.path.join(dir, "text_emnlp.txt")
    valid_file = os.path.join(dir, "valid.txt")

    vocab = dict()
    concept_vocab = dict()

    train_kb_triples = _load_triples(train_file, vocab, concept_vocab)
    valid_kb_triples = _load_triples(valid_file, vocab, concept_vocab)
    text_triples = _load_dep_paths(text_file, vocab, concept_vocab)
    test_kb_triples = _load_triples(test_file, vocab, concept_vocab)

    if max_vocab >= 0:
        vocab = {k: v for k, v in vocab.iteritems() if v < max_vocab-1}  # -1 because there is the unknown token
        vocab["<unk>"] = len(vocab)

    return {"train": train_kb_triples,
            "text": text_triples,
            "valid": valid_kb_triples,
            "test": test_kb_triples
            }, vocab, concept_vocab


def _load_triples(fn, vocab, concept_vocab):
    triples = []
    with open(fn) as f:
        for l in f:
            [id1, relation, id2] = l.strip().split("\t")
            triple = (_get_or_else_update(id1, concept_vocab),
                      [_get_or_else_update(relation, vocab)],
                      _get_or_else_update(id2, concept_vocab))
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
            for i in xrange(len(dep_path)-2):
                if dep_path[i:i+3] == ":<-":
                    if c > 0:  # do not keep [XXX]
                        dep_path_arr.append(_get_or_else_update(dep_path[c:i], vocab))
                    dep_path_arr.append(_get_or_else_update(":<-", vocab))
                    c = i+3
                elif dep_path[i:i+2] == ":<":
                    if c > 0:
                        dep_path_arr.append(_get_or_else_update(dep_path[c:i], vocab))
                    dep_path_arr.append(_get_or_else_update(":<", vocab))
                    c = i+2
                elif dep_path[i:i+2] == ">:":
                    if c > 0:
                        dep_path_arr.append(_get_or_else_update(dep_path[c:i], vocab))
                    c = i+2
            if id1 != id2:
                triple = (_get_or_else_update(id1, concept_vocab), dep_path_arr, _get_or_else_update(id2, vocab), int(ct))
                triples.append(triple)
    return triples


def _get_or_else_update(entry, vocab):
    val = vocab.get(entry)
    if not val:
        val = len(vocab)
        vocab[entry] = val
    return val
