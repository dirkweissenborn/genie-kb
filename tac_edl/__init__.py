import xml.etree.ElementTree
import re

def read_query_file(fn):
    queries = dict()
    x = xml.etree.ElementTree.parse(fn)
    for q in x.findall("query"):
        id = q.get("id")
        source = q.findall("docid")[0].text
        names = q.findall("name")
        if not names:
            names = q.findall("mention")
        name = names[0].text
        typs = q.findall("menttype")
        typ = typs[0].text if typs else None
        begin = int(q.findall("beg")[0].text)
        end = int(q.findall("end")[0].text)
        if not source in queries:
            queries[source] = []
        queries[source].append((id, name, begin, end, typ))
    return queries


def read_link_file(fn):
    #query_id	entity_id	entity_type	genre	web_search?	wiki_text?	unknown?
    with open(fn, 'rb') as f:
        first = True
        queries = dict()
        for l in f:
            if first:
                first = False
                continue
            [q_id, e_id, e_typ, genre] = l.decode("utf-8").split("\t")[:4]
            queries[q_id] = (e_id, e_typ, genre)
    return queries


def normalize(text):
    return re.sub('\d', 'D', text)


def get_batch(sess, sampler, searcher, model):
    batch = sampler.get_batch()
    fact_kb = sampler.fact_kb
    el_batch = [q for q in batch if q.source.startswith("TAC_EL/") or q.source == "TAC_KB"]
    if el_batch:
        query_reps = model.run(sess, model.query, el_batch)
        nns = [searcher.search(query_reps[i]) for i in range(query_reps.shape[0])]
    k = 0
    for queries in batch:
        if queries.source.startswith("TAC_ET/"):
            cands = ["FAC", "GPE", "LOC", "PER", "ORG"]
        elif queries.source.startswith("TAC_ED/"):
            cands = ["[NO_ENTITY]","[ENTITY]"]
        else:
            cands = []
        cands = [fact_kb.id(c) for c in cands if fact_kb.id(c) >= 0]
        for q in queries.queries:
            if queries.source.startswith("TAC_EL/") or queries.source == "TAC_KB":
                cands = [nn for nn in nns[k] if fact_kb.entity_vocab[nn].startswith("E")]
                k += 1
            q.neg_candidates = [c for c in cands if c != q.answer]
    return batch
