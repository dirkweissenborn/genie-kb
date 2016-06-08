from kb import FactKB
import os
import sys
from multiprocessing.dummy import Pool
from spacy.en import English
import tac_edl as util

query_file = sys.argv[1]
print("Read Queries from %s" % query_file)
link_file = sys.argv[2]
print("Read Linking from %s" % link_file)
doc_dir = sys.argv[3]
print("Read TAC Documents from %s" % doc_dir)
out = sys.argv[4]
with_ed = sys.argv[5] == "True"
dataset = sys.argv[6] if len(sys.argv) > 6 else "train"

nlp = English(parser=False)

queries = util.read_query_file(query_file)
answers = util.read_link_file(link_file)

pool = Pool()
kb = FactKB()

if os.path.exists(out):
    print("Loading existing TAC EDL KB from %s" % out)
    kb.load(out)

k = [0]
num_files = [len(queries)]

def add2kb(fn, queries):
    # queries of form: [(id, name, begin, end, typ), ...]
    with open(fn, 'rb') as f:
        content = f.read().decode('utf-8')

    content = util.normalize(content)
    begin_lookup = {q[2]:q for q in queries}
    end_lookup = {q[3]:q for q in queries}
    tokens = nlp(content)
    i = 0
    start = -1
    spans = []
    entities = []
    typs = []

    token_spans = []
    token_typs = []
    content = []
    within_tags = False
    while i < len(tokens):
        t = tokens[i]
        if t.text == "<":
            within_tags = True

        if not within_tags and not t.is_space:
            q = begin_lookup.get(t.idx)
            offset = len(content)
            if q:
                start = offset

            if with_ed:
                token_spans.append(([offset, offset+1]))
                token_typs.append("[ENTITY]" if start >= 0 else "[NO_ENTITY]")

            content.append(t.text)
            q = end_lookup.get(t.idx+len(t)-1)
            if q and start >= 0:
                if q[0] in answers:
                    entities.append(answers[q[0]][0])
                    typs.append(answers[q[0]][1])
                    spans.append((start, offset+1))
                else:
                    print("WARN: %s not in linking file..." % q[0])
                start = -1
        else:
            if t.text == ">":
                within_tags = False

        i += 1

    kb.add(content, spans, entities, "TAC_EL/"+dataset)
    kb.add(content, spans, typs, "TAC_ET/"+dataset)
    if with_ed:
        kb.add(content, token_spans, token_typs, "TAC_ED/"+dataset)

    k[0] += 1
    sys.stdout.write("\r%.1f %% of %d files read ..." % (k[0]*100.0 / (num_files[0]), num_files[0]))
    sys.stdout.flush()


for t in queries.items():
    add2kb(os.path.join(doc_dir, t[0]+".xml"), t[1])
print("")
print("Pickle KB to %s" % out)
kb.save(out)

print("Done")