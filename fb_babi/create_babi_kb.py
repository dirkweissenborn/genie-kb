import os
import sys
from kb import KB
from multiprocessing.dummy import Pool
from fb_babi import *
from spacy.en import English

dir = sys.argv[1]
print("Read dataset from %s" % dir)
out = sys.argv[2]

pool = Pool()

kb = KB()
nlp = English(parser=False)
"""
1 Mary journeyed to the hallway.
2 Mary travelled to the bedroom.
3 Where is Mary?        bedroom 2
4 Mary journeyed to the garden.
5 Sandra journeyed to the hallway.
6 Where is Mary?        garden  4
"""
def add2kb(fn, typ):
    print("Adding dataset: %s" % typ)
    with open(fn, 'rb') as f:
        content, answers, query_spans = [], [], []
        for l in f:
            l = l.decode("utf-8").strip()
            if content and l.startswith('1 '):
                kb.add(content, query_spans, answers, typ)
                content = []
                answers = []
                query_spans = []

            if '\t' in l:
                [l, a] = l.split("\t")
                aws = a.split(' ')[0].split(",")
                for a in aws:
                    if a != 'nothing':
                        a = 'NIL'
                    query_spans.append((len(content), len(content)+1))
                    answers.append(a)

            tokens = [t.orth_ for t in nlp(l) if not t.is_space][1:]
            if content:
                content.append(newline)
            for t in tokens:
                content.append(t.orth_.lower())
        if content:
            kb.add(content, query_spans, answers, typ)

def filename_iterator(path):
    for fn in os.listdir(path):
        if fn.startswith("task_"):
            yield os.path.join(path, fn), fn[fn.index('_')+1:-4]

for fn in filename_iterator(dir):
    add2kb(fn[0], fn[1])

print("")
print("Order vocabulary by frequency...")
kb.order_vocab_by_freq()
print("Pickle KB to %s" % out)
kb.save(out)
print("Done")

