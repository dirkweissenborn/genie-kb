import os
import sys
from kb import KB
from multiprocessing.dummy import Pool
from cbt import *

dir = sys.argv[1]
print("Read dataset from %s" % dir)
out = sys.argv[2]

pool = Pool()

kb = KB()

def add2kb(fn, typ):
    print("Adding dataset: %s" % typ)
    with open(fn, 'rb') as f:
        content = []
        for l in f:
            l = l.decode("utf-8").strip()
            if l == "":
                if content:
                    # context is over
                    start = content.index(placeholder)
                    answer = content[-2]
                    answer_cands = content[-1].split("|")
                    for i in range(len(content)):
                        if content[i] in answer_cands:
                            k = answer_cands.index(content[i])
                            content[i] = "@candidate%d" % k
                    content = content[:-2] + [answer_sep] + ["@candidate%d" % k for k in range(len(answer_cands))]
                    kb.add(content, [(start, start+1)], [answer], typ)
                    content = []
            else:
                if content:
                    content.append(newline)
                content.extend(l.split()[1:])


def filename_iterator(path):
    for fn in os.listdir(path):
        if fn.startswith("cbtest"):
            yield os.path.join(path, fn), fn[fn.index('_')+1:-4]

for fn in filename_iterator(dir):
    add2kb(fn[0], fn[1])

print("")
print("Order vocabulary by frequency...")
kb.order_vocab_by_freq()
print("Pickle KB to %s" % out)
kb.save(out)
print("Done")

