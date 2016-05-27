import os
import sys
from kb import KB
from multiprocessing.dummy import Pool

dir = sys.argv[1]
print("Read dataset from %s" % dir)
out = sys.argv[2]

pool = Pool()

train_path = os.path.join(dir, "training")
valid_path = os.path.join(dir, "validation")
test_path = os.path.join(dir, "test")

kb = KB()

k = [0]
num_files = [0]
def add2kb(fn, typ):
    with open(fn, 'rb') as f:
        content = f.readlines()
        document = content[2].decode('utf-8').strip().split(' ')
        question = content[4].decode('utf-8').strip().split(' ')
        total = document+["||"]+question
        positions = [i for i in range(len(total)) if total[i].startswith("@entity")]

        answer = content[6].decode('utf-8').strip()
        for i in range(len(question)):
            if question[i] == "@placeholder":
                position = i+len(document)+1
                total[position] = answer
                positions.append(position)
        kb.add(total, positions, typ)

        k[0] += 1
        if k[0] % 100 == 0:
            sys.stdout.write("\r%.1f %% of %d files read ..." % (k[0] / (num_files[0]//100), num_files[0]))
            sys.stdout.flush()


def filename_iterator(path):
    k[0] = 0
    num_files[0] = len(os.listdir(path))
    print("Processing %s" % path)
    for fn in os.listdir(path):
        yield os.path.join(path, fn)

if os.path.exists(out):
    kb.load(out)

if kb.num_contexts("valid") == 0:
    pool.map(lambda fn: add2kb(fn, "valid"), filename_iterator(valid_path))
    print("")
    print("Pickle KB to %s" % out)
    kb.save(out)
if kb.num_contexts("test") == 0:
    pool.map(lambda fn: add2kb(fn, "test"), filename_iterator(test_path))
    print("")
    print("Pickle KB to %s" % out)
    kb.save(out)
if kb.num_contexts("train") == 0:
    pool.map(lambda fn: add2kb(fn, "train"), filename_iterator(train_path))
    print("")
    print("Pickle KB to %s" % out)
    kb.save(out)
print("Done")

