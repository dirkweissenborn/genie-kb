from kb import FactKB
import os
import sys
from multiprocessing.dummy import Pool
import xml.etree.ElementTree
import re

dir = sys.argv[1]
print("Read TAC KB from %s" % dir)
out = sys.argv[2]

pool = Pool()
kb = FactKB()

k = [0]
num_files = [0]
def add2kb(fn):
    with open(fn, 'rb') as f:
        content = f.read().decode('utf-8')
    content = content.replace("<link>", " $!! ").replace("</link>", " !!$ ")
    content = re.sub(r'<link entity_id="(E\d+)">', r' $!!{{\1}} ', content)

    e = xml.etree.ElementTree.fromstring(content)
    for entity in e.findall('entity'):
        e_name = entity.get("name").split()
        e_id = entity.get("id")
        for facts in entity.findall("facts"):
            for fact in facts.findall('fact'):
                #<fact name="countryofbirth"><link entity_id="E0145816">England</link></fact>
                slot = fact.get("name")
                text = fact.text.split()
                clean_fact = e_name + [slot]
                spans = [(0, len(e_name))]
                entities = [e_id]
                last_start = 0
                is_entity = False
                for w in text:
                    if w == "$!!":
                        is_entity = False
                    elif w.startswith("$!!{{"):
                        is_entity = True
                        entities.append(w[5:-2])
                        last_start = len(clean_fact)
                    elif w == "!!$":
                        if is_entity:
                            spans.append((last_start, len(clean_fact)))
                    else:
                        clean_fact.append(w)
                kb.add(clean_fact, spans, entities)

    k[0] += 1
    sys.stdout.write("\r%.1f %% of %d files read ..." % (k[0]*100.0 / (num_files[0]), num_files[0]))
    sys.stdout.flush()


def filename_iterator(path):
    k[0] = 0
    num_files[0] = len(os.listdir(path))
    print("Processing %s" % path)
    for fn in os.listdir(path):
        yield os.path.join(path, fn)

pool.map(lambda fn: add2kb(fn), filename_iterator(dir))
print("")
print("Pickle KB to %s" % out)
kb.save(out)

print("Done")