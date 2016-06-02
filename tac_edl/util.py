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