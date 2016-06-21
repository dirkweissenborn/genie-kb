import xml.etree.ElementTree
import os
from xml.dom.minidom import parse
import xapian as x
import pickle
from tac_extraction.corpus.sentence import Sentence

doc_splitter = "||||"

def read_query_file(fn):
    queries = list()
    x = xml.etree.ElementTree.parse(fn)
    for q in x.findall("query"):
        id = q.get("id")
        source = q.findall("docid")[0].text
        entity_name = q.findall("name")[0].text
        entity_type = q.findall("enttype")[0].text
        begin = int(q.findall("beg")[0].text)
        end = int(q.findall("end")[0].text) + 1
        queries.append((id, entity_name, source, begin, end, entity_type))
    return queries


def read_assessments_dir(dir):
    assessments = dict()
    for fn in os.listdir(dir):
        with open(os.path.join(dir, fn), 'rb') as f:
            for l in f:
                l = l.decode("utf-8")
                split = l.strip().split("\t")
                try:
                    if len(split) == 12:
                        #6	SF13_ENG_001:per:age	LTW_ENG_20090727.0007	44	3563-3564	3551-3560,954-971	3489-3678	C	C	C	C	4
                        [_, slot_query, doc_id, answer, spans, _, _, assessment, _, _, _, _] = split
                        [query_id, slot] = slot_query.split(":", 1)
                        if query_id not in assessments:
                            assessments[query_id] = dict()
                        a = assessments[query_id]
                        if slot not in a:
                            a[slot] = list()
                        for span in spans.split(","):
                            [begin, end] = span.split("-")
                            a[slot].append((doc_id, answer, int(begin), int(end)+1, assessment))
                    else:
                        #3	SF14_ENG_001:per:cause_of_death	LTW_ENG_20091012.0003:1353-1417	war	LTW_ENG_20091012.0003:1353-1355	W	W	0
                        [_, slot_query, _, answer, doc_id_spans, assessment, _, _] = split
                        [query_id, slot] = slot_query.split(":", 1)

                        if query_id not in assessments:
                            assessments[query_id] = dict()
                        a = assessments[query_id]
                        if slot not in a:
                            a[slot] = list()
                        for doc_id_span in doc_id_spans.split(","):
                            [doc_id, span] = doc_id_span.split(":", 1)
                            [begin, end] = span.split("-")
                            a[slot].append((doc_id, answer, int(begin), int(end)+1, assessment))
                except ValueError as err:
                    raise Exception("Could not parse assessment line %s. \n %s" % (l, err))
    return assessments


def indexed_sentence_to_paragraph(sentence, enquire, max_window=-1):
    split_id = sentence.sentence_id.split(".")
    doc_id = '.'.join(split_id[:-2])
    paragraph_id = '.'.join(split_id[:-1])
    sentence_idx = int(split_id[-1])
    q = x.Query("XD"+doc_id)
    enquire.set_query(q)
    matches = enquire.get_mset(0, 1000)
    sentences = [pickle.loads(match.document.get_data()) for match in matches]
    #sentences = [s for s in sentences if s.sentence_id.startswith(paragraph_id)]
    sentences = sorted(sentences, key=lambda s: s.sentence_id)
    tokens = []
    entities = []
    entity_positions = []
    start = -1
    end = -1
    for s in sentences:
        s_idx = int(s.sentence_id.split(".")[-1])
        if max_window < 0 or max_window >= abs(s_idx - sentence_idx):
            if start == -1:
                start = s.span[0]
            for e, p in zip(s.entities, s.entity_positions):
                entities.append(e)
                entity_positions.append(p+len(tokens))
            tokens.extend(s.tokens)
            end = s.span[1]
    new_sentence = Sentence(paragraph_id, start, end, tokens)
    new_sentence.entities = entities
    new_sentence.entity_positions = entity_positions
    return new_sentence


def load_geonames(fn):
    cities = set()
    state_or_province = set()
    countries = set()
    with open(fn, 'rb') as f:
        for l in f:
            split = l.decode("utf-8").split("\t")
            names = split[3].split(",")
            if split[7].startswith("PPL"): # then city
                for n in names:
                    cities.add(n.lower())
            elif split[7].startswith("ADM"):
                for n in names:
                    state_or_province.add(n.lower())
            elif split[7].startswith("PCL"):
                for n in names:
                    countries.add(n.lower())
    return cities, state_or_province, countries


def relabel_entity_type(e, answer_types, cities, countries, state_or_province):
    if e.type == "LOC" or e.type == "GPE":
        relabeled = False
        # give preference to answer_typ, because same surface form can refer to many location types
        if "CITY" in answer_types and e.surface_form.lower() in cities:
            e.type = "CITY"
            relabeled = True
        elif "COUNTRY" in answer_types and e.surface_form.lower() in countries:
            e.type = "COUNTRY"
            relabeled = True
        elif "STATEORPROVINCE" in answer_types and e.surface_form.lower() in state_or_province:
            e.type = "STATEORPROVINCE"
            relabeled = True
        if not relabeled: # ordered by granularity
            if e.surface_form.lower() in countries:
                e.type = "COUNTRY"
            elif e.surface_form.lower() in state_or_province:
                e.type = "STATEORPROVINCE"
            elif e.surface_form.lower() in cities:
                e.type = "CITY"


slot_to_text = {
    "per:alternate_names": "is also known as",
    "org:alternate_names": "is also known as",
    "per:date_of_birth": "was born on",
    "org:political_religious_affiliation": "is affiliated with",
    "per:age": "has age of",
    "org:top_members_employees": "has top employees",
    "per:country_of_birth": "was born in country",
    "org:number_of_employees_members": "has number of employees or members",
    "per:stateorprovince_of_birth": "was born in state",
    "org:members": "has members",
    "per:city_of_birth": "was born in city",
    "org:member_of": "is member of",
    "per:origin": "has nationality or ethnicity",
    "per:date_of_death": "died on",
    "org:subsidiaries": "has subsidiaries",
    "org:parents": "has parent organization",
    "per:country_of_death": "died in country",
    "org:founded_by": "was founded by",
    "per:stateorprovince_of_death": "died in state",
    "org:date_founded": "was founded on",
    "per:city_of_death": "died in city",
    "org:date_dissolved": "dissolved on",
    "per:cause_of_death": "died of",
    "org:country_of_headquarters": "has headquarters in country",
    "org:stateorprovince_of_headquarters": "has headquarters in state",
    "org:city_of_headquarters": "has headquarters in city",
    "per:countries_of_residence": "lived in countries",
    "per:statesorprovinces_of_residence": "lived in states",
    "per:cities_of_residence": "lived in cities",
    "org:shareholders": "has shareholders",
    "per:schools_attended": "attended schools",
    "org:website": "has website",
    "per:title": "has title",
    "per:employee_or_member_of": "is employed at",
    "per:religion": "has religion",
    "per:spouse": "is married to",
    "per:children": "has children",
    "per:parents": "has parents",
    "per:siblings": "has siblings",
    "per:other_family": "has family members",
    "per:charges": "has charges",
    }


slot_answer_types = {
    "per:alternate_names": ["PERSON"],
    "org:alternate_names": ["ORG"],
    "per:date_of_birth": ["DATE"],
    "org:political_religious_affiliation": ["ORG", "NORP"],
    "per:age": ["CARDINAL", "DATE"],
    "org:top_members_employees": ["PERSON"],
    "per:country_of_birth": ["COUNTRY", "LOC", "GPE"],
    "org:number_of_employees_members": ["CARDINAL"],
    "per:stateorprovince_of_birth": ["STATEORPROVINCE", "LOC", "GPE"],
    "org:members": ["ORG", "GPE"],
    "per:city_of_birth": ["CITY", "LOC", "GPE"],
    "org:member_of": ["ORG"],
    "per:origin": ["NORP"],
    "per:date_of_death": ["DATE"],
    "org:subsidiaries": ["ORG"],
    "org:parents": ["ORG", "GPE", "NORP"],
    "per:country_of_death": ["COUNTRY", "LOC", "GPE"],
    "org:founded_by": ["PERSON", "ORG", "GPE"],
    "per:stateorprovince_of_death": ["STATEORPROVINCE", "LOC", "GPE"],
    "org:date_founded": ["DATE"],
    "per:city_of_death": ["CITY", "LOC", "GPE"],
    "org:date_dissolved": ["DATE"],
    "per:cause_of_death": ["DISEASE"],
    "org:country_of_headquarters": ["COUNTRY", "LOC", "GPE"],
    "org:stateorprovince_of_headquarters": ["STATEORPROVINCE", "LOC", "GPE"],
    "org:city_of_headquarters": ["CITY", "LOC", "GPE"],
    "per:countries_of_residence": ["COUNTRY", "LOC", "GPE"],
    "per:statesorprovinces_of_residence": ["STATEORPROVINCE", "LOC", "GPE"],
    "per:cities_of_residence": ["CITY", "LOC", "GPE"],
    "org:shareholders": ["PERSON", "ORG", "GPE"],
    "per:schools_attended": ["ORG"],
    "org:website": ["URL"],
    "per:title": ["TITLE"],
    "per:employee_or_member_of": ["ORG", "GPE"],
    "per:religion": ["NORP"],
    "per:spouse": ["PERSON"],
    "per:children": ["PERSON"],
    "per:parents": ["PERSON"],
    "per:siblings": ["PERSON"],
    "per:other_family": ["PERSON"],
    "per:charges": ["CHARGE"],
    }