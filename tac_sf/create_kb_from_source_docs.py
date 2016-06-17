import sys
import xapian as x
import random
import pickle
from kb import KB
import os
from tac_sf import *

kb_file = sys.argv[1]
print("Read KB from %s" % kb_file)
num_examples = int(sys.argv[2])
vocab_file = sys.argv[3]
geonames = sys.argv[4]
out = sys.argv[5]
dataset = sys.argv[6] if len(sys.argv) > 6 else "train_unsupervised"

kb = KB()

if os.path.exists(out):
    kb.load(out)

print("Loading vocab...")
with open(vocab_file, "rb") as vocab_pickle:
    id_to_word = pickle.load(vocab_pickle)
print("Loading geonames...")
cities, state_or_province, countries = load_geonames(geonames)

index = x.Database(kb_file)
enquire = x.Enquire(index)
num_docs = index.get_doccount()

parser = x.QueryParser()

count = 0
print("Adding examples ...")
location_types = ["COUNTRY", "CITY", "STATEORPROVINCE"]
while count < num_examples:
    context_entities = dict()
    doc_id = random.randint(1, num_docs+1)
    doc = pickle.loads(index.get_document(doc_id).get_data())

    def add_doc_to_context(doc, answer_typ):
        answer_typ = [answer_typ]
        for e, p in zip(doc.entities, doc.entity_positions):
            relabel_entity_type(e, answer_typ, cities, countries, state_or_province)
            if e.type not in context_entities:
                context_entities[e.type] = set()
            context_entities[e.type].add(id_to_word[int(doc.tokens[p])] + "@" + e.type)

    matches = []
    if len(doc.entities) > 1:
        add_doc_to_context(doc, None)

        def add_match_to_support(support, match, answer_entity):
            support_doc = pickle.loads(match.document.get_data())
            if support_doc.tokens != doc.tokens:
                add_doc_to_context(support_doc, answer_entity.type)
                support_tokens = support_doc.translate(id_to_word)
                for e, p in zip(support_doc.entities, support_doc.entity_positions):
                    support_tokens[p] = support_tokens[p] + "@" + e.type
                potential_answers = [p for e, p in zip(support_doc.entities, support_doc.entity_positions)
                                     if e.type == answer_entity.type]

                support.append((support_tokens, potential_answers))

        rnd_idxs = list(range(0, len(doc.entities)))
        random.shuffle(rnd_idxs)
        for rnd_idx in rnd_idxs:
            answer_entity = doc.entities[rnd_idx]
            entity_queries = [x.Query("XM"+doc.tokens[p]) for p in doc.entity_positions]
            # retrieve distant supervision support (positive)
            or_query = x.Query(x.Query.OP_OR, entity_queries)
            query = x.Query(x.Query.OP_FILTER, or_query, x.Query("XM"+doc.tokens[doc.entity_positions[rnd_idx]]))

            enquire.set_query(query)
            pos_matches = enquire.get_mset(0, 100)
            if pos_matches:
                query_tokens = doc.translate(id_to_word)
                for e, p in zip(doc.entities, doc.entity_positions):
                    query_tokens[p] = query_tokens[p] + "@" + e.type
                answer = query_tokens[doc.entity_positions[rnd_idx]]
                # retrieve support without candidate entity
                enquire.set_query(or_query)
                neg_matches = enquire.get_mset(0, 1000)

                pos_support = []
                for m in pos_matches:
                    add_match_to_support(pos_support, m, answer_entity)

                neg_support = []
                for m in neg_matches:
                    add_match_to_support(neg_support, m, answer_entity)

                support = random.sample(pos_support, min(10, len(pos_support))) + \
                                        random.sample(neg_support, min(100, len(neg_support)))

                if any(any(context[p] != answer for p in answer_pos) for (context, answer_pos) in support):
                    # anonymize entities
                    anonym_map = dict()
                    for typ, entities in context_entities.items():
                        rnd_ids = list(range(max(100, len(entities))))
                        random.shuffle(rnd_ids)
                        for e, ix in zip(entities, rnd_ids):
                            anonym_map[e] = "@"+typ+str(ix)

                    def anonymize(context):
                        for i in range(len(context)):
                            context[i] = anonym_map[context[i]]

                    # There should be at least one wrong answer in support
                    context = query_tokens
                    spans = [(doc.entity_positions[rnd_idx], doc.entity_positions[rnd_idx]+1)]
                    answers = [anonym_map[answer]]
                    context.append(doc_splitter)
                    for (supp_context, supp_answer_pos) in support:
                        offset = len(context)
                        context.extend(supp_context)
                        for p in supp_answer_pos:
                            spans.append((p + offset, p + offset + 1))
                            answers.append(anonym_map[supp_context[p]])
                        context.append(doc_splitter)
                    for i in range(len(context)):
                        context[i] = anonym_map.get(context[i], context[i])
                    kb.add(context, spans, answers, dataset)
                    count += 1

                    if count % max(1, num_examples//1000):
                        sys.stdout.write("\r%d / %d training examples added ..." % (count, num_examples))
                        sys.stdout.flush()
                    break

print("Order vocabulary by frequency...")
kb.order_vocab_by_freq()
print("Pickle KB to %s" % out)
kb.save(out)
print("Done")
