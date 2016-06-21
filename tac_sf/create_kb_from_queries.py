import sys
import xapian as x
import random
import pickle
from kb import KB
import os
from tac_sf import *

kb_file = sys.argv[1]
query_file = sys.argv[2]
assessment_dir = sys.argv[3]
print("Read KB from %s, queries from %s, and assessments from %s" % (kb_file, query_file, assessment_dir))
vocab_file = sys.argv[4]
rev_vocab_file = sys.argv[5]
geonames = sys.argv[6]
out = sys.argv[7]
dataset = sys.argv[8]

kb = KB()

if os.path.exists(out):
    kb.load(out)

print("Loading vocab...")
with open(vocab_file, "rb") as vocab_pickle:
    id_to_word = pickle.load(vocab_pickle)

with open(rev_vocab_file, "rb") as vocab_pickle:
    word_to_id = pickle.load(vocab_pickle)

index = x.Database(kb_file)
enquire = x.Enquire(index)
num_docs = index.get_doccount()

# [(id, entity_name, begin, end, entity_type), ...]
tac_queries = read_query_file(query_file)
# { query_id: { slot: [(doc_id, answer, span, assessment), ...], ...}, ... }
tac_assessments = read_assessments_dir(assessment_dir)

print("Loading geonames...")
cities, state_or_province, countries = load_geonames(geonames)

count = 0
total = 0
print("Adding examples ...")
empty_dict = dict()
for (query_id, entity_name, doc_id, entity_begin, entity_end, entity_type) in tac_queries:
    print("Processing query: %s" % query_id)
    # find entity document
    entity_query = x.Query("XD"+doc_id)
    enquire.set_query(entity_query)
    matches = enquire.get_mset(0, 1000)
    sentences = [pickle.loads(match.document.get_data()) for match in matches]

    query_entity, query_entity_token_id, query_entity_token = None, None, None
    for sentence in sentences:
        if sentence.span[0] <= entity_begin and sentence.span[1] >= entity_end:
            for i, entity in enumerate(sentence.entities):
                if entity.start <= entity_begin and entity.end >= entity_end:
                    query_entity = entity
                    query_entity_token_id = sentence.tokens[sentence.entity_positions[i]]
                    query_entity_token = id_to_word[int(query_entity_token_id)]
                    break
        if query_entity:
            break

    if query_entity is None:
        print("WARN: Skipping! Could not resolve query %s with entity %s in doc %s at %d-%d." %
              (query_id, entity_name, doc_id, entity_begin, entity_end))
    else:
        # find support
        entity_query = x.Query("XM" + query_entity_token_id)
        if query_entity.type == "PERSON" and ' ' in query_entity.surface_form:
            last_name_id = word_to_id.get(query_entity.surface_form.split()[-1])
            if last_name_id:
                # also query for last name
                entity_query = x.Query(x.Query.OP_OR, [entity_query, x.Query("XM" + str(last_name_id))])
        elif query_entity.type == "ORG" or query_entity.type == "GPE" and not query_entity_token.startswith("the"):
            last_name_id = word_to_id.get("the " + query_entity_token)
            if last_name_id:
                # also query for last name
                entity_query = x.Query(x.Query.OP_OR, [entity_query, x.Query("XM" + str(last_name_id))])
        elif query_entity.type == "ORG" or query_entity.type == "GPE" and query_entity_token.startswith("the"):
            last_name_id = word_to_id.get(query_entity_token[4:])
            if last_name_id:
                # also query for last name
                entity_query = x.Query(x.Query.OP_OR, [entity_query, x.Query("XM" + str(last_name_id))])
        if query_entity_token.endswith("'s"):
            last_name_id = word_to_id.get(query_entity_token[:-2])
            if last_name_id:
                # also query for last name
                entity_query = x.Query(x.Query.OP_OR, [entity_query, x.Query("XM" + str(last_name_id))])

        enquire.set_query(entity_query)

        matches = enquire.get_mset(0, 100000)
        support_docs = [indexed_sentence_to_paragraph(pickle.loads(doc.get_data()), enquire, -1)
                        for doc in set(match.document for match in matches)]

        if not support_docs:
            print("WARN: No support for entity '%s' ('%s') of doc %s at %d-%d." %
                  (entity_name, query_entity_token, doc_id, entity_begin, entity_end))
        else:
            # add all slots
            for slot, assessed_answers in tac_assessments.get(query_id, empty_dict).items():
                pos_assessments = [a for a in assessed_answers if a[4] == "C"]
                if pos_assessments:
                    answer_types = slot_answer_types[slot]
                    # keep track of all entities in context of this query for anonymization
                    context_entities = dict()
                    context_entities[query_entity.type] = set()
                    query_entity_token = query_entity_token + "@" + query_entity.type
                    context_entities[query_entity.type].add(query_entity_token)

                    def add_doc_to_context(doc):
                        for e, p in zip(doc.entities, doc.entity_positions):
                            relabel_entity_type(e, answer_types, cities, countries, state_or_province)
                            if e.type not in context_entities:
                                context_entities[e.type] = set()
                            context_entities[e.type].add(id_to_word[int(doc.tokens[p])] + "@" + e.type)

                    for support_doc in support_docs:
                        add_doc_to_context(support_doc)

                    total += 1
                    support = []
                    answer = None

                    for support_doc in support_docs:
                        potential_answers = [p for e, p in zip(support_doc.entities, support_doc.entity_positions)
                                             if e.type in answer_types]
                        if potential_answers:
                            support_tokens = support_doc.translate(id_to_word)
                            for e, p in zip(support_doc.entities, support_doc.entity_positions):
                                support_tokens[p] = support_tokens[p] + "@" + e.type
                            support.append((support_tokens, potential_answers))

                            for (doc_id, _, start, end, assessment) in pos_assessments:
                                if support_doc.sentence_id.startswith(doc_id):
                                    potential_answer_entities = [e for e in support_doc.entities if e.type in answer_types]
                                    for e, p in zip(support_doc.entities, support_doc.entity_positions):
                                        if e.type in answer_types and \
                                                (e.start <= start <= e.end or start <= e.start <= end):
                                            answer = id_to_word[int(support_doc.tokens[p])] + "@" + e.type
                                            break
                                    if answer:
                                        break

                    if not answer:
                        if not support:
                            print("WARN: No support for slot '%s' and query entity '%s' ('%s') of doc %s at %d-%d." %
                                  (slot, entity_name, id_to_word[int(query_entity_token_id)], doc_id, entity_begin, entity_end))
                        else:
                            print("WARN: Correct answers not in support for slot %s and query entity '%s' ('%s') of doc %s at %d-%d." %
                                  (slot, entity_name, id_to_word[int(query_entity_token_id)], doc_id, entity_begin, entity_end))
                            print("Positive Assessments:")
                            print(pos_assessments)
                    else:
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
                        context = [anonym_map[query_entity_token]]
                        context.extend(slot_to_text[slot].split())
                        context.append("@placeholder")

                        spans = [(len(context)-1, len(context))]
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

print("Extracted %d of %d possible examples." % (count, total))

print("Order vocabulary by frequency...")
kb.order_vocab_by_freq()
print("Pickle KB to %s" % out)
kb.save(out)
print("Done")
