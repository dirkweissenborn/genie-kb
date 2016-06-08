class ContextQueries:

    def __init__(self, context, queries, supporting_evidence=None, collaborative_support=False, source=None):
        assert all((isinstance(q, ContextQuery) and q.context == context for q in queries)), \
            "Context queries must share same context."
        self.context = context
        self.queries = queries
        self.collaborative_support = collaborative_support
        #supporting evidence for all queries
        self.supporting_evidence = supporting_evidence
        self.source = source  # information about source of this query (optional, maybe useful for sampling)


class ContextQuery:

    def __init__(self, context, start, end, answer, neg_candidates, supporting_evidence=None):
        self.context = context
        self.start = start
        self.end = end
        self.answer = answer
        self.neg_candidates = neg_candidates
        self.supporting_evidence = supporting_evidence


def flatten_queries(context_queries_list):
    ret = []
    for qs in context_queries_list:
        ret.extend(qs.queries)
    return ret