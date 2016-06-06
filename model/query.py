class ContextQueries:

    def __init__(self, context, queries, supporting_evidence=None, collaborative_support=False):
        assert all((isinstance(q, ContextQuery) and q.context == context for q in queries)), \
            "Context queries must share same context."
        self.context = context
        self.queries = queries
        self.collaborative_support = collaborative_support
        self.supporting_evidence = supporting_evidence


class ContextQuery:

    def __init__(self, context, start, end, answer, neg_candidates):
        self.context = context
        self.start = start
        self.end = end
        self.answer = answer
        self.neg_candidates = neg_candidates
