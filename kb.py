import pickle
from array import array
from threading import Lock


class KB:
    """
     KB represents a knowledge base of contexts with "points of interest", i.e., parts of the context
     that ought to be predicted.
     """
    def __init__(self):
        # holds all contexts
        self.__contexts = dict()
        # holds all spans of interest for aligned contexts
        self.__starts = dict()
        self.__ends = dict()
        # hold answers to respective spans, if None answer, span in context is answer itself
        self.__answers = dict()
        # holds all offsets for spans and contexts which are all stored in one memory efficient array
        self.__context_offsets = dict()
        self.__span_offsets = dict()
        # holds list of all symbols
        self.__vocab = list()
        # holds mappings of symbols to indices in every dimension
        self.__ids = dict()
        self.__max_context_length = 0
        self.__max_span_length = 0
        self.__lock = Lock()

    def add(self, context, spans, answers=None, dataset="train"):
        self.__lock.acquire()
        try:
            if dataset not in self.__contexts:
                self.__contexts[dataset] = array('I')
                self.__starts[dataset] = array('I')
                self.__ends[dataset] = array('I')
                self.__answers[dataset] = array('I')
                self.__context_offsets[dataset] = list()
                self.__span_offsets[dataset] = list()

            self.__context_offsets[dataset].append(len(self.__contexts[dataset]))
            self.__span_offsets[dataset].append(len(self.__starts[dataset]))
            self.__contexts[dataset].extend(self.__add_to_vocab(w) for w in context)
            for span in spans:
                self.__starts[dataset].append(span[0])
                self.__ends[dataset].append(span[1])
            if answers is not None and self.__answers:
                assert len(answers) == len(spans), "answers must align with spans"
                for answer in answers:
                    self.__answers[dataset].append(self.__add_to_vocab(answer))
            self.__max_context_length = max(self.__max_context_length, len(context))
            self.__max_span_length = max(self.__max_span_length, len(spans))
        finally:
            self.__lock.release()

    def __add_to_vocab(self, key):
        if key not in self.__ids:
            self.__ids[key] = len(self.__vocab)
            self.__vocab.append(key)
        return self.__ids[key]

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump([self.__contexts, self.__starts, self.__ends, self.__answers, self.__vocab, self.__ids,
                         self.__context_offsets, self.__span_offsets,
                         self.__max_context_length, self.__max_span_length], f)

    def load(self, file):
        with open(file, 'rb') as f:
            [self.__contexts, self.__starts, self.__ends, self.__answers, self.__vocab, self.__ids,
             self.__context_offsets, self.__span_offsets,
             self.__max_context_length, self.__max_span_length] = pickle.load(f)

    def context(self, typ, i):
        offset = self.__context_offsets[typ][i]
        end = self.__context_offsets[typ][i+1] if i+1<len(self.__context_offsets[typ]) else -1
        return self.__contexts[typ][offset:end]

    def num_contexts(self, typ):
        return len(self.__context_offsets.get(typ, []))

    def spans(self, typ, i):
        offset = self.__span_offsets[typ][i]
        end = self.__span_offsets[typ][i + 1] if i + 1 < len(self.__span_offsets[typ]) else -1
        return self.__starts[typ][offset:end], self.__ends[typ][offset:end]

    def answers(self, typ, i):
        offset = self.__span_offsets[typ][i]
        end = self.__span_offsets[typ][i + 1] if i + 1 < len(self.__span_offsets[typ]) else -1
        if self.__answers:
            return self.__answers[typ][offset:end]
        else:
            # if now answers provided used starts as answers
            return [self.context(typ, i)[p] for p in self.__starts[typ][offset * 2:end * 2]]

    def id(self, word, fallback=-1):
        return self.__ids.get(word, fallback)

    def iter_contexts(self, typ):
        for i in range(len(self.__context_offsets[typ])):
            yield self.context(typ, i)

    def iter_spans(self, typ):
        for i in range(len(self.__span_offsets[typ])):
            yield self.spans(typ, i)

    def iter_answers(self, typ):
        for i in range(len(self.__span_offsets[typ])):
            yield self.answers(typ, i)

    @property
    def max_context_length(self):
        return self.__max_context_length

    @property
    def max_span_length(self):
        return self.__max_span_length

    @property
    def vocab(self):
        return self.__vocab
