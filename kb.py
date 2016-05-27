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
        # holds all positions of interest for aligned contexts
        self.__positions = dict()
        # holds all offsets for positions and contexts which are all stored in one memory efficient array
        self.__context_offsets = dict()
        self.__position_offsets = dict()
        # holds list of all symbols
        self.__vocab = list()
        # holds mappings of symbols to indices in every dimension
        self.__ids = dict()
        self.__max_context_length = 0
        self.__max_position_length = 0
        self.__lock = Lock()

    def add(self, context, positions, dataset="train"):
        self.__lock.acquire()
        try:
            for w in context:
                self.__add_to_vocab(w)
            if dataset not in self.__contexts:
                self.__contexts[dataset] = array('I')
                self.__positions[dataset] = array('I')
                self.__context_offsets[dataset] = list()
                self.__position_offsets[dataset] = list()

            self.__context_offsets[dataset].append(len(self.__contexts[dataset]))
            self.__position_offsets[dataset].append(len(self.__positions[dataset]))
            self.__contexts[dataset].extend(self.__ids[w] for w in context)
            self.__positions[dataset].extend(positions)
            self.__max_context_length = max(self.__max_context_length, len(context))
            self.__max_position_length = max(self.__max_position_length, len(positions))
        finally:
            self.__lock.release()

    def __add_to_vocab(self, key):
        if key not in self.__ids:
            self.__ids[key] = len(self.__vocab)
            self.__vocab.append(key)

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump([self.__contexts, self.__positions, self.__vocab, self.__ids], f)

    def load(self, file):
        with open(file, 'rb') as f:
            [self.__contexts, self.__positions, self.__vocab, self.__ids] = pickle.load(f)

    def context(self, typ, i):
        offset = self.__context_offsets[typ][i]
        end = self.__context_offsets[typ][i+1] if i+1<len(self.__context_offsets[typ]) else -1
        return self.__contexts[typ][offset:end]

    def num_contexts(self, typ):
        return len(self.__context_offsets.get(typ, []))

    def positions(self, typ, i):
        offset = self.__position_offsets[typ][i]
        end = self.__position_offsets[typ][i+1] if i+1<len(self.__position_offsets[typ]) else -1
        return self.__positions[typ][offset:end]

    def id(self, word, fallback=-1):
        self.__ids.get(word, fallback)

    def iter_contexts(self, typ):
        for i in range(len(self.__context_offsets[typ])):
            yield self.context(typ, i)

    def iter_positions(self, typ):
        for i in range(len(self.__position_offsets[typ])):
            yield self.positions(typ, i)

    @property
    def max_context_length(self):
        return self.__max_context_length

    @property
    def max_context_length(self):
        return self.__max_position_length

    @property
    def vocab(self):
        return self.__vocab
