from annoy import AnnoyIndex

class ANNSearcher:

    def __init__(self, repr, num_trees=1, num_candidates=100):
        self.__index = AnnoyIndex(repr.shape[1])
        self.__num_trees = num_trees
        self.__num_candidates = num_candidates
        self.update(repr)

    def set_num_candidate(self, num_candidates):
        self.__num_candidates = num_candidates

    def update(self, repr):
        self.__index.unload()
        for i in range(repr.shape[0]):
            self.__index.add_item(i, repr[i])
        self.__index.build(self.__num_trees)

    def search(self, rep):
        self.__index.get_nns_by_vector(rep, self.__num_candidates)

