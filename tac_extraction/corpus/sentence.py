from .entity import Entity

class Sentence(object):

    def __init__(self, sentence_id, start, end, tokens):
        self.sentence_id = sentence_id
        self.span = (int(start), int(end))
        self.tokens = tokens
        self.entity_positions = None
        self.entities = None
        self.__init_entities()

    def translate(self, vocab):
        return [vocab[int(token)] for token in self.tokens]

    def get_source_document(self):
        return self.sentence_id.rsplit(".", 2)[0]

    def __init_entities(self):
        if not self.entity_positions:
            self.entity_positions = []
            self.entities = []
            for i,t in enumerate(self.tokens):
                if t.find(Entity.METADATA_SEPERATOR) != -1:
                    vocab_id, entity = Entity.from_token(t)
                    self.tokens[i] = vocab_id
                    self.entity_positions.append(i)
                    self.entities.append(entity)




