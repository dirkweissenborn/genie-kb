

#E0349508@@-@@Russia:GPE:Russia:194-200
class Entity(object):
    METADATA_SEPERATOR = "@@-@@"
    METADATA_VALUE_SEPERATOR = ":::"

    def __init__(self, surface_form, e_type, wiki_title, start, end):
        self.surface_form = surface_form
        self.type = e_type
        self.wiki_title = wiki_title
        self.start = int(start)
        self.end = int(end)


    def __str__(self):
        return "%s:%s:%s:%d-%d" % (self.surface_form, self.type, self.wiki_title, self.start, self.end)


    @staticmethod
    def from_token(token):
        vocab_id, metadata = token.split(Entity.METADATA_SEPERATOR)
        try:
            surface_form, e_type, wiki_title, span = metadata.split(Entity.METADATA_VALUE_SEPERATOR)
        except Exception as e:
            print(token)
            raise e
        start, end = span.split("-")
        return vocab_id, Entity(surface_form, e_type, wiki_title, start, end)

    @staticmethod
    def is_token_entity(token):
        return token.find(Entity.METADATA_SEPERATOR) != -1
