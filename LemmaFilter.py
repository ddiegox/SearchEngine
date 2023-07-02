from whoosh.analysis import Filter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class LemmaFilter(Filter):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, tokens):
        for t in tokens:
            t.text = self.lemmatizer.lemmatize(t.text, wordnet.ADJ)
            yield t
