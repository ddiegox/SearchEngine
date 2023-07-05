from whoosh.scoring import WeightingModel
from whoosh import scoring

class SentimentWeighting(WeightingModel):
    use_final = True

    def __init__(self, r):
        self.ranking = r

    def scorer(self, searcher, fieldnum, text, qf=1):
        if(self.ranking == 2):
            return scoring.BM25F().scorer(searcher, fieldnum, text, qf)
        else:
            return scoring.TF_IDF().scorer(searcher, fieldnum, text, qf)

    def final(self, searcher, docnum, score):
        # multiply the BM25F score by the sentiment score
        sentiment_score = searcher.stored_fields(docnum)["sentiment"]
        return score * sentiment_score
