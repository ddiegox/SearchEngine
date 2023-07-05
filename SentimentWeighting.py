from whoosh.scoring import WeightingModel
from whoosh import scoring

class SentimentWeighting(WeightingModel):
    use_final = True
    ranking = 1

    def __init__(self, r):
        self.ranking = r

    @property
    def scorer(self, searcher, fieldnum, text, qf=1):
        if(self.ranking == 2):
            return scoring.BM25FScorer(searcher, fieldnum, text, qf, K1=1.5)
        else:
            return scoring.TF_IDFScorer(searcher, fieldnum, text)

    def final(self, searcher, docnum, score):
        # multiply the BM25F score by the sentiment score
        sentiment_score = searcher.stored_fields(docnum)["sentiment"]
        return score * sentiment_score
