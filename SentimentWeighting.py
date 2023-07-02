from whoosh.scoring import WeightingModel
from whoosh import scoring

class SentimentWeighting(WeightingModel):
    use_final = True

    def scorer(self, searcher, fieldnum, text, qf=1):
        return scoring.BM25FScorer(searcher, fieldnum, text, qf, K1=1.5)

    def final(self, searcher, docnum, score):
        # multiply the BM25F score by the sentiment score
        sentiment_score = searcher.stored_fields(docnum)["sentiment"]
        return score * sentiment_score
