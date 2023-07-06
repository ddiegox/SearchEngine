from whoosh.scoring import WeightingModel
from whoosh import scoring

class SentimentWeighting(WeightingModel):
    use_final = True

    def __init__(self, irmodel, ranking_function):
        self.irmodel = irmodel
        self.ranking_function = ranking_function

    def scorer(self, searcher, fieldnum, text, qf=1):
        if(self.irmodel == 2):
            return scoring.BM25F().scorer(searcher, fieldnum, text, qf)
        else:
            return scoring.TF_IDF().scorer(searcher, fieldnum, text, qf)

    def final(self, searcher, docnum, score):
        # multiply the BM25F score by the sentiment score
        sentiment_score = searcher.stored_fields(docnum)["sentiment"]

        if self.ranking_function == 2:
            return score * 0.6 + sentiment_score * 0.4
        else:
            return score * sentiment_score