from whoosh import scoring

class RankingScorer:
    use_final = False
    def __init__(self, ranking=1):
        self.ranking = ranking

    def __call__(self, searcher, fieldname, text, qf=1):
        class Scorer(scoring.WeightLengthScorer):
            if self.ranking == 1:
                self.model = scoring.TF_IDF()
            else:
                self.model = scoring.BM25F()

            def _score(self, weight_length):
                # Compute base model score
                base_score = self.model._score(weight_length)

                # Compute sentiment score
                document = searcher.stored_fields(weight_length.id())
                sentiment_score = searcher.stored_fields(weight_length.id())["sentiment"]

                # Combine scores. Here we simply multiply them, but you could use any function you like
                return base_score * (1 + sentiment_score)

            def supports_block_quality(self):
                return self.model.supports_block_quality()

        return Scorer(searcher, fieldname, text, qf)