from whoosh import scoring


class SentimentWeightedTF_IDF(scoring.TF_IDF):
    def scorer(self, searcher, fieldname, text, qf=1):
        # Ottieni il punteggio originale TF-IDF
        tfidf = super().scorer(searcher, fieldname, text, qf)



        # Restituisce un punteggio pesato basato sul sentimento
        return tfidf * (1 + searcher.stored_fields(docnum)["sentiment"])