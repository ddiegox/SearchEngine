from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import NumericRange, And
from whoosh import scoring

from LemmaFilter import LemmaFilter
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter, StemFilter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from SentimentFilter import SentimentFilter
from SentimentWeighting import SentimentWeighting
from RankingScorer import RankingScorer
from textblob import TextBlob

import os, os.path
import csv
import argparse

def decoratorFn(ranking=1):
    def pos_score_fn(searcher, fieldname, text, matcher):
        if ranking==2:
            return scoring.BM25F().scorer(searcher, fieldname, text).score(matcher)
        else:
            return scoring.TF_IDF().scorer(searcher, fieldname, text).score(matcher)
    return pos_score_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search Restaurant Reviews")
    parser.add_argument('-s', '--sentiment', default=1, type=int, help="Indica la libreria di Sentiment Analysis da utilizzare (1-Vader, 2-Textblob), default: Vader")
    parser.add_argument('-r', '--ranking', default=1, type=int, help="Indica la funzione di ranking da utilizzare (1-TF-IDF (modello vettoriale), 2-BM25 (modello probabilistico)), default: TF-IDF")
    args = parser.parse_args()

    sentiment = int(args.sentiment)
    ranking = int(args.ranking)

    nltk.download('wordnet')
    nltk.download('vader_lexicon')

    print("Search Restaurant Reviews\n")

    #use csv parser to read reviews
    documents_list = []
    with open('src/Restaurant_Reviews.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            documents_list.append(row)

    #create analyzer for preprocessing
    analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter() | LemmaFilter() | StemFilter()
    #In questo codice:
    # RegexTokenizer suddivide il testo in token utilizzando espressioni regolari
    # LowercaseFilter converte tutti i token in minuscolo
    # StopFilter rimuove le parole di stop
    # LemmaFilter esegue la lemmatizzazione
    # StemFilter esegue lo stemming.

    #create schema
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True, analyzer=analyzer), sentiment=NUMERIC(stored=True, numtype=float))

    #create index
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    ix = create_in("indexdir", schema)

    #create writer
    writer = ix.writer()

    #create sentiment intensity analyzer (Vader)
    sentiment_analyzer = SentimentIntensityAnalyzer()

    title = "Review"
    num_document = 1
    for document in documents_list:
        content = document[0]
        document_title = title + " " + str(num_document)
        num_document = num_document + 1

        #calculate sentiment and add document to index
        if sentiment == 2:
            polarity = TextBlob(content)
            writer.add_document(title=document_title, content=content, sentiment=float(polarity.sentiment.polarity))
        else:
            polarity = sentiment_analyzer.polarity_scores(content)
            writer.add_document(title=document_title, content=content, sentiment=float(polarity["compound"]))


    writer.commit()

    query = input("Insert a query: ")
    filter = input("Insert 1 for filter, any value for not filter: ")
    if filter == "1":
        filter_value = int(input("Insert -1 for negative reviews or 1 for positive reviews: "))
        sentiment_filter = SentimentFilter(filter_value)

    ix = open_dir("indexdir")

    # Crea un oggetto di scoring
    scoringObject = scoring.FunctionWeighting(decoratorFn(ranking))

    searcher = ix.searcher(weighting=scoringObject)
    parser = QueryParser("content", ix.schema)
    query = parser.parse(query)

    if filter == "1":
        query = And([query,sentiment_filter])
    results = searcher.search(query)
    print("Number of documents returned: " + str(len(results)))
    for result in results:
        print(result)