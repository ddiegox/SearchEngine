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
from textblob import TextBlob

import os, os.path
import csv
import argparse
from math import log
import json

dcg_output = {}

def run_query(f, text_query, filter, ranking, sentiment, sentiment_filter=None):
    f.write("\nQuery to analize: "+text_query)
    ix = open_dir("indexdir")

    if text_query not in dcg_output:
        dcg_output[text_query] = []

    # Crea un oggetto di scoring
    scoringObject = SentimentWeighting(ranking)

    searcher = ix.searcher(weighting=scoringObject)
    parser = QueryParser("content", ix.schema)
    query = parser.parse(text_query)

    if filter == "1":
        query = And([query, sentiment_filter])
    results = searcher.search(query)

    f.write("\nNumber of documents returned: " + str(len(results)) + "\n")
    i = 1
    dcg = 0.0
    for hit in results:
        f.write("Document title: "+hit["title"]+"\n")
        f.write("Content: "+hit["content"]+"\n")
        f.write("Sentiment: "+str(hit["sentiment"])+"\n")
        f.write("Weight: "+str(hit.score)+"\n")
        f.write("-------------------------"+"\n")
        dcg += (2 ** hit.score - 1) / (log(i + 1, 2))
    f.write("DCG = " + str(dcg)+"\n")
    f.write("-------------------------\n")

    dcg_output[text_query].append({"sentiment": "VADER" if sentiment==1 else "TextBlob", "ranking": "TF-IDF" if ranking==1 else "BM25F", "dcg": dcg})

def run(f, documents_list, schema, ranking=1, sentiment=1, test_query=None):

    ix = create_in("indexdir", schema)

    # create writer
    writer = ix.writer()

    # create sentiment intensity analyzer (VADER)
    sentiment_analyzer = SentimentIntensityAnalyzer()

    f.write("RUN\n\n")

    f.write("Sentiment Analysis used library: "+"\n")
    if sentiment == 2:
        f.write("TextBlob\n")
    else:
        f.write("VADER\n")
    f.write("\nIR Model used: "+"\n")
    if(ranking == 2):
        f.write("BM25F\n")
    else:
        f.write("TF_IDF\n")

    title = "Review"
    num_document = 1
    for document in documents_list:
        content = document[0]
        document_title = title + " " + str(num_document)
        num_document = num_document + 1

        # calculate sentiment and add document to index
        if sentiment == 2:
            polarity = TextBlob(content)
            writer.add_document(title=document_title, content=content, sentiment=float(polarity.sentiment.polarity))
        else:
            polarity = sentiment_analyzer.polarity_scores(content)
            writer.add_document(title=document_title, content=content, sentiment=float(polarity["compound"]))

    writer.commit()

    if test_query is None:
        query = input("Insert a query: ")
        query = query.strip()
        filter = input("Insert 1 for filter, any value for not filter: ")
        sentiment_filter = None
        if filter == "1":
            filter_value = int(input("Insert -1 for negative reviews or 1 for positive reviews: "))
            sentiment_filter = SentimentFilter(filter_value)

        run_query(f, query, filter, ranking,sentiment,sentiment_filter)
    else:
        for query in test_query:
            run_query(f, query, "0", ranking, sentiment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search Restaurant Reviews")
    parser.add_argument('-t', '--test', default=0, type=int, help="Se presente, indica che lo script deve essere attivato in modalità test")
    parser.add_argument('-s', '--sentiment', default=1, type=int, help="Indica la libreria di Sentiment Analysis da utilizzare (1-VADER, 2-Textblob), default: VADER")
    parser.add_argument('-r', '--ranking', default=1, type=int, help="Indica la funzione di ranking da utilizzare in combinazione con la sentiment analysis (1-TF-IDF (modello vettoriale), 2-BM25 (modello probabilistico)), default: TF-IDF")
    args = parser.parse_args()

    sentiment = int(args.sentiment)
    ranking = int(args.ranking)
    test = int(args.test)

    nltk.download('wordnet')
    nltk.download('VADER_lexicon')

    # use csv parser to read reviews
    documents_list = []
    with open('src/Restaurant_Reviews.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            documents_list.append(row)

    # use csv parser to read reviews
    test_query = None
    if test == 1:
        test_query = []
        with open('src/benchmark_queries', 'r') as f:
            for line in f:
                test_query.append(line.strip())

    # create analyzer for preprocessing
    analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter() | LemmaFilter() | StemFilter()
    # In questo codice:
    # RegexTokenizer suddivide il testo in token utilizzando espressioni regolari
    # LowercaseFilter converte tutti i token in minuscolo
    # StopFilter rimuove le parole di stop
    # LemmaFilter esegue la lemmatizzazione
    # StemFilter esegue lo stemming.

    # create schema
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True, analyzer=analyzer),
                    sentiment=NUMERIC(stored=True, numtype=float))

    # create index
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    output_file = 'output_file.txt'
    output_json = 'dcg_output.json'

    with open(output_file, 'w') as f:
        f.write("Search Restaurant Reviews\n\n")

        if test:
            run(f, documents_list, schema, 1, 1, test_query)
            run(f, documents_list, schema, 1, 2, test_query)
            run(f, documents_list, schema, 2, 1, test_query)
            run(f, documents_list, schema, 2, 2, test_query)
        else:
            run(f, documents_list, schema, ranking, sentiment, test_query)

        with open(output_json, 'w') as f2:
            print(dcg_output)
            f2.write(json.dumps(dcg_output, indent=4))
