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

def calculate_dcg(ordered_documents, num_query):
    value = 0
    count = 1
    for d in ordered_documents:
        value = value + float(2 ** d[num_query] - 1) / log(1+count)
        count = count+1
    return value
def run_query(f, text_query, filter, irmodel, sentiment, ranking, sentiment_filter=None):
    f.write("\nQuery to analize: "+text_query)
    ix = open_dir("indexdir")

    if text_query not in dcg_output:
        dcg_output[text_query] = []

    # Crea un oggetto di scoring
    scoringObject = SentimentWeighting(irmodel, ranking)

    searcher = ix.searcher(weighting=scoringObject)
    parser = QueryParser("content", ix.schema)
    query = parser.parse(text_query)

    if filter == "1":
        query = And([query, sentiment_filter])
    results = searcher.search(query)

    f.write("\nNumber of documents returned: " + str(len(results)) + "\n")
    for hit in results:
        f.write("Document num: "+str(hit["docnum"])+"\n")
        f.write("Content: "+hit["content"]+"\n")
        f.write("Sentiment: "+str(hit["sentiment"])+"\n")
        f.write("Weight: "+str(hit.score)+"\n")
        f.write("-------------------------"+"\n")

    return results

def run(f, documents_list, schema, irmodel=1, sentiment=1, ranking = 1, test_query=None, valutations_list=None):

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
    if(irmodel == 2):
        f.write("BM25F\n")
    else:
        f.write("TF_IDF\n")

    num_document = 0
    for content in documents_list:

        # calculate sentiment and add document to index
        if sentiment == 2:
            polarity = TextBlob(content)
            writer.add_document(docnum=num_document, content=content, sentiment=float(polarity.sentiment.polarity))
        else:
            polarity = sentiment_analyzer.polarity_scores(content)
            writer.add_document(docnum=num_document, content=content, sentiment=float(polarity["compound"]))
        num_document = num_document + 1

    writer.commit()

    if test_query is None:
        query = input("Insert a query: ")
        query = query.strip()
        filter = input("Insert 1 for filter, any value for not filter: ")
        sentiment_filter = None
        if filter == "1":
            filter_value = int(input("Insert -1 for negative reviews or 1 for positive reviews: "))
            sentiment_filter = SentimentFilter(filter_value)

        results = run_query(f, query, filter, irmodel,sentiment,ranking,sentiment_filter)
    else:
        num_query = 0
        for query in test_query:
            results = run_query(f, query, "0", irmodel, sentiment, ranking)
            valutations_list_sorted = sorted(valutations_list, key=lambda x:x[num_query], reverse=True)[:9]
            results_sorted = []
            for x in results:
                results_sorted.append(valutations_list[x["docnum"]])

            optimal_dcg = calculate_dcg(valutations_list_sorted, num_query)
            calculated_dcg = calculate_dcg(results_sorted, num_query)

            dcg_output[query].append({"sentiment": "VADER" if sentiment == 1 else "TextBlob", "irmodel": "TF-IDF" if irmodel == 1 else "BM25F", "ranking_function": "Naive" if ranking == 1 else "Average Weighted", "optimal_dcg": optimal_dcg, "calculated_dcg": calculated_dcg})
            num_query = num_query+1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search Restaurant Reviews")
    parser.add_argument('-t', '--test', default=0, type=int, help="Se presente, indica che lo script deve essere attivato in modalità test")
    parser.add_argument('-s', '--sentiment', default=1, type=int, help="Indica la libreria di Sentiment Analysis da utilizzare (1-VADER, 2-Textblob), default: VADER")
    parser.add_argument('-i', '--irmodel', default=1, type=int, help="Indica l'IR Model da utilizzare per la ricerca (1-TF-IDF (modello vettoriale), 2-BM25 (modello probabilistico)), default: TF-IDF")
    parser.add_argument('-r', '--ranking', default=1, type=int, help="Indica la funzione di ranking da utilizzare per combinare il risultato con la sentiment analysis (1-naive, 2-weighted_avg, 3-balanced_weighted_avg), default: NAIVE")
    args = parser.parse_args()

    sentiment = int(args.sentiment)
    irmodel = int(args.irmodel)
    test = int(args.test)
    ranking = int(args.ranking)

    nltk.download('wordnet')
    nltk.download('VADER_lexicon')

    test_query = None
    documents_list = []
    valutations_list = None

    if test != 1:
        # use csv parser to read reviews
        with open('src/Restaurant_Reviews.tsv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                documents_list.append(row[0])
    else:
        valutations_list = []
        num_document = 0
        test_query = []
        with open('src/benchmark_queries', 'r') as f:
            for line in f:
                test_query.append(line.strip())

        with open('src/dcg_sample.txt', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                documents_list.append(row[0])
                valutations_list.append(list(map(lambda x:int(x), row[1:])))
                num_document = num_document + 1


    # create analyzer for preprocessing
    analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter() | LemmaFilter() | StemFilter()
    # In questo codice:
    # RegexTokenizer suddivide il testo in token utilizzando espressioni regolari
    # LowercaseFilter converte tutti i token in minuscolo
    # StopFilter rimuove le parole di stop
    # LemmaFilter esegue la lemmatizzazione
    # StemFilter esegue lo stemming.

    # create schema
    schema = Schema(docnum=NUMERIC(numtype=int, stored=True), content=TEXT(stored=True, analyzer=analyzer),
                    sentiment=NUMERIC(stored=True, numtype=float))

    # create index
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")

    output_file = 'output_file.txt'
    output_json = 'dcg_output.json'

    with open(output_file, 'w') as f:
        f.write("Search Restaurant Reviews\n\n")

        if test:
            run(f, documents_list, schema, 1, 1, 1, test_query, valutations_list)
            run(f, documents_list, schema, 1, 2, 1, test_query, valutations_list)
            run(f, documents_list, schema, 2, 1, 1, test_query, valutations_list)
            run(f, documents_list, schema, 2, 2, 1, test_query, valutations_list)
            run(f, documents_list, schema, 1, 1, 2, test_query, valutations_list)
            run(f, documents_list, schema, 1, 2, 2, test_query, valutations_list)
            run(f, documents_list, schema, 2, 1, 2, test_query, valutations_list)
            run(f, documents_list, schema, 2, 2, 2, test_query, valutations_list)
        else:
            run(f, documents_list, schema, irmodel, sentiment, ranking, test_query)

        with open(output_json, 'w') as f2:
            f2.write(json.dumps(dcg_output, indent=4))
