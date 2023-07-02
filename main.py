from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import NumericRange, And

from LemmaFilter import LemmaFilter
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter, StemFilter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from SentimentFilter import SentimentFilter
from SentimentWeighting import SentimentWeighting
import os, os.path
import csv

if __name__ == '__main__':
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

    #create sentiment intensity analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()

    title = "Review"
    num_document = 1
    for document in documents_list:
        content = document[0]
        document_title = title + " " + str(num_document)
        num_document = num_document + 1

        polarity = sentiment_analyzer.polarity_scores(content)

        #add document to index
        writer.add_document(title=document_title, content=content, sentiment=float(polarity["compound"]))

    writer.commit()

    query = input("Insert a query: ")
    filter = input("Insert 1 for filter, any value for not filter: ")
    if filter == "1":
        filter_value = float(input("Insert -1 for negative reviews or 1 for positive reviews: "))
        sentiment_filter = SentimentFilter(filter_value)

    ix = open_dir("indexdir")
    searcher = ix.searcher(weighting=SentimentWeighting())
    parser = MultifieldParser(["content", "sentiment"], ix.schema)
    query = parser.parse(query)
    if(filter == "1"):
        query = And([query,sentiment_filter])
    results = searcher.search(query)
    print("Number of documents returned: " + str(len(results)))
    for result in results:
        print(result)