from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer
from LemmaFilter import LemmaFilter
import nltk

import os, os.path
import csv

if __name__ == '__main__':
    nltk.download('wordnet')

    print("Search Restaurant Reviews\n")

    #use csv parser to read reviews
    documents_list = []
    with open('src/Restaurant_Reviews.tsv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            documents_list.append(row)

    #create analyzer for preprocessing
    analyzer = StandardAnalyzer() | LemmaFilter()

    #create schema
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True, analyzer=analyzer))

    #create index
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    ix = create_in("indexdir", schema)

    #create writer
    writer = ix.writer()

    title = "Review"
    num_document = 1
    for document in documents_list:
        content = document[0]
        document_title = title + " " + str(num_document)
        num_document = num_document + 1

        #add document to index
        writer.add_document(title=document_title, content=content)

    writer.commit()

    query = input("Insert a query: ")

    ix = open_dir("indexdir")
    searcher = ix.searcher()
    parser = QueryParser("content", ix.schema)
    query = parser.parse(query)
    results = searcher.search(query)
    print("Number of documents returned: " + str(len(results)))
    for result in results:
        print(result)
