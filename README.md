
# Search Engine for Restaurant Reviews

This is a Search Engine for Natural Language Processing on a set of restaurant reviews.


## Dependencies

These dependencies are mandatory:

    whoosh
    nltk
    textblob
    

## Quick Start


- Clone the repo;
- Install the dependencies;
- Run main.py to start the program
- Insert any query;
- Check output_file.txt to see the results;

## Parameters

The default sentiment analysis tool is VADER, the default IR is Tf-Idf, the default ranking function is Naive. You can change those with parameters.

- -s (--sentiment): Indicates the Sentiment Analysis tool to use (1 - VADER, 2 - Textblob), default: 1
- -i (--irmodel): Indicates the IR Model to be used for the search (1 - TF-IDF (vector space model), 2-BM25 (probabilistic model)), default: 1
- -r (--ranking): Indicates the ranking function to use to combine the result with the sentiment analysis (1-naive, 2-weighted_avg, 3-balanced_weighted_avg), default: 1
- -t (--test): Indicates that the script should be activated in test mode (benchmark)
## Dataset

The dataset is a tsv file containing 1000 restaurant reviews, with a score of 0/1 indicating whether the review is negative or positive. This score has not been used to leverage our sentiment analysis techniques.
## Benchmark

We extracted a random sample of 37 reviews in the file src/dcg_sample.txt, then we manually annotated a level of "satisfaction" in a [0-3] range for each review for each query present in the file src/benchmark_queries, based on the user original need (pertinence and sentiment are both considered).

The result of the benchmark is produced in the dcg_output.json file.
For each query we try every possible combination of sentiment analysis technique, IR model and ranking function among those we have implemented. Then we calculate the optimal DCG and the one calculated on the result of the query, to see which combination produces the closest DCG to the optimal one.