"""
Put all the Stanford Sentiment Treebank phrase data into test, training, and dev CSVs.
Socher, R., Perelygin, A., Wu, J. Y., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). Recursive Deep Models
for Semantic Compositionality Over a Sentiment Treebank. Presented at the Conference on Empirical Methods in Natural
Language Processing EMNLP.
https://nlp.stanford.edu/sentiment/
https://gist.github.com/wpm/52758adbf506fd84cff3cdc7fc109aad

TODO: 1) order and put all positive sentences up top (first half), 2) remove the sentiment, fine and coarse columns, 
3) combine train, test
"""

import os
import sys

import pandas


def get_phrase_sentiments(base_directory):
    def group_labels(label):
        if label in ["very negative", "negative"]:
            return "negative"
        elif label in ["positive", "very positive"]:
            return "positive"
        else:
            return "neutral"

    dictionary = pandas.read_csv(os.path.join(base_directory, "dictionary.txt"), sep="|")
    dictionary.columns = ["phrase", "id"]
    # column headers: id, phrase
    dictionary = dictionary.set_index("id")

    sentiment_labels = pandas.read_csv(os.path.join(base_directory, "sentiment_labels.txt"), sep="|")
    # column headers: phrase_ids, sentiment_labels
    sentiment_labels.columns = ["id", "sentiment"]
    sentiment_labels = sentiment_labels.set_index("id")

    # column headers: id, phrase, sentiment
    phrase_sentiments = dictionary.join(sentiment_labels)

    phrase_sentiments["fine"] = pandas.cut(phrase_sentiments.sentiment, [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                           include_lowest=True,
                                           labels=["very negative", "negative", "neutral", "positive", "very positive"])
    phrase_sentiments["coarse"] = phrase_sentiments.fine.apply(group_labels)
    # column headers: id, phrase, sentiment, fine, coarse
    return phrase_sentiments


def get_sentence_partitions(base_directory):
    # use sentence_index as row label of DataFrame
    sentences = pandas.read_csv(os.path.join(base_directory, "datasetSentences.txt"), index_col="sentence_index",
                                sep="\t")
    # splitset_label i.e. 1,2,3
    splits = pandas.read_csv(os.path.join(base_directory, "datasetSplit.txt"), index_col="sentence_index")
    # join 'sentences' with 'splits' and make "sentence" the index
    return sentences.join(splits).set_index("sentence")

# 
def partition(base_directory):
    # column headers: id, phrase, sentiment, fine, coarse
    phrase_sentiments = get_phrase_sentiments(base_directory)
    # column headers: sentence, sentence_index, splitset_label 
    sentence_partitions = get_sentence_partitions(base_directory)
    # noinspection PyUnresolvedReferences
    # use phrase in phrase_sentiments to join on sentence in sentence_partitions
    # column_headers: id, phrase, sentiment, fine, coarse, sentence, sentence_index, splitset_label 
    data = phrase_sentiments.join(sentence_partitions, on="phrase", how="right")
    data["splitset_label"] = data["splitset_label"].fillna(1).astype(int)
    data["phrase"] = data["phrase"].str.replace(r"\s('s|'d|'re|'ll|'m|'ve|n't)\b", lambda m: m.group(1))
    return data.groupby("splitset_label")


base_directory, output_directory = sys.argv[1:3]
os.makedirs(output_directory, exist_ok=True)
for splitset, partition in partition(base_directory):
    split_name = {1: "train", 2: "test", 3: "dev"}[splitset]
    filename = os.path.join(output_directory, "stanford-sentiment-treebank.%s.csv" % split_name)
    del partition["splitset_label"]
    partition.to_csv(filename)