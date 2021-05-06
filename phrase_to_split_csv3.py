"""
Put all the Stanford Sentiment Treebank phrase data into test, training, and dev CSVs.
Socher, R., Perelygin, A., Wu, J. Y., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). Recursive Deep Models
for Semantic Compositionality Over a Sentiment Treebank. Presented at the Conference on Empirical Methods in Natural
Language Processing EMNLP.
https://nlp.stanford.edu/sentiment/
https://gist.github.com/wpm/52758adbf506fd84cff3cdc7fc109aad
"""

import os
import sys
import csv
import pandas


gram_split = '@$'
# string -> string of only 2grams 
# string of 1grams separated by spaces
def to_bigrams(sentence):
    split_sent = sentence.split()
    split_bigram =  list(zip(split_sent[:-1], split_sent[1:]))
    bigram_sentences = list(map(lambda x: gram_split.join(x), split_bigram))
    return " ".join(bigram_sentences)

def to_trigrams(sentence):
    split_sent = sentence.split()
    split_trigram =  list(zip(split_sent[:-1], split_sent[1:], split_sent[2:]))
    trigram_sentences = list(map(lambda x: gram_split.join(x), split_trigram))
    return " ".join(trigram_sentences)


fine_sentiment_arr = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
    dictionary = dictionary.set_index("id")

    sentiment_labels = pandas.read_csv(os.path.join(base_directory, "sentiment_labels.txt"), sep="|")
    sentiment_labels.columns = ["id", "sentiment"]
    sentiment_labels = sentiment_labels.set_index("id")

    phrase_sentiments = dictionary.join(sentiment_labels)

    phrase_sentiments["fine"] = pandas.cut(phrase_sentiments.sentiment, fine_sentiment_arr, 
                                           include_lowest=True,
                                           labels=["very negative", "negative", "neutral", "positive", "very positive"])
    phrase_sentiments["coarse"] = phrase_sentiments.fine.apply(group_labels)
    return phrase_sentiments


def get_sentence_partitions(base_directory):
    sentences = pandas.read_csv(os.path.join(base_directory, "datasetSentences.txt"), index_col="sentence_index",
                                sep="\t")
    splits = pandas.read_csv(os.path.join(base_directory, "datasetSplit.txt"), index_col="sentence_index")
    return sentences.join(splits).set_index("sentence")


def partition(base_directory):
    phrase_sentiments = get_phrase_sentiments(base_directory)
    sentence_partitions = get_sentence_partitions(base_directory)
    # noinspection PyUnresolvedReferences
    data = phrase_sentiments.join(sentence_partitions, on="phrase", how='right')
    
    data["splitset_label"] = data["splitset_label"].fillna(1).astype(int)

    data["phrase"] = data["phrase"].str.replace(r"\s('s|'d|'re|'ll|'m|'ve|n't)\b", lambda m: m.group(1))
    # This actually does drop a bunch..... 
    data = data[['phrase', 'coarse', 'splitset_label']].dropna(axis=0)
    data['splitset_label'] = data['splitset_label'].map(lambda x: int (x>1))
    data.index = data.index.astype(int)
    return data.groupby("splitset_label")

test = True
show_sentiment = False
base_directory, output_directory, gram = sys.argv[1:4]
os.makedirs(output_directory, exist_ok=True)
for splitset, partition in partition(base_directory):
    split_name = {0: "train", 1: "test"}[splitset]
    
    filename = os.path.join(output_directory, "alldata-id_p"+gram + "gram.txt")
    # adding bigrams 
    partition['bigram'] = partition['phrase'].map(to_bigrams)

    # adding trigrams
    partition['trigram'] = partition['phrase'].map(to_trigrams)
    
    # removing neturals 
    partition2 = partition[partition['coarse'] != 'neutral'] 

    # grouping them --> negative, then positive
    n, p = partition2.groupby('coarse', sort=True)
    partition2 = pandas.concat([n[1], p[1]])
    if gram == "1":
        partition2['full'] = partition2['phrase'] # + partition2['bigram']  + partition2['trigram']
    elif gram == "2":
        partition2['full'] = partition2['phrase'] + partition2['bigram'] # + partition2['trigram']
    elif gram == "3":
        partition2['full'] = partition2['phrase'] + partition2['bigram'] + partition2['trigram']
    else:
        print("error with gram input")
    if test: 
        print('\n', split_name, len(partition2))
        print(partition2.groupby(['coarse', 'splitset_label']).count())
    partition2[['full']].to_csv(filename, sep='\t', quoting=csv.QUOTE_NONE, escapechar="\\", header=False, mode='a')

