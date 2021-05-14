import numpy as np
from collections import namedtuple, Counter
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from scipy.sparse import vstack, hstack
from nltk.corpus import stopwords
import math

#params here

ngram_lower=1
ngram_upper=3

######################################################################

print('Reading embeddings')

def read_embeddings(filename):
    X=[]
    with open(filename) as file:
        for line in file:
            vector = line.split()[1:]
            vector = np.array(vector)
            vector = vector.astype(np.float64)
            X.append(vector)
    return np.array(X)

X_train_embedding = read_embeddings('train_vectors.txt')
X_test_embedding = read_embeddings('test_vectors.txt')

#######################################################################

print('Reading documents')

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
alldocs = []  # will hold all docs in original order
filename='alldata-id_p1gram.txt'

with open(filename, encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = line.split()
        words = tokens[1:]
        tags = [line_no] # `tags = [tokens[0]]` would also work at extra memory cost
        split = ['train','test','extra','extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))
    
train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']

########################################################################################

print('Extracting features')

count_vect = CountVectorizer(tokenizer=lambda text: text ,preprocessor=lambda text:text, binary=True,ngram_range=(ngram_lower,ngram_upper))
X_train_NB = count_vect.fit_transform([x.words for x in train_docs])
Y_train = [doc.sentiment for doc in train_docs]


print('Calculating probabilities')

nb=naive_bayes.BernoulliNB()
nb.fit(X_train_NB,Y_train)
prob=nb.feature_log_prob_ #index 0 is positive

r=prob[0]-prob[1]

print('Weighing features')
X_train=[x.multiply(r).tocsr() for x in X_train_NB]
X_train=vstack(X_train)

X_test_pre=count_vect.transform([x.words for x in test_docs])
X_test=[x.multiply(r).tocsr() for x in X_test_pre]
X_test=vstack(X_test)
############################################################################################

X_train = hstack((X_train,X_train_embedding))
X_test = hstack((X_test,X_test_embedding))

print('Training classifier')

svc=linear_model.LogisticRegression()
svc.fit(X_train,Y_train)

print('Testing classifier')

high_file = open("high_confidence.txt", "w")
low_file = open("low_confidence.txt", "w")
incorrect_file = open("incorrect.txt", "w")

high_threshold = 0.9
low_threshold = 0.6

predictions = svc.predict(X_test)
class_probs = svc.predict_proba(X_test)

# Y_test contains the actual labels
Y_test=[doc.sentiment for doc in test_docs]

highWords = Counter()
lowWords = Counter()
incorrectWords = Counter()

probBuckets = [0]*21
incorrectProbBuckets = [0]*21

for i, sentiment in enumerate(Y_test):

    # string = str(i) + " " + str(sentiment) + " " + str(predictions[i]) + " " + str(class_probs[i]) + "\n"
    # string += " ".join(test_docs[i].words) + "\n"

    if (sentiment != predictions[i]):
        # incorrectly labeled
        # incorrect_file.write(string)
        incorrectWords += Counter(test_docs[i].words)

        bucket = math.floor(max(class_probs[i])*20)
        incorrectProbBuckets[bucket] += 1

    elif (max(class_probs[i]) > high_threshold):
        # high confidence
        # high_file.write(string)
        highWords += Counter(test_docs[i].words)
        
        bucket = math.floor(max(class_probs[i])*20)
        probBuckets[bucket] += 1

    elif (max(class_probs[i]) < low_threshold):
        # low confidence
        # low_file.write(string)
        lowWords += Counter(test_docs[i].words)
        
        bucket = math.floor(max(class_probs[i])*20)
        probBuckets[bucket] += 1
    
    else:
        bucket = math.floor(max(class_probs[i])*20)
        probBuckets[bucket] += 1


for d in [highWords, lowWords, incorrectWords]:
    for k, v in d.most_common(100):
        if k in stopwords.words() or k in ' !#$%&"()*+, -./:;<=>?@[\]^_`{|}~':
            del d[k]

topWords = Counter()    
for d in [highWords, lowWords, incorrectWords]:
    for k, v in d.most_common(40):
        if k in topWords.keys():
            topWords[k] += 1
        else:
            topWords[k] = 1

for key in topWords.keys():
    if topWords[key] > 1:
        print(key)
        for d in [highWords, lowWords, incorrectWords]:
            del d[key]

print('Top Five High Confidence Words', highWords.most_common(10))
print('Top Five Low Confidence Words', lowWords.most_common(10))
print('Top Five Incorrect Words', incorrectWords.most_common(10))
print('Counts', probBuckets)
print('Incorrect Counts', incorrectProbBuckets)
print('Accuracy=',svc.score(X_test,Y_test)*100)