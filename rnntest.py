import numpy as np
import json
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

# import os
# os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop
import convert


# with open("glove.6B.50d.txt", "rb") as lines:
#     w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
#     # for count, (key, value) in enumerate(w2v.items()):
#     #     if (count > 5):
#     #         break
# print(w2v['b\'fairbanks-morse'])

def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding="utf-8")
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

w2v=loadGloveModel("glove.twitter.27B.100d.txt")

# In[35]:

def max_indicator(x):
    return (x == np.max(x)).astype(float)


def min_indicator(x):
    return (x == np.min(x)).astype(float)


# In[96]:


# class MeanEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         # if a text is empty we should return a vector of zeros
#         # with the same dimensionality as all the other vectors
#         self.dim = len(word2vec.itervalues().next())
#
#     def fit(self, X, y):
#         return self
#
#     # change to return x3
#     # column that has the most maxes
#     # column that has the most min
#     def transform(self, X):
#         return np.array([
#                             np.concatenate([
#
#                                 # mean of vectors
#                                 np.mean([self.word2vec[w] for w in words if w in self.word2vec],
#                                         # or [np.zeros(self.dim)],
#                                         axis=0),
#
#                                 # min of vectors
#                                 np.min([self.word2vec[w] for w in words if w in self.word2vec],
#                                        # or [np.zeros(self.dim)],
#                                        axis=0),
#
#                                 # max of vectors
#                                 np.max([self.word2vec[w] for w in words if w in self.word2vec],
#                                        # or [np.zeros(self.dim)],
#                                        axis=0),
#
#                                 # max of dimmensions
#                                 np.mean([max_indicator(self.word2vec[w]) for w in words if w in self.word2vec]
#                                         # or [np.zeros(self.dim)]
#                                         , axis=0),
#                                 # is currently returning the word's index (where the word has the maximum)
#
#                                 # min of dimmensions
#                                 np.mean([min_indicator(self.word2vec[w]) for w in words if w in self.word2vec]
#                                         # or [np.zeros(self.dim)]
#                                         , axis=0)],
#
#                             )
#                             for words in X
#
#                             # np.mean([self.word2vec[w] for w in words if w in self.word2vec]
#                             #        or [np.zeros(self.dim)], axis=0)
#                             # for words in X
#                             ])
#
#
# # In[97]:
#
#
#
# class TfidfEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         self.word2weight = None
#         #cur=next(iter(word2vec.values()))
#         self.dim = len(next(iter(word2vec.values())))
#
#     def fit(self, X, y):
#         tfidf = TfidfVectorizer(analyzer=lambda x: x)
#         tfidf.fit(X)
#         # if a word was never seen - it must be at least as infrequent
#         # as any of the known words - so the default idf is the max of
#         # known idf's
#         max_idf = max(tfidf.idf_)
#         self.word2weight = defaultdict(
#             lambda: max_idf,
#             [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
#
#         return self
#
#     def transform(self, X):
#         return np.array([
#                             np.concatenate([
#
#                                 # mean of vectors
#                                 np.mean([self.word2vec[w] * self.word2weight[w]
#                                          for w in words if w in self.word2vec], axis=0),
#
#                                 # min of vectors
#                                 np.min([self.word2vec[w] * self.word2weight[w]
#                                         for w in words if w in self.word2vec], axis=0),
#
#                                 # max of vectors
#                                 np.max([self.word2vec[w] * self.word2weight[w]
#                                         for w in words if w in self.word2vec], axis=0),
#
#                                 # max of dimmensions
#                                 np.mean([self.word2vec[w] * self.word2weight[w]
#                                          for w in words if w in self.word2vec], axis=0),
#                                 # is currently returning the word's index (where the word has the maximum)
#
#                                 # min of dimmensions
#                                 np.mean([self.word2vec[w] * self.word2weight[w]
#                                          for w in words if w in self.word2vec], axis=0)],
#
#                             )
#                             for words in X
#
#                             # np.mean([self.word2vec[w] for w in words if w in self.word2vec]
#                             #        or [np.zeros(self.dim)], axis=0)
#                             # for words in X
#                             ])


from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# '''
# etree_w2v = Pipeline([
#     ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
#     ("extra trees", ExtraTreesClassifier(n_estimators=200))]) #extratrees is from sklearn; try randomforest or gradient boosted
# '''
# etree_w2v = Pipeline([
#     ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
#     ("random forest",
#      RandomForestClassifier(n_estimators=200))])  # extratrees is from sklearn; try randomforest or gradient boosted
#
# etree_w2v_tfidf = Pipeline([
#     ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
#     ("extra trees", SVC())])  # you can use 1000


import convert, string
import emoji
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def strip(text):
    exclude = set(string.punctuation)
    text = ''.join(c for c in text if c not in exclude)
    return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)


X_text = []
X=[]
Y = []
# with open("hate_data.csv","rU") as f:
#     readCSV = csv.reader(f, delimiter=',')
#     for row in readCSV:
#         tweet=strip(row[19]) #row[6]
#         label=(row[5]) #row[5]
#         tweet = tweet.lower().decode('utf-8','ignore').encode("utf-8")
#         label = label.encode('utf-8')
#         for word in tweet.split():
#             if (word in w2v):
#                 X.append(tweet.split())
#                 if label=="The tweet uses offensive language but not hate speech":
#                     Y.append(1)
#                 elif label=="The tweet contains hate speech":
#                     Y.append(2)
#                 else:
#                     Y.append(0)
#                 break

# with open("labeled_data.csv","rU") as f:
#     readCSV = csv.reader(f, delimiter=',')
#     for row in readCSV:
#         tweet=strip(row[6]) #row[6]
#         label=(row[5]) #row[5]
#         tweet = tweet.lower()
#         label = label.encode('utf-8')
#         for word in tweet.split():
#             X_text.append(tweet.split())
#                 # if label=="The tweet uses offensive language but not hate speech":
#                 #     Y.append(1)
#             if label=="0":
#                 Y.append('1')
#                 # elif label=="1":
#                 #     Y.append(1)
#             else:
#                 Y.append('0')
#             # Y.append(label)
#
#             break


with open("wassem_hovy_naacl.csv", encoding="utf-8")as f:
    readCSV = csv.reader(f, delimiter="\t")
    for row in readCSV:
        tweet=strip(row[3]) #row[6]
        label=(row[4]) #row[5]
        tweet = tweet.lower()
        # label = label.encode('utf-8')
        for word in tweet.split():
            X_text.append(tweet.split())
                # if label=="The tweet uses offensive language but not hate speech":
                #     Y.append(1)
            # if label=="none":
            #     Y.append(0)
            #     # elif label=="1":
            #     #     Y.append(1)
            # else:
            #     Y.append(1)

            if label=="none":
                Y.append([1, 0,0])

            elif label=="racism":
                Y.append([0,1,0])

            else:
                Y.append([0,0,1])

            break


X_text=X_text[1:]
test_tweets=X_text[:200]
for tweet in X_text:
    cur=[]
    for words in tweet:
        if words in w2v:
            cur.append(w2v[words])
    X.append(cur)
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# tokenize = lambda doc: doc.lower().split(" ")
# sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
# sklearn_representation = sklearn_tfidf.fit_transform(X)
#
# print(len(sklearn_representation.toarray()), len(sklearn_representation.toarray()[0]))



# print(X[1090])
# for x in range(0, 10):
#    print(X[x])
#    print(Y[x])


# X=X[1:]
Y=Y[1:]

# add padding
maxlen=0
for line in X:
    if len(line)>maxlen:
        maxlen=len(line)

for line in X:
    if len(line)<maxlen:
        for i in range(maxlen-len(line)):
            line.append([0.0]*(100))

X=np.array(X)
Y=np.array(Y)

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


shuffle_in_unison_scary(X,Y)


Xtest=X[:200, :]
X=X[200:, :]
Ytest=Y[:200]
Y=Y[200:]



timesteps=31 # number of words in a sentence
input_dim=100 # dimension of embedding
features=100


hidden_size=50
# tokenize and build vocab
model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3,activation='sigmoid'))


# model = Sequential()
# model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(len(w2v))))
# model.add(Activation('softmax'))

optim=RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
model.fit(X, Y, epochs=5)



# test_X = ['The Palestinians are beasts walking on two legs'.split(),
#           'speech that is intended to offend, insult, intimidate, or threaten an individual or group based on a trait or attribute, such as sexual orientation, religion, color, gender, or disability'.split(),
#           ['purple'],
#           ['trump'],
#           'Wipe out the Jews'.split(),
#           'Women are like grass, they need to be beaten/cut regularly'.split(),
#           '8 mins til JesusChristSuperstarLive!! I cant wait to learn about Jesus and superstars'.split(),
#           'seen this one, the popsicle sticks are a keyboard. He moved to  to follow his. Wait, are they from other shows?'.split(),
#           'fuck the muslims'.split(),
#           'women are stupid'.split(),
#           'you jew fag lmao'.split()]



# test=[]
# for tweet in test_X:
#     cur=[]
#     for words in tweet:
#         if words in w2v:
#             cur.append(w2v[words])
#     if len(cur)<32:
#         for i in range(32-len(cur)):
#             cur.append([0.0]*100)
#     test.append(cur)
# test_X=np.array(test)


preds=(model.predict_classes(Xtest))
for i in range(len(preds)):
    print(test_tweets[i], "\t", preds[i], "\t", Ytest[i])

print(model.evaluate(Xtest, Ytest, verbose=0))

from sklearn.metrics import precision_recall_fscore_support
ytrue=[]
ypred=[]
for y in Ytest:
    if y==[1,0,0]:
        ytrue.append(0)
    if y==[0,1,0]:
        ytrue.append(1)
    if y==[0,0,1]:
        ytrue.append(2)


print(precision_recall_fscore_support(ytrue, preds))