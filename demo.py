import numpy as np
import json
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

from reveiw dog import dog

with open("glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}
    for count, (key, value) in enumerate(w2v.iteritems()):
        if (count > 5):
            break
            # print key,value


# In[35]:


def max_indicator(x):
    return (x == np.max(x)).astype(float)


def min_indicator(x):
    return (x == np.min(x)).astype(float)


# In[96]:


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    # change to return x3
    # column that has the most maxes
    # column that has the most min
    def transform(self, X):
        return np.array([
                            np.concatenate([

                                # mean of vectors
                                np.mean([self.word2vec[w] for w in words if w in self.word2vec],
                                        # or [np.zeros(self.dim)],
                                        axis=0),

                                # min of vectors
                                np.min([self.word2vec[w] for w in words if w in self.word2vec],
                                       # or [np.zeros(self.dim)],
                                       axis=0),

                                # max of vectors
                                np.max([self.word2vec[w] for w in words if w in self.word2vec],
                                       # or [np.zeros(self.dim)],
                                       axis=0),

                                # max of dimmensions
                                np.mean([max_indicator(self.word2vec[w]) for w in words if w in self.word2vec]
                                        # or [np.zeros(self.dim)]
                                        , axis=0),
                                # is currently returning the word's index (where the word has the maximum)

                                # min of dimmensions
                                np.mean([min_indicator(self.word2vec[w]) for w in words if w in self.word2vec]
                                        # or [np.zeros(self.dim)]
                                        , axis=0)],

                            )
                            for words in X

                            # np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                            #        or [np.zeros(self.dim)], axis=0)
                            # for words in X
                            ])


# In[97]:


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                            np.concatenate([

                                # mean of vectors
                                np.mean([self.word2vec[w] * self.word2weight[w]
                                         for w in words if w in self.word2vec], axis=0),

                                # min of vectors
                                np.min([self.word2vec[w] * self.word2weight[w]
                                        for w in words if w in self.word2vec], axis=0),

                                # max of vectors
                                np.max([self.word2vec[w] * self.word2weight[w]
                                        for w in words if w in self.word2vec], axis=0),

                                # max of dimmensions
                                np.mean([self.word2vec[w] * self.word2weight[w]
                                         for w in words if w in self.word2vec], axis=0),
                                # is currently returning the word's index (where the word has the maximum)

                                # min of dimmensions
                                np.mean([self.word2vec[w] * self.word2weight[w]
                                         for w in words if w in self.word2vec], axis=0)],

                            )
                            for words in X

                            # np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                            #        or [np.zeros(self.dim)], axis=0)
                            # for words in X
                            ])


from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

'''
etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))]) #extratrees is from sklearn; try randomforest or gradient boosted
'''
etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("random forest",
     RandomForestClassifier(n_estimators=200))])  # extratrees is from sklearn; try randomforest or gradient boosted

etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", SVC())])  # you can use 1000

import convert, string
import emoji
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def strip(text):
    exclude = set(string.punctuation)
    text = ''.join(c for c in text if c not in exclude)
    return ''.join(c for c in text if c not in emoji.UNICODE_EMOJI)


X = []
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

with open("labeled_data.csv","rU") as f:
    readCSV = csv.reader(f, delimiter=',')
    for row in readCSV:
        tweet=strip(row[6]) #row[6]
        label=(row[5]) #row[5]
        tweet = tweet.lower().decode('utf-8','ignore').encode("utf-8")
        label = label.encode('utf-8')
        for word in tweet.split():
            if (word in w2v):
                X.append(tweet.split())
                # if label=="The tweet uses offensive language but not hate speech":
                #     Y.append(1)
                if label=="0":
                    Y.append(1)
                # elif label=="1":
                #     Y.append(1)
                else:
                    Y.append(0)
                break

# with open('test.json', 'r') as input_file:
#     for line in input_file:
#         data = json.loads(line.strip())
#         # print data
#         tweet = data['tweet']
#         tweet = strip(tweet)
#         tweet = tweet.lower().encode('utf-8')
#         label = data['class'].encode('utf-8')
#         for word in tweet.split():
#             if (word in w2v and label != ""):
#                 X.append(tweet.split())
#                 Y.append(label)
#                 break


# with open('jamesjson.json', 'r') as input_file:
#     for line in input_file:
#         data = json.loads(line)
#         # print data
#         tweet = data['text']
#         tweet = strip(tweet)
#         tweet = tweet.lower().encode('utf-8')
#         label = data['label'].encode('utf-8')
#         for word in tweet.split():
#             if (word in w2v and label != ""):
#                 X.append(tweet.split())
#                 if (label == "None " or label == "noen" or label == "None"):
#                     # print("/here")
#                     Y.append(0)
#
#                 else:
#                     Y.append(1)
#                 break
#
# with open('mattjson.json', 'r') as input_file:
#     for line in input_file:
#         data = json.loads(line)
#         # print data
#         tweet = data['text']
#         tweet = strip(tweet)
#         tweet = tweet.lower().encode('utf-8')
#         label = data['label'].encode('utf-8')
#         for word in tweet.split():
#             if (word in w2v and label != ""):
#                 X.append(tweet.split())
#                 if (label == "None " or label == "noen" or label == "None"):
#                     # print("/here")
#                     Y.append(0)
#
#                 else:
#                     Y.append(1)
#                 break
#
# with open('trentjson.json', 'r') as input_file:
#     for line in input_file:
#         data = json.loads(line)
#         # print data
#         tweet = data['text']
#         tweet = strip(tweet)
#         tweet = tweet.lower().encode('utf-8')
#         label = data['label'].encode('utf-8')
#         for word in tweet.split():
#             if (word in w2v and label != ""):
#                 X.append(tweet.split())
#                 if (label == "None " or label == "noen" or label == "None"):
#                     # print("/here")
#                     Y.append(0)
#
#                 else:
#                     Y.append(1)
#                 break
#
# with open('yaxinjson.json', 'r') as input_file:
#     for line in input_file:
#         data = json.loads(line)
#         # print data
#         tweet = data['text']
#         tweet = strip(tweet)
#         tweet = tweet.lower().encode('utf-8')
#         label = data['label'].encode('utf-8')
#         for word in tweet.split():
#             if (word in w2v and label != ""):
#                 X.append(tweet.split())
#                 if (label == "None " or label == "noen" or label == "None"):
#                     # print("/here")
#                     Y.append(0)
#
#                 else:
#                     Y.append(1)
#                 break

# print(X[1090])
# for x in range(0, 10):
#    print(X[x])
#    print(Y[x])

X=X[1:]
Y=Y[1:]

score=cross_val_score(etree_w2v_tfidf, X, Y, cv=5)
print(score)

#etree_w2v.fit(X, Y)
# etree_w2v_tfidf.fit(X, Y)


# for i in range(5):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
#     etree_w2v_tfidf.fit(X, Y)
#     y_pred=etree_w2v_tfidf.predict(X_test)
#     print("Recall:" , recall_score(Y_test, y_pred))
#     print("Precision ", average_precision_score(Y_test, y_pred))

test_X = ['The Palestinians are beasts walking on two legs'.split(),
          'speech that is intended to offend, insult, intimidate, or threaten an individual or group based on a trait or attribute, such as sexual orientation, religion, color, gender, or disability'.split(),
          ['purple'],
          ['trump'],
          'Wipe out the Jews'.split(),
          'Women are like grass, they need to be beaten/cut regularly'.split(),
          '8 mins til JesusChristSuperstarLive!! I cant wait to learn about Jesus and superstars'.split(),
          'seen this one, the popsicle sticks are a keyboard. He moved to  to follow his. Wait, are they from other shows?'.split(),
          'fuck the muslims'.split(),
          'women are stupid'.split(),
          'you jew fag lmao'.split()]

testing = [
    'I thought field niggers were smarter than house niggers. Instead of ploughing the field yall gonna be stunting. I guess JZ was right in saying'.split()]

# print etree_w2v.classes_
# print(
#     "The following is the probabilities for each category of this sentence:\n\nI thought field niggers were smarter than house niggers. Instead of ploughing the field yall gonna be stunting. I guess JZ was right in saying\n")
#
# print("This does not use word-frequencies")
# print(pd.DataFrame(etree_w2v.predict_proba(testing), columns=etree_w2v.classes_))
# print("\n")
# print("\n")
# print etree_w2v.predict_proba(testing)

# print("This does use word-frequencies (tfidf)\n")
# print(pd.DataFrame(etree_w2v_tfidf.predict_proba(testing), columns=etree_w2v_tfidf.classes_))
# print("\n")
# print("\n")
#
# print("The following are the probabilites for each category of these sentences:")
# print("\n")
# print("0: The Palestinians are beasts walking on two legs")
# print(
#     "1: 'speech that is intended to offend, insult, intimidate, or threaten an individual or group based on a trait or attribute, such as sexual orientation, religion, color, gender, or disability")
# print("2: purple")
# print("3: trump")
# print("4: Wipe out the Jews")
# print("5: Women are like grass, they need to be beaten/cut regularly")
# print("6: 8 mins til JesusChristSuperstarLive!! I cant wait to learn about Jesus and superstars")
# print(
#     "7: seen this one, the popsicle sticks are a keyboard. He moved to  to follow his. Wait, are they from other shows?")
#
# print("\n")
# print("\n")
# print("this is the frame for no frequencies")
# print(pd.DataFrame(etree_w2v.predict_proba(test_X), columns=etree_w2v.classes_))
# print("this is the frame for frequencies")
# print(pd.DataFrame(etree_w2v_tfidf.predict_proba(test_X), columns=etree_w2v.classes_))

print etree_w2v_tfidf.predict(test_X)
# print etree_w2v.predict(X)
#print etree_w2v.predict_proba(X)
print etree_w2v_tfidf.predict_proba(test_X)
