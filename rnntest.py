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
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS

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



import convert, string
import emoji
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict




X_text = []
X=[]
Y = []
Y_raw=[]


with open("wassem_hovy_naacl.csv", encoding="utf-8")as f:
    readCSV = csv.reader(f, delimiter="\t")
    for row in readCSV:
        tweet=row[3] #row[6]
        label=(row[4]) #row[5]
        tweet = tweet.lower()
        # label = label.encode('utf-8')
        splitted=tweet.split()
        to_append=[]
        for tw in splitted:
            if tw not in punctuation and "@" not in tw and tw[:4]!='http': #and tw not in STOPWORDS
                to_append.append(tw)
        X_text.append(to_append)
        # if label=="none":
        #     Y.append([1, 0,0])
        # elif label=="racism":
        #     Y.append([0,1,0])
        #
        # else:
        #     Y.append([0,0,1])

        if label=="none":
            Y.append(0)
            Y_raw.append(0)
        elif label=="sexism":
            Y.append(1)
            Y_raw.append(1)
        elif label=="racism":
            Y.append(1)
            Y_raw.append(2)



# using pretrained glove file
X_text=X_text[1:]
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

def shuffle_in_unison(a, b, c,d):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
    np.random.set_state(rng_state)
    np.random.shuffle(d)


shuffle_in_unison(X,Y, X_text, Y_raw)


Xtest=X[:500, :]
X=X[500:, :]
Ytest=Y[:500]
Y=Y[500:]
X_text=X_text[:500]



timesteps=maxlen # number of words in a sentence
input_dim=100 # dimension of embedding
features=100


hidden_size=200
# tokenize and build vocab
model = Sequential()
model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
model.add(Dropout(0.5))
# model.add(LSTM(hidden_size))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


# model = Sequential()
# model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(len(w2v))))
# model.add(Activation('softmax'))

optim=RMSprop(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X, Y, epochs=8)



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
    print(X_text[i], "\t", preds[i], "\t", Ytest[i])

print(model.evaluate(Xtest, Ytest, verbose=0))

from sklearn.metrics import precision_recall_fscore_support
# ytrue=[]
# ypred=[]
# for y in Ytest:
#     if y[0]==1:
#         ytrue.append(0)
#     elif y[1]==1:
#         ytrue.append(1)
#     else:
#         ytrue.append(2)


print(precision_recall_fscore_support(Ytest, preds))