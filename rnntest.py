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



import string
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



with open("wassem_hovy_naacl.csv", encoding="utf-8")as f:
    readCSV = csv.reader(f, delimiter="\t")
    for row in readCSV:
        tweet=strip(row[3]) #row[6]
        label=(row[4]) #row[5]
        tweet = tweet.lower()
        # label = label.encode('utf-8')
        for word in tweet.split():
            X_text.append(tweet.split())
            if label=="none":
                Y.append([1, 0,0])

            elif label=="racism":
                Y.append([0,1,0])

            else:
                Y.append([0,0,1])

            break


X_text=X_text[1:]
for tweet in X_text:
    cur=[]
    for words in tweet:
        if words in w2v:
            cur.append(w2v[words])
    X.append(cur)

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

Xtest = X[:1000, :]
X_active=X[6000:, :]
X = X[1000:6000, :]
Ytest = Y[:1000]
Y_active=Y[6000:, :]
Y = Y[1000:6000]
results=[]

def lstm(Xtest, Ytest, X_active, Y_active, X, Y, maxlen, results):

    timesteps=maxlen # number of words in a sentence
    input_dim=100 # dimension of embedding


    hidden_size=200

    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3,activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(X, Y, epochs=5)




    preds=(model.predict_classes(Xtest))

    print(model.evaluate(Xtest, Ytest, verbose=0))

    from sklearn.metrics import precision_recall_fscore_support
    ytrue=[]
    for y in Ytest:
        if y[0]==1:
            ytrue.append(0)
        elif y[1]==1:
            ytrue.append(1)
        else:
            ytrue.append(2)
    results.append(precision_recall_fscore_support(ytrue, preds))

    pred_active=model.predict(X_active[:1000, :])
    X_cur=X_active[:1000,:]
    # pred_class=model.predict_classes(X_active[:1000, :])
    # for i in range(len(pred_active)):
    #     print(pred_active[i], pred_class[i])
    X_active=X_active[1000:, :]
    ytrue_active=Y_active[:1000]
    Y_active=Y_active[1000:]

    for i in range(len(pred_active)):
        pred_active[i].sort()
        m1=pred_active[i][2]
        m2=pred_active[i][1]
        if m1-m2 < 0.25:
            X=np.append(X, [X_cur[i]], axis=0)
            Y = np.append(Y, [ytrue_active[i]], axis=0)
    return Xtest, Ytest, X_active, Y_active, X, Y


for i in range(9):
    Xtest, Ytest, X_active, Y_active, X, Y=lstm(Xtest, Ytest, X_active, Y_active, X, Y, maxlen, results)

print(results)
