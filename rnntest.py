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
maxlen=60
for line in X:
    if len(line)<maxlen:
        for i in range(maxlen-len(line)):
            line.append([0.0]*(100))

X=np.array(X)
Y=np.array(Y)

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


shuffle_in_unison(X,Y)

Xtest = X[:1000, :]
X = X[1000:, :]
Ytest = Y[:1000]
Y = Y[1000:]
results=[]


def lstm(Xtest, Ytest, X, Y, maxlen, results, w2v, file, iteration):
    iter_size=2000
    X_active=[]
    start=(iteration-1)*iter_size+1
    end=iteration*iter_size+1
    with open(file, encoding='utf-8') as input_file:
        CSV = csv.reader(input_file, delimiter=",")
        i=0
        for line in CSV:
            if i>=start and i<end:
                tweet=line[19]
                tweets=tweet.lower().split()
                cur=[]
                for words in tweets:
                    if words in w2v:
                        cur.append(w2v[words])
                X_active.append(cur)
            i+=1

    for lines in X_active:
        if len(lines) < maxlen:
            for i in range(maxlen - len(lines)):
                lines.append([0.0] * (100))

    X_active = np.array(X_active)

    if iteration:
        with open("active.txt", "r+") as newInput:
            for line in newInput:
                tweet=line.split("\t")[0]
                label=line.split("\t")[1]
                cur=[]
                if label == 'sexist':
                    Y.append([0, 0, 1])
                elif label == 'racist':
                    Y.append([0, 1, 0])
                else:
                    Y.append([1, 0, 0])
                for word in tweet.split(" "):
                    if word in w2v:
                        cur.append(w2v[word])
                if len(cur)<maxlen and cur:
                    for i in range(maxlen - len(cur)):
                        cur.append([0.0] * (100))
                if cur:
                    X.append(cur)




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


    pred_active=model.predict(X_active)


    with open("active.txt", "a+") as writeFile:
        for i in range(len(pred_active)):
            pred_active[i].sort()
            m1=pred_active[i][2]
            m2=pred_active[i][1]
            if m1-m2 < 0.1 :
                writeFile.write(" ".join(X_active[i]))


    return X, Y

file="hate_data_utf.csv"
iteration=0
X, Y=lstm(Xtest, Ytest,X, Y, maxlen, results, w2v, file, iteration)

print(results)
