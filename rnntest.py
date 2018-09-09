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


def shuffle_in_unison(a, b, c,d):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
    np.random.set_state(rng_state)
    np.random.shuffle(d)

def falseNeg(dict_s, dict_r, w2v):

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
            else:
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



    # X=X[1:]
    Y=Y[1:]
    Y_raw=Y_raw[1:]
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
    Y_raw=np.array(Y_raw)


    shuffle_in_unison(X,Y, X_text, Y_raw)
    test_size=1000

    Xtest=X[:test_size, :]
    X=X[test_size:, :]
    Ytest=Y[:test_size]
    Y=Y[test_size:]
    X_text=X_text[:test_size]
    Y_raw_test=Y_raw[:test_size]
    Y_raw=Y_raw[test_size:]
    X_text=X_text[:test_size]




    timesteps=maxlen # number of words in a sentence
    input_dim=100 # dimension of embedding


    hidden_size=200
    # tokenize and build vocab
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(X, Y, epochs=8)


    # second step
    Y2=[]
    X2=[]
    for i in range(len(Y_raw)):
        if Y_raw[i]!=0:
            if Y_raw[i]==2:
                Y2.append(1)
            else:
                Y2.append(0)
            X2.append(X[i])

    X2=np.array(X2)
    Y2=np.array(Y2)


    hidden_size=100
    # tokenize and build vocab
    model2 = Sequential()
    model2.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
    model2.add(Dropout(0.5))
    # model.add(LSTM(hidden_size))
    model2.add(Flatten())
    model2.add(Dense(1,activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model2.fit(X2, Y2, epochs=3)


    preds=(model.predict_classes(Xtest))
    for i in range(len(preds)):
        if preds[i]==1:
            to_change=[]
            to_change.append(Xtest[i])
            to_change=np.array(to_change)
            pred2=model2.predict_classes(to_change)[0]
            if pred2==1:
                preds[i]=2
            else:
                preds[i]=1

    # print(model.evaluate(Xtest, Ytest, verbose=0))

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


    print(precision_recall_fscore_support(Y_raw_test, preds))
    for i in range(len(preds)):
        if Y_raw_test[i]==1:
            #misclassified sexism
            line=" ".join(X_text[i])
            if line in dict_s:
                dict_s[line][1]+=1
            else:
                dict_s[line]=[0,1]
            if preds[i]==0:
                dict_s[line][0]+=1
        if Y_raw_test[i]==2:
            #misclassified racism
            line=" ".join(X_text[i])
            if line in dict_r:
                dict_r[line][1]+=1
            else:
                dict_r[line]=[0,1]
            if preds[i]==0:
                dict_r[line][0]+=1