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


# model = Sequential()
# model.add(LSTM(hidden_size, return_sequences=True, input_shape=(timesteps, input_dim)))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(TimeDistributed(Dense(len(w2v))))
# model.add(Activation('softmax'))

# optim=RMSprop(lr=0.001)
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


f1=open("sexism.txt", "a+", encoding="utf-8")
f2=open("racism.txt", "a+", encoding="utf-8")
print(precision_recall_fscore_support(Y_raw_test, preds))
for i in range(len(preds)):
    if preds[i]==0 and Y_raw_test[i]==1:
        #misclassified sexism
        state=True
        line=" ".join(X_text[i])
        for l in f1:
            if l==line:
                state=False
        if state:
            f1.write(line)
            f1.write("\n")
    if preds[i]==0 and Y_raw_test[i]==2:
        #misclassified racism
        state = True
        line = " ".join(X_text[i])
        for l in f2:
            if l == line:
                state = False
        if state:
            f2.write(line)
            f2.write("\n")
f1.close()
f2.close()
