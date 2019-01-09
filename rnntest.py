import numpy as np
import json
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

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
                Y.append(0)

            elif label=="racism":
                Y.append(1)

            else:
                Y.append(2)

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
    np.random.seed(42)
    np.random.shuffle(a)
    np.random.seed(42)
    np.random.shuffle(b)


shuffle_in_unison(X,Y)

# divide into train and test sets

Xtest = X[:1000, :]
Xtrain = X[1000:, :]
ytest = Y[:1000]
ytrain = Y[1000:]
results=[]

# lle to reduce dimensions
# print('Reducing dimensions ...')
# from sklearn.manifold import LocallyLinearEmbedding
# embedding = LocallyLinearEmbedding(n_components=1, random_state=42)
# X_red=[]
# for x in X:
#     x = embedding.fit_transform(x)
#     cur=[]
#     for xs in x:
#         cur.append(xs[0])
#     X_red.append(cur)
# X=np.array(X_red)
# print('Done')

# PCA for encoding

from sklearn.decomposition import PCA
Xtrain_red=[]
for x in Xtrain:
    Xtrain_red.append(x.flatten())
Xtrain_red=np.array(Xtrain_red)
# pca=PCA(n_components=3)
# Xtrain_red=pca.fit_transform(Xtrain_red)

# resampling by smote
print('Resampling ...')
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import RandomOverSampler
import collections
print(collections.Counter(ytrain))
sample_dict={1:4000, 2:6000}
# sm = SMOTETomek(random_state=42, smote=SMOTE(sampling_strategy=sample_dict, random_state=42, kind='borderline1'))

sm = SMOTE(sampling_strategy=sample_dict, random_state=42, kind='borderline1')
X_res, Y_res=sm.fit_resample(Xtrain_red, ytrain)
print('Undersampling ...')
enn=EditedNearestNeighbours(sampling_strategy='majority', n_neighbors=5, kind_sel='mode')
X_res, Y_res=enn.fit_resample(X_res, Y_res)
enn2=EditedNearestNeighbours(sampling_strategy=[1,2])
X_res, Y_res=enn2.fit_resample(X_res, Y_res)




print(X_res.shape)
print(Y_res.shape)
print('final count', collections.Counter(Y_res))
print('Done')

# make X back into the 3 dimensional format, maybe back into the unreduced format
Xtrain_fin = []
for x1 in X_res:
    cur=[x1[x:x+100] for x in range(0, len(x1),100)]
    Xtrain_fin.append(cur)
Xtrain_fin=np.array(Xtrain_fin)
Xtrain=Xtrain_fin
print("final Xtrain", Xtrain_fin.shape)

# making y back

ytrain_fin=[]
for i in range(len(Y_res)):
    if Y_res[i]==0:
        ytrain_fin.append([1,0,0])
    elif Y_res[i]==1:
        ytrain_fin.append([0,1,0])
    elif Y_res[i]==2:
        ytrain_fin.append([0,0,1])
ytrain=np.array(ytrain_fin)

ytest_fin=[]
for i in range(len(ytest)):
    if ytest[i]==0:
        ytest_fin.append([1,0,0])
    elif ytest[i]==1:
        ytest_fin.append([0,1,0])
    elif ytest[i]==2:
        ytest_fin.append([0,0,1])
ytest=np.array(ytest_fin)

X_active = []
X_active_text = []

with open('hate_clean.csv', encoding='utf-8') as input_file:
    CSV = csv.reader(input_file, delimiter=",")
    for line in CSV:
        tweet = line[19]
        tweets = tweet.lower().split()
        cur = []
        for words in tweets:
            if words in w2v:
                cur.append(w2v[words])
        X_active.append(cur)
        X_active_text.append(tweet.lower())

for lines in X_active:
    if len(lines) < maxlen:
        for i in range(maxlen - len(lines)):
            lines.append([0.0] * (100))

X_active = np.array(X_active)
X_active_text = np.array(X_active_text)

#shuffle_in_unison(X_active_text, X_active)

def lstm(Xtest, Ytest, X, Y, maxlen, results, w2v, iteration, X_active, X_active_text):
    iter_size=1600
    start = (iteration - 1) * iter_size + 1
    end = iteration * iter_size + 1
    X_active=X_active[start:end]
    X_active_text=X_active_text[start:end]

    if iteration>1:
        with open("active.txt", "r+", encoding='utf-8') as newInput:
            Y=Y.tolist()
            X=X.tolist()
            for line in newInput:
                tweet=line.split("\t")[0].strip()
                label=line.split("\t")[1].strip()
                cur=[]
                if label == 'sexist':
                    Y.append([0,0,1])
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

            X=np.array(X)
            Y=np.array(Y)


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

    with open("active.txt", "a+", encoding='utf-8') as writeFile:
        for i in range(len(pred_active)):
            pred_active[i].sort()
            m1=pred_active[i][2]
            m2=pred_active[i][1]
            if m1-m2 < 0.2 :
                writeFile.write(X_active_text[i]+'\n\n')


    return X, Y

#change this
iteration=1
X, Y=lstm(Xtest, ytest,Xtrain, ytrain, maxlen, results, w2v, iteration, X_active, X_active_text)

print(results)
