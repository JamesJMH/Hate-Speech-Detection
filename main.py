import rnntest
import numpy as np

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

dict_s={}
dict_r={}

for i in range(200 ):
    rnntest.falseNeg(dict_s, dict_r, w2v)

f1=open("sexism.txt", "a+", encoding="utf-8")
f2=open("racism.txt", "a+", encoding="utf-8")

for key in dict_s:
    f1.write(key + "\t" + str(dict_s[key]))
    f1.write("\n")
f1.close()

for key in dict_r:
    f2.write(key + "\t" + str(dict_r[key]))
    f2.write("\n")
f2.close()