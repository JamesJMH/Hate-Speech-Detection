iteration=2 
if iteration>1:
    with open("active.txt", "r+", encoding='utf-8') as newInput:
        X=[]
        Y=[]
        for line in newInput:
            tweet=line.split("\t")[0]
            label=line.split("\t")[1]
            print(label)
            