__author__ = '-Derek-'
#Code used from Feature engineering homework CSCI 5622 https://github.com/ezubaric/ml-hw/blob/master/feat_eng/classify.py
from csv import DictReader, DictWriter
from Featurizer import *
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
def LabelNamestoNum(LabelNames):
    return 0

def GetCorrectAnswers(DataDict):
    answerlist = []

    for x in DataDict:
        correctAns=x["correctAnswer"]
        answerstring= "answer" + correctAns
        Answer=x[answerstring]
        answerlist.append(Answer)
    return answerlist


def LabelDict(DataDict):
    labelList= []
    answerlist=GetCorrectAnswers(DataDict)
    uniqueList=list(set(answerlist))
    for dictval, answer in zip(DataDict,answerlist):
        dictval["Label"] = uniqueList.index(answer)
        labelList.append(uniqueList.index(answer))
    return np.asarray(labelList),uniqueList

def PickBestAnswer(answerlist,percentages,x):
    a=x["answerA"]
    b=x["answerB"]
    c=x["answerC"]
    d=x["answerD"]
    percent=[0,0,0,0]

    for x,i in zip([a,b,c,d],range(4)):
        try:
            ix=answerlist.index(x)
            percent[i]=percentages[ix]
        except:
            percent[i]=0.00000
    percent=np.asarray(percent)
    percent=percent/max(percent)
    percent=list(percent)
    if min(percent) == 0:
        percentsorted=sorted(percent)
        # if next best answer is close to best answer, pick unknown answer
        if percentsorted[2] > .95:
            answer=percent.index(min(percent))
        else:
            answer=percent.index(max(percent))
    else:
        answer=percent.index(max(percent))
    if answer==0:
        answer="A"
    if answer==1:
        answer="B"
    if answer==2:
        answer="C"
    if answer==3:
        answer="D"
    return answer








wikidict=pickle.load( open("wiki_pages_dict.pkl","rb"))
train = list(DictReader(open("sci_train.csv", 'r')))
test=   list(DictReader(open("sci_test.csv",'r')))
feat = Featurizer()
x_train= feat.train_feature(list(x["question"] for x in train))
x_test= feat.test_feature(list(x["question"] for x in test))
labels,uniqueList=LabelDict(train)
lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)







lr.fit(x_train,labels)
out=lr.predict_proba(x_test)
csvobject = DictWriter(open("scipredictions.csv", 'wb'), ["id", "correctAnswer"])
csvobject.writeheader()
for line, x in zip(out,test):
   answer= PickBestAnswer(uniqueList,line,x)
   csvobject.writerow({"id":x["id"],"correctAnswer":answer})





