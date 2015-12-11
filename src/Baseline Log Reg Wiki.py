__author__ = '-Derek-'
#Code used from Feature engineering homework CSCI 5622 https://github.com/ezubaric/ml-hw/blob/master/feat_eng/classify.py
from csv import DictReader, DictWriter
from Featurizer import *
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
from sklearn import cross_validation

#ADDED
from lengths import length_gen


WORDS = {'article', 'references', 'sources', 'pages', 'script', 'dmy',
         'wikidata', 'maint', 'use', 'links', 'mdy', 'Engvarb', 'cs1'}

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
    answerlist=list(answerlist)
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

def checkGoodString(instring):
    instring=str(instring.encode('utf-8'))
    for x in WORDS:
        if x in instring:
            return "BAD"

    return "GOOD"






crossval=1
wikidict=pickle.load( open("../data/wiki_pages_dict.pkl","rb"))
train = list(DictReader(open("../data/sci_train.csv", 'r')))
test=   list(DictReader(open("../data/sci_test.csv",'r')))


mainTrain, testNums, labellist, correct = cross_validation.train_test_split(range(0,len(train)),range(0,len(train)),test_size=.1,random_state=0)
newtrain=[]
for x in mainTrain:
    newtrain.append(train[x])
newtest=[]
for x in testNums:
    newtest.append(train[x])

features= []
labels= []









if crossval==0:
    for x in train:
        correctans= "answer" + x['correctAnswer']
        labels.append(x[correctans])
        features.append(x['question'])
    for x in wikidict.keys():
        tempdict=wikidict[x]
        tempTrain={}
        tempTrain['content']=str(tempdict['content'].encode('utf-8'))
        tempTrain['summary']=str(tempdict['summary'].encode('utf-8'))
        splitsum= tempTrain['summary']
        splitsum=splitsum.split(".")
        for bbb in splitsum:
            features.append(bbb)
            labels.append(str(x.encode('utf-8')))
        catlist=tempdict["categories"]
        catstring=""
        # try:
        #     for bbb in catlist:
        #         if checkGoodString(bbb)=="GOOD":
        #             catstring=catstring+bbb
        #     features.append(catstring)
        #     labels.append(str(x.encode('utf-8')))
        #
        # except:
        #     b=2


    b=2

    # ADDED
    length_pages = length_gen(wikidict)
    
    feat = Featurizer()
    x_train= feat.train_feature(list(x for x in features))
    x_test= feat.test_feature(list(x["question"] for x in test))

    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)

    print x_train
    print x_test






    lr.fit(x_train,labels)
    orderedlabels=lr.classes_
    out=lr.predict_proba(x_test)
    csvobject = DictWriter(open("../data/scipredictions.csv", 'wb'), ["id", "correctAnswer"])
    csvobject.writeheader()
    for line, x in zip(out,test):
       answer= PickBestAnswer(orderedlabels,line,x)
       csvobject.writerow({"id":x["id"],"correctAnswer":answer})



else:
    train=newtrain
    test=newtest
    for x in train:
        correctans= "answer" + x['correctAnswer']
        labels.append(x[correctans])
        features.append(x['question'])
    for x in wikidict.keys():
        tempdict=wikidict[x]
        tempTrain={}
        tempTrain['content']=str(tempdict['content'].encode('utf-8'))
        tempTrain['summary']=str(tempdict['summary'].encode('utf-8'))
        splitsum= tempTrain['summary']
        splitsum=splitsum.split(".")
        for bbb in splitsum:
            features.append(bbb)
            labels.append(str(x.encode('utf-8')))
        catlist=tempdict["categories"]
        catstring=""
        # try:
        #     for bbb in catlist:
        #         if checkGoodString(bbb)=="GOOD":
        #             catstring=catstring+bbb
        #     features.append(catstring)
        #     labels.append(str(x.encode('utf-8')))
        #
        # except:
        #     b=2

    b=2


    feat = Featurizer()
    x_train= feat.train_feature(list(x for x in features))
    x_test= feat.test_feature(list(x["question"] for x in test))

    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)






    lr.fit(x_train,labels)
    orderedlabels=lr.classes_
    out=lr.predict_proba(x_test)
    correct=0
    total=0
    for line, x in zip(out,test):
        answer= PickBestAnswer(orderedlabels,line,x)
        if answer==x["correctAnswer"]:
            correct=correct+1
        total=total+1
    print float(correct)/float(total)
