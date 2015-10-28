from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'


class Featurizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, analyzer='word', ngram_range=(1,1), strip_accents = None, min_df=1)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

    def stemming(self, index, text, data):
        from porter2 import stem
        
        if len(text) <= 129:
            text += ' twentyfivesentence'
        elif len(text) <= 181:
            text += ' fiftysentence'
        elif len(text) <= 243:
            text += ' seventyfivesentence'
        else:
            text += ' largesentence'
        
        for i in xrange(text.count('%')):
            text += ' uniqpercent'
        for i in xrange(text.count('@')):
            text += ' uniqatmark'
        for i in xrange(text.count(',')):
            text += ' uniqcomma'
        for i in xrange(text.count("'")):
            text += ' uniqapostrophe'
        for i in xrange(text.count('...')):
            text += ' uniqellipses'
        for i in xrange(text.count(':')):
            text += ' uniqcolon'
        for i in xrange(text.count('!')):
            text += ' uniqexclamation'
        if '(' or ')' in text:
            text += ' uniqparentheses'
        for i in xrange(text.count('?')):
            text += ' uniqquestion'
        for i in xrange(text.count('"')):
            text += ' uniqquote'
        for i in xrange(text.count('#')):
            text += ' uniqhashtag'
        for i in xrange(text.count('0')):
            text += ' uniqzero'
        for i in xrange(text.count('1')):
            text += ' one'
        for i in xrange(text.count('2')):
            text += ' two'
        for i in xrange(text.count('3')):
            text += ' three'
        for i in xrange(text.count('4')):
            text += ' four'
        for i in xrange(text.count('5')):
            text += ' five'
        for i in xrange(text.count('6')):
            text += ' six'
        for i in xrange(text.count('7')):
            text += ' seven'
        for i in xrange(text.count('8')):
            text += ' eight'
        for i in xrange(text.count('9')):
            text += ' nine'
        if '/' in text:
            text = text.replace('/', ' ')
            text += ' forwardslash'
        
        upper = sum(1 for i in text if i.isupper())
        iterate = 1
        while upper - iterate > 0:
            text += ' formalword'
            iterate += 1
        
        
        import wikipedia
        import re
        if index % 100 == 0:
            p = 100*index/len(data)
            print "Wiki part, percent: %d" % p
        searchForEpi = data[index]['trope']
        searchForEpi = re.sub(r'([A-Z])', r' \1', searchForEpi)
        searchForTitle = data[index]['page']
        searchForTitle = re.sub(r'([A-Z])', r' \1', searchForTitle)
        search = searchForEpi + ' ' + searchForTitle
        try:
            summary = wikipedia.summary(search, sentences=1)
            newtext = str(text) + ' ' + str(summary)
        except:
            summary = None
        
        
        word_list = text.split()
        for i in xrange(len(word_list)):
            word_list[i] = stem(word_list[i])
        join_list = ' '.join(word_list)
        
        join_list += ' ' + data[index]['page']
        
        return join_list
                

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))

    for i in xrange(len(train)):
        train[i]['sentence'] = feat.stemming(i, train[i]['sentence'], train)

    for i in xrange(len(test)):
        test[i]['sentence'] = feat.stemming(i, test[i]['sentence'], test)
    
    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

    
    # sub_x sets
    x_sub_train = x_train[range(0, int(0.8*len(train)))]
    x_sub_test = x_train[range(int(0.8*len(train))+1, len(train))]
    

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    
    # sub_y sets
    y_sub_train = y_train[range(0, int(0.8*len(train)))]
    y_sub_test = y_train[range(int(0.8*len(train))+1, len(train))]
    

    print(len(train), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_sub_train, y_sub_train)

    feat.show_top10(lr, labels)

    
    # ADDED CODE
    acc = lr.score(x_sub_test, y_sub_test)
    print("Accuracy of training set: %f" % acc)
    preds = lr.predict(x_sub_test)
    right = 0
    wrong = 0
    for i in xrange(len(preds)):
        if preds[i] != y_sub_test[i]:
            wrong += 1
            # print train[i]['sentence']
        else:
            right += 1
    print "Correct: %d" % right
    print "Incorrect: %d" % wrong
    
    
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"], lineterminator = '\n')
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)

    print('done')
