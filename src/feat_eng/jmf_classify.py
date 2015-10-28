import csv
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

import argparse
import random
import time
import itertools

import cPickle as pickle

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
#from sklearn.grid_search import GridSearchCV # we can't change solver params!

import re
import string

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

from pytvdbapi import api as tvapi

# code moved below for Moodle...
#import tvdb # this is my module tvdb.py

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
kTAGSET = ['False', 'True']


# use with sklearn.feature_selection.SelectKBest
def info_gain_score(X, y):
    """
    Compute IG(X,y) = H(X) - H(X|y) \ge 0 and use it to score the most
    useful features.
    """

    # this code is stolen from
    # http://stackoverflow.com/questions/25462407/fast-information-gain-computation
    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
        for c in classCnt:
            probs = classCnt[c] / float(featureTot)
            entropy_x_set = entropy_x_set - probs * np.log(probs)
            probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
            entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        for c in classTotCnt:
            if c not in classCnt:
                probs = classTotCnt[c] / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                             +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)

    tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1
    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log(probs)

    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain = []
    for i in range(0, len(nz[0])):
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre+1, nz[0][i]):
                information_gain.append(0)
            ig = _calIg()
            information_gain.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1
    ig = _calIg()
    information_gain.append(ig)

    return np.asarray(information_gain), np.zeros_like(information_gain)


class Analyzer:
    def __init__(self):
        self.min_ngram = 1
        self.max_ngram = 3
        sw = set(map(lambda x: str(x).lower(), stopwords.words('english')))
        [sw.add(x) for x in []]
        self.stop_words = frozenset(sw)


    def __call__(self, feat_string): # generate features!
        base_feats = feat_string.split(" TOKENSEP ")
        
        for n in xrange(self.min_ngram, self.max_ngram+1):
            for ng in nltk.ngrams([bf for bf in base_feats if bf.startswith("LEMM:") and bf[5:].lower() not in self.stop_words], n):
                s = ' '.join(map(lambda s: s.strip("LEMM:"), ng))
                if len(ng) > 1:
                    yield s
                else: # only strip stop words for 1-grams
                    if s not in self.stop_words:
                        yield s

        #for bf in [bf for bf in base_feats if bf.startswith("ORIG:") and bf[5:].lower() not in self.stop_words]:
        #    yield bf

        for bf in [bf for bf in base_feats if bf.startswith("TENSE:")]:
            yield bf

        #for bf in [bf for bf in base_feats if bf.startswith("SYN:")]:
        #    yield bf

        for bf in [bf for bf in base_feats if bf.startswith("PAGE:")]:
            yield bf

        for bf in [bf for bf in base_feats if bf.startswith("TROPE:")]:
            yield bf

        for bf in [bf for bf in base_feats if bf.startswith("TVDB_GENRE:")]:
            yield bf

        for bf in [bf for bf in base_feats if bf.startswith("THIS_:")]:
            yield bf


class Featurizer:
    def __init__(self, train, analyzer):
        
        self.train = train
        self.analyzer = analyzer
        self.train_examples = []
        self.test_examples = []
        self.x_train = None
        self.y_train = None
        self.x_test = None

        self.vectorizer = CountVectorizer(max_df=0.75, max_features=75000, analyzer=analyzer)
        self.tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=False)
        #self.kbest_feats = sklearn.feature_selection.SelectKBest(score_func=sklearn.feature_selection.chi2,\
        self.kbest_feats = sklearn.feature_selection.SelectKBest(score_func=info_gain_score,\
            k=20000)
        
        # set of first names
        # http://www.cs.princeton.edu/introcs/data/names.csv
        self.names = set(map(lambda x: x[0].lower(), csv.reader(open('../data/spoilers/names.csv', 'r'))))
        #self.names = ["tim", "johnny", "olivia", "regina", "michael", "sherlock", "peter"]

        self.ref_text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())

    
    def process_examples(self, examples, pages, tropes):
        examples = list(examples) # make a list from the generator
        pages = list(pages)
        tropes = list(tropes)
        
        print ("Loading TVDB data")
        page_names, tvdb_data = tvdb.load_show_info() # load data from locally pickled file

        token_sep = " TOKENSEP "
        
        # set up some vars
        wnl = WordNetLemmatizer()                
        
        # remove various characters
        re_numbers = re.compile("\d|^\d+\s|\s\d+\s|\s\d+$") # numbers
        #re_punc = re.compile("[^\P{P}-]+") # punctuation
        punc = '!"#$%&()*+,\'-./:;<=>?@[\\]^_`{|}~' # string.punctuation
        #punc = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~' # string.punctuation - "'"
        re_punc = re.compile('[%s]' % re.escape(punc))# punctuation

        new_examples = []
        for example_ind, example in enumerate(examples):
            print ("\rexample {:>06d} of {:>06d}".format(example_ind+1, len(examples))),

            original_example = ""
            new_example = ""
            additional_feats = ""

            # Tokenize and clean up numbers, puntuation, etc
            example = re_numbers.sub(" ", example) # remove digits
            example = re_punc.sub("", example) # remove punctuation
            #tokens = ("".join(re_numbers.split(example))).split()
            #tokens = ("".join(re_notword.split(example))).split()
            tokens = nltk.tokenize.word_tokenize(example)
            tokens = [token for token in tokens if token not in \
                    frozenset((".","?","!",",",":",";","$","\"","(",")","[","]","{","}","`",""))]
            #print (tokens)

            ## remove names (well, except Shyamalan, because anything one says is a spoiler!)
            ## this is pretty expensive... and didn't work too well
            #pos = nltk.pos_tag(tokens)
            #chunk = nltk.ne_chunk(pos)
            #for st in chunk.subtrees(filter=lambda t: t.label() == 'PERSON'):
            #    #print (st.label(), ' '.join(c[0] for c in st.leaves()))
            #    for c in st.leaves():
            #        example = example.replace(c[0], " ")
           
            # XXX lemmatizing settings; use the stupid one to save some time
            tokens_pos = nltk.pos_tag(tokens) # (this is _slow_)
            #tokens_pos = [('','n')]*len(tokens) # (this is stupid)

            added_synsets = set()
            for token_ind, token in enumerate(tokens):

                # original (can make a unigram out of this)
                original_example += "ORIG:" + token.lower() + token_sep
                

                # Lemmatize things
                token_wn_pos = tokens_pos[token_ind][1].lower()[0] # taking first char from upenn tag to use in wordnet
                if token_wn_pos in set(['n','v','a']):
                    lemmatized_token = wnl.lemmatize(token, pos=token_wn_pos)
                else:
                    lemmatized_token = token

                new_example += "LEMM:" + lemmatized_token.lower() + token_sep


                #if len(lemmatized_token) > 2 and lemmatized_token.lower() not in self.stop_words:
                ##if len(lemmatized_token) > 2:
                #    new_example += lemmatized_token
                #new_example += " "

                # replace first names with "name"
                #if lemmatized_token.lower() in self.names:
                #    lemmatized_token = "name"

                # add in some synonyms (this is _slow_)
                #if lemmatized_token.lower() not in self.analyzer.stop_words:
                #    if token_wn_pos in set(['n','v','a']):
                #        synsets = wn.synsets(token, token_wn_pos)
                #    else:
                #        synsets = wn.synsets(token)

                #    for synset in synsets:
                #        count = 0
                #        for ln in synset.lemma_names():
                #            if count > 1:
                #                break
                #            ln = ln.lower()
                #            if ln not in added_synsets:
                #                additional_feats += "SYN:" + ln
                #                additional_feats += token_sep
                #                added_synsets.add(ln)
                #                count += 1
                
                # Add in verb tense (think that future tense might indicate spoiler)
                #if token_wn_pos == 'v':
                #    #print (tokens_pos[i])
                #    additional_feats += "TENSE:" + tokens_pos[i][1] + token_sep
                
                if lemmatized_token == "this": # look out for "this *" features
                    if token_ind < len(tokens)-1:
                        additional_feats += "THIS_:"+tokens[token_ind].lower()+"_"+tokens[token_ind+1].lower()+token_sep



            # add the page
            additional_feats += "PAGE:" + pages[example_ind].lower() + token_sep

            # add the trope
            additional_feats += "TROPE:" + tropes[example_ind].lower() + token_sep
           

            # add in discrete data from TVDB
            key = pages[example_ind]
            if key in tvdb_data:
                show = tvdb_data[key]

                # add genre
                for genre in show.Genre:
                    additional_feats += "TVDB_GENRE:" + genre.lower() + token_sep

                # don't add runtime here; not discrete enough
                # set([u'', 1, 7, 10, 11, 15, 20, 25, 30, 35, 40, 45, 50, 180, 55, 60, 65, 70, 75, 80, 85, 90, 120])
                

            new_example = new_example + original_example + additional_feats
            new_examples.append(new_example)
            #print (new_example)
            #raise SystemExit
        
        print ("\r")

        examples = new_examples 

        return examples


    def train_feature(self, examples, y_train, pages, tropes):
        examples = self.process_examples(examples, pages, tropes)

        self.train_examples = examples
        
        feat_mat = self.vectorizer.fit_transform(examples)
        feat_mat = self.tfidf_transformer.fit_transform(feat_mat)
        feat_mat = self.kbest_feats.fit_transform(feat_mat, y_train)

        self.x_train = feat_mat
        self.y_train = y_train
        
        return feat_mat

    def test_feature(self, examples, pages, tropes):
        examples = self.process_examples(examples, pages, tropes)

        self.test_examples = examples

        feat_mat = self.vectorizer.transform(examples)
        feat_mat = self.tfidf_transformer.transform(feat_mat)
        feat_mat = self.kbest_feats.transform(feat_mat) 

        self.x_test = feat_mat
        
        return feat_mat

    def train_from_examples(self):
        print ("Fitting feat_mat from precomputed train_examples")
        examples = self.train_examples
        
        feat_mat = self.vectorizer.fit_transform(examples)
        feat_mat = self.tfidf_transformer.fit_transform(feat_mat)
        feat_mat = self.kbest_feats.fit_transform(feat_mat, self.y_train)

        self.x_train = feat_mat

        return feat_mat
    
    def test_from_examples(self):
        print ("Fitting feat_mat from precomputed test_examples")
        examples = self.test_examples

        feat_mat = self.vectorizer.transform(examples)
        feat_mat = self.tfidf_transformer.transform(feat_mat)
        feat_mat = self.kbest_feats.transform(feat_mat) 
        
        self.x_test = feat_mat

        return feat_mat


    def show_top10(self, classifier, categories):
        N_feats = 20
        
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2: # sklearn has a different shape for coef_ in this case
            sort_feats = np.argsort(classifier.coef_[0])
            for i, category in enumerate(categories):
                print ("Category: {}".format(category))
                
                if i == 0:
                    top_feats = sort_feats[0:N_feats]
                else:
                    top_feats = reversed(sort_feats[-N_feats:])

                for feat in top_feats:
                    print ("    \"{}\", \t\t\tbeta_j = {}".format(feature_names[feat], classifier.coef_[0][feat]))


        else:
            for i, category in enumerate(categories):
                print ("Category: {}".format(category))
                
                top_feats = np.argsort(classifier.coef_[i])[-N_feats:]
                for feat in top_feats:
                    print ("    \"{}\"".format(feature_names[feat]))


# Some utility functions
def split_list(l, splits):
    """
    Split a list according to the iterable splits, giving the split percentages
    """
    lists = []
    splits = map(int, splits)
    if sum(splits) != 100:
        raise ValueError("splits should sum to 100")

    if splits[0] == 100:
        splits = [100, 0]

    ind_start = 0
    for split in splits:
        ind_end = ind_start + int(float(split)/100.*float(len(l)))

        if ind_end == len(l) - 1: # clean up last index
            ind_end = len(l)

        lists.append(l[ind_start:ind_end])
        ind_start = ind_end

    return lists


def accuracy(classifier, x, y, test, test_feats):
    predictions = classifier.predict(x)
    cm = confusion_matrix(y, predictions)

    print("Accuracy: %f" % accuracy_score(y, predictions))

    print("\t".join(kTAGSET))
    for ii in cm:
        print("\t".join(str(x) for x in ii))
   
    num_to_print = 10
    i = 0
    while True:
        if y[i] != predictions[i]:
            print ("{}\n{}\n\tTrue: {}, \t Guess: {}".format(test[i]['sentence'],test_feats[i],
                kTAGSET[y[i]], kTAGSET[predictions[i]]))
            num_to_print -= 1

        if num_to_print < 0:
            break

        i += 1

## Code from tvdb.py ##
# I'm going to make calls to thetvdb.com's API to retrive information about shows
# If I get data successfully, I'll need to host it locally (e.g. pickle it to 
# play nice with their servers)
TVDB_API_KEY = "3889EDD44469A33D"


def split_camel_case(arg):
    """
    split list of strings based on changes of case in a
    CamelCase string
    """
    
    # http://stackoverflow.com/questions/2273462/regex-divide-with-upper-case
    case_change = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+")

    splits = []
    for s in arg:
        matches = case_change.findall(s)
        splits.append(" ".join(matches))

    return splits


def get_all_page_names():
    
    train = list(csv.DictReader(open("../data/spoilers/train.csv","r")))
    test = list(csv.DictReader(open("../data/spoilers/test.csv","r")))

    page_names = set()
    for entry in itertools.chain(train, test):
        page_names.add(entry["page"])
    
    return list(page_names)


def get_show_info():
    
    tvdb = tvapi.TVDB(TVDB_API_KEY)
    
    page_names = get_all_page_names()
    split_page_names = split_camel_case(page_names)
    
    logfile = open("tvdb_data.log", "w")

    tvdb_data = {}
    num_pages_found = 0
    for page, sp in itertools.izip(page_names, split_page_names):
        
        try:
            print (page),
            logfile.write((page + ','))
            show = None
            results = tvdb.search(sp, 'en')
            # just pick the first result, I guess
            if len(results) > 0:
                show = results[0]
                num_pages_found += 1
            
            if show:
                tvdb_data[page] = show

            print (show)
            logfile.write(repr(show))

        except:
            print ("There was an error; continuing")
            logfile.write(',ERROR')
       
        logfile.write('\n')
        logfile.flush() # I'm impatient
        time.sleep(2)

    
    print ("TVDB entries found for {} of {} pages".format(num_pages_found, len(page_names)))
    print ("pickling output to tvdb_data.pkl")
    pickle.dump((page_names, tvdb_data), open("tvdb_data.pkl", "wb"), -1)


def load_show_info():
    
    page_names, tvdb_data = pickle.load(open("tvdb_data.pkl", "rb"))
    if tvdb_data:
        return page_names, tvdb_data
    else:
        raise IOError("Something terrible has happened...")


#def manually_update_show_info():
## some of the searches didn't work!  Manually fix them here.
## {{{
#
#    tvdb = tvapi.TVDB(TVDB_API_KEY)
#    
#    page_names, tvdb_data = load_show_info()
#   
#    missed_names = []
#    missed_search = []
#    missed_index = []
#    missed_lang = []
#    
#    missed_names.append('LosUnicos')
#    missed_search.append('Los Unicos')
#    missed_index.append(0)
#    missed_lang.append('en')
#
#    for mn, ms, mi, ml in itertools.izip(missed_names, missed_search,\
#            missed_index, missed_lang):
#
#        results = tvdb.search(ms, ml)
#
#
#    
#    #pickle.dump((page_names, tvdb_data), open("tvdb_data.pkl", "rb"), -1)
#
## }}}


## End Code from tvdb.py


#def make_predictions(args):
#    """
#    This will use all the training data to make predictions on the test set.
#    """
#    # Cast to list to keep it all in memory
#    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
#    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
#
#    feat = Featurizer()
#
#    labels = []
#    for line in train:
#        if not line[kTARGET_FIELD] in labels:
#            labels.append(line[kTARGET_FIELD])
#
#    x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
#    x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)
#
#    y_train = array(list(labels.index(x[kTARGET_FIELD])
#                         for x in train))
#
#    # Train classifier
#    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
#    lr.fit(x_train, y_train)
#
#    feat.show_top10(lr, labels)
#
#    predictions = lr.predict(x_test)
#    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
#    o.writeheader()
#    for ii, pp in zip([x['id'] for x in test], predictions):
#        d = {'id': ii, 'spoiler': labels[pp]}
#        o.writerow(d)


def run_development(args):
    """
    This will split the trianing database into a train and test set.
    """

    if not args.load and not args.loadex:
        train_full = list(DictReader(open("../data/spoilers/train.csv", 'r')))
        random.shuffle(train_full) 
        
        if args.limit > 0:
            train_full = train_full[0:args.limit]
        
        train, test = split_list(train_full, (args.split, 100.-args.split))
        
        print (len(train), len(test))
        print ("Generating features")

        analyzer = Analyzer()
        feat = Featurizer(train, analyzer)

        # train data
        #x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)

        train_labels = ['False', 'True']
        for line in train:
            if not line[kTARGET_FIELD] in train_labels:
                train_labels.append(line[kTARGET_FIELD])

        y_train = array(list(train_labels.index(x[kTARGET_FIELD])
                             for x in train))
        x_train = feat.train_feature((x[kTEXT_FIELD] for x in train), y_train,\
                (x['page'] for x in train), (x['trope'] for x in train))
        
        # test data
        x_test = feat.test_feature((x[kTEXT_FIELD] for x in test),\
                (x['page'] for x in test), (x['trope'] for x in test))

        test_labels = ['False', 'True']
        for line in test:
            if not line[kTARGET_FIELD] in test_labels:
                test_labels.append(line[kTARGET_FIELD])
        
        if len(test_labels) != len(train_labels):
            raise ValueError("Bad train/test categories!")
        else:
            test_labels = train_labels

        y_test = array(list(test_labels.index(x[kTARGET_FIELD])
                             for x in test))

        with open('features.pkl', 'wb') as f:
            pickle.dump((train, test, x_train, y_train, x_test, y_test, feat, train_labels), f, -1)

    elif args.load:
        with open('features.pkl', 'rb') as f:
            train, test, x_train, y_train, x_test, y_test, feat, train_labels = pickle.load(f)

    elif args.loadex:
        with open('features.pkl', 'rb') as f:
            train, test, _, y_train, _, y_test, feat_old, train_labels = pickle.load(f)

            analyzer = Analyzer()
            feat = Featurizer(train, analyzer)
            feat.train_examples = feat_old.train_examples
            feat.test_examples = feat_old.test_examples
            feat.y_train = y_train
            x_train = feat.train_from_examples()
            x_test = feat.test_from_examples()

    
    ## Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    #lr = gs.best_estimator_ # this gets refit with the best params from Xval
    feat.show_top10(lr, train_labels)
    mean_accuracy = lr.score(x_train, y_train)
    print ("mean train accuracy: {}".format(mean_accuracy))
    mean_accuracy = lr.score(x_test, y_test)
    print ("mean test accuracy: {}".format(mean_accuracy))

    print ("\nconf mat on test set:")
    accuracy(lr, x_test, y_test, test, feat.test_examples)


    # Make predictions
    if args.test:
        print ("\nMaking predictions.csv")
        # Cast to list to keep it all in memory
        train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
        test = list(DictReader(open("../data/spoilers/test.csv", 'r')))

        analyzer = Analyzer()
        feat = Featurizer(train, analyzer)

        labels = ['False', 'True']
        for line in train:
            if not line[kTARGET_FIELD] in labels:
                labels.append(line[kTARGET_FIELD])

        #x_train = feat.train_feature(x[kTEXT_FIELD] for x in train)
        #x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

        y_train = array(list(labels.index(x[kTARGET_FIELD])
                             for x in train))

        x_train = feat.train_feature((x[kTEXT_FIELD] for x in train), y_train,\
                (x['page'] for x in train), (x['trope'] for x in train))
        
        x_test = feat.test_feature((x[kTEXT_FIELD] for x in test),\
                (x['page'] for x in test), (x['trope'] for x in test))

        #x_train = feat.train_feature((x[kTEXT_FIELD] for x in train), y_train)
        #x_test = feat.test_feature(x[kTEXT_FIELD] for x in test)

        # Train classifier
        lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
        lr.fit(x_train, y_train)

        feat.show_top10(lr, labels)

        predictions = lr.predict(x_test)
        o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
        o.writeheader()
        for ii, pp in zip([x['id'] for x in test], predictions):
            d = {'id': ii, 'spoiler': labels[pp]}
            o.writerow(d)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", help="Percent going to training set",
            type=float, default=90., required=False)
    argparser.add_argument("--limit", help="Limit training size",
            type=int, default=-1, required=False)
    argparser.add_argument("--load", help="Load precomputed features",
            type=bool, default=False, required=False)
    argparser.add_argument("--loadex", help="Load precomputed examples only",
            type=bool, default=False, required=False)
    argparser.add_argument("--test", help="Make predictions on the true test set",
            type=bool, default=False, required=False)

    args = argparser.parse_args()

    #make_predictions(args)
    run_development(args)

