"""
Make feature vectors and compute a topic model.
"""
import sys
import re
import string
import sklearn
import numpy as np
import scipy
import math
import nltk

import lda

TOKENSEP = " TOKENSEP "

class Analyzer:
    def __init__(self):
        self.min_ngram = 1
        self.max_ngram = 3
        sw = set(map(lambda x: str(x).lower(), nltk.corpus.stopwords.words('english')))
        [sw.add(x) for x in []]
        self.stop_words = frozenset(sw)


    def __call__(self, feat_string): # generate features!
        base_feats = feat_string.split(TOKENSEP)
        
        # make ngrams from lemmatized words
        for n in xrange(self.min_ngram, self.max_ngram+1):
            for ng in nltk.ngrams([bf for bf in base_feats if bf.startswith("LEMM:") and bf[5:].lower() not in self.stop_words], n):
                s = ' '.join(map(lambda s: s.strip("LEMM:"), ng))
                if len(ng) > 1:
                    yield s
                else: # only strip stop words for 1-grams
                    if s not in self.stop_words:
                        yield s
        
        # use the original words, but not stop words
        #for bf in [bf for bf in base_feats if bf.startswith("ORIG:") and bf[5:].lower() not in self.stop_words]:
        #    yield bf
        
        # this will overlap with ORIG:, but if we don't use ORIG, then it'll work
        for bf in [bf for bf in base_feats if bf.startswith("NNP:")]:
            yield bf

        #for bf in [bf for bf in base_feats if bf.startswith("TENSE:")]:
        #    yield bf

        ##for bf in [bf for bf in base_feats if bf.startswith("SYN:")]:
        ##    yield bf
        
        # wiki categories
        for bf in [bf for bf in base_feats if bf.startswith("WIKICAT:")]:
            yield bf


class Featurizer:
    def __init__(self, analyzer, pages_dict):
        
        self.analyzer = analyzer
        self.wiki_pages_dict = pages_dict
        
        self.pquestions = {} # processed questions
        self.pcontent = {}
        self.psummary = {}
        self.psections = {}
        
        # need true counts for LDA
        self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(\
                min_df=0.0, max_df=0.5, max_features=75000, analyzer=self.analyzer)
        # LDA expects true counts, not a TF-IDF feature
        #self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(\
        #        min_df=0.0, max_df=0.5, max_features=75000, analyzer=self.analyzer)
        

    @staticmethod
    def process_text(text, pos_tag="good"):
        """
        Lemmatize and POS tag a chunk of text to be used later to build a feature vector.

        Args:
            text: document string (either question or Wiki page summary, content, etc.)

            pos_tag="good": which POS tagger to use.
                "good" - nltk.pos_tag, which is slow
                "bad" - assume everything is a noun, which is stupid

        Returns:
            feat_string: tokenized string of features.  
                Tokens are separated by similarity.TOKENSEP.
        """
        # set up some vars
        wnl = nltk.stem.WordNetLemmatizer()                
        
        # remove various characters
        re_numbers = re.compile("\d|^\d+\s|\s\d+\s|\s\d+$") # numbers
        #punc = '!"#$%&()*+,\'-./:;<=>?@[\\]^_`{|}~' # string.punctuation
        punc = '!"#$%&()*+,\'./:;<=>?@[\\]^_`{|}~' # string.punctuation - "-"
        re_punc = re.compile('[%s]' % re.escape(punc))# punctuation
        punc_split = '-'
        re_punc_split = re.compile('[%s]' % re.escape(punc_split))# split these puncs
        
        original_tokens = ""
        new_tokens = ""
        additional_tokens = ""

        # Tokenize and clean up numbers, puntuation, etc
        #text = re_numbers.sub(" ", text) # remove digits
        text = re_punc.sub("", text) # remove punctuation
        text = re_punc_split.sub(" ", text)
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [token for token in tokens if token not in \
                frozenset((".","?","!",",",":",";","$","\"","(",")","[","]","{","}","`",""))]
        
        if pos_tag == "good":
            tokens_pos = nltk.pos_tag(tokens) # this is _slow_
        elif pos_tag == "bad":
            tokens_pos = [('','n')]*len(tokens) # this is stupid
        else:
            raise ValueError("pos_tag={} is not recognized".format(pos_tag))

        added_synsets = set()
        for token_ind, token in enumerate(tokens):

            # original (can make a unigram out of this)
            original_tokens += "ORIG:" + token.lower() + TOKENSEP
            #original_tokens += "ORIG:" + token + TOKENSEP
            
            # Lemmatize things
            #token_wn_pos = tokens_pos[token_ind][1].lower()[0] # taking first char from upenn tag to use in wordnet
            #if token_wn_pos in set(['n','v','a']):
            #    lemmatized_token = wnl.lemmatize(token, pos=token_wn_pos)
            #else:
            #    lemmatized_token = token

            ##TODO: we probably want to weight proper nouns very heavily, if we can find them
            #if tokens_pos[token_ind][1][0:3] == "NNP": # proper noun (singular or plural)
            #    additional_tokens += "NNP:" + token.lower() + TOKENSEP

            #new_tokens += "LEMM:" + lemmatized_token.lower() + TOKENSEP
        
        return original_tokens + new_tokens + additional_tokens

        
    def compute_feat_strings(self, questions, print_info=False):
        """
        Compute a list of feature strings (parse with Analyzer) for each question and answer.

        Args:
            questions: list of question dicts
            
            print_info=False: print current iterate and total number

        Returns:
            feat_strings: list of feature strings
            feat_vocab: dict mapping question 'id' or answer string to index of feat_strings
        """
        feat_vocab = {}
        feat_ind = 0
        feat_strings = [] # collect all feature strings to build the feature matrix
        for ind, q in enumerate(questions):
            if print_info:
                print ("\rquestion {:>06d} of {:>06d}".format(ind+1, len(questions))),
            sys.stdout.flush()
            
            if q['id'] not in self.pquestions:
                #self.pquestions[q['id']] = self.process_text(q['question'], pos_tag="good")
                self.pquestions[q['id']] = self.process_text(q['question'], pos_tag="bad")

                feat_vocab[q['id']] = feat_ind
                feat_strings.append(self.pquestions[q['id']])
                feat_ind += 1
            else:
                print ("repeated question id: {}".format(q['id']))


            for ans in (q['answerA'], q['answerB'], q['answerC'], q['answerD']):
                feat_string = ""
                if ans not in self.pcontent:
                    self.pcontent[ans] = self.process_text(\
                            self.wiki_pages_dict[ans]['content'], pos_tag="bad")
                    feat_string += self.pcontent[ans] + TOKENSEP

                if ans not in self.psummary:
                    self.psummary[ans] = self.process_text(\
                            #self.wiki_pages_dict[ans]['summary'], pos_tag="good")
                            self.wiki_pages_dict[ans]['summary'], pos_tag="bad")
                    feat_string += self.psummary[ans] + TOKENSEP

                #if ans not in self.psections: # this is completely untested
                #    self.psections[ans] = self.process_text(\
                #            " ".join(self.wiki_pages_dict[ans]['sections']))
                
                if ans not in feat_vocab:
                    feat_vocab[ans] = feat_ind
                    feat_strings.append(feat_string)                    
                    feat_ind += 1
        
        if print_info:
            print ("\r")
        
        return feat_strings, feat_vocab

    
    def compute_feats(self, feat_strings, test=False):
        """
        Use analyzer to parse feature strings and build a feature matrix.

        Args:
            feat_strings: list of feature strings

            test=False: whether to use fit_transform or transform

        Returns:
            feat_mat: n_strings x n_feat matrix of feature vectors

                One can access the feature names via self.vectorizer.get_feature_names()
        """
        if not test:
            feat_mat = self.vectorizer.fit_transform(feat_strings)
        else:
            feat_mat = self.vectorizer.transform(feat_strings)

        return feat_mat


class Scorer:
    def __init__(self):
        pass
    
    #@staticmethod
    #def lskdjflskdjf


# add wiki categories to feature string
def add_wiki_categories(questions, fs, fv, pages_dict):

    # Bad words in categories (too general)
    WORDS = frozenset([u'article', u'references', u'sources', u'pages', u'script', u'dmy',
         u'wikidata', u'maint', u'use', u'links', u'mdy', u'Engvarb', u'cs1',
         u'wikipedia'])

    completed_inds = set()
    for q in questions:
        for key in ['answerA', 'answerB', 'answerC', 'answerD']:
            ans = q[key]
            i = fv[ans]
            if i not in completed_inds:
                cats = pages_dict[ans]['categories']
                if cats:
                    for cat in cats:
                        if all(l not in cat.lower() for l in WORDS):
                            fs[i] += "WIKICAT:" + cat.lower() + TOKENSEP
                completed_inds.add(i)

    return fs


