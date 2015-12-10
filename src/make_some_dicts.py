
import sys
import numpy as np
import argparse
import random
import re
import string
import itertools
import csv
import cPickle as pickle
import lda
import sklearn

# our modules
import read_csv_data
import similarity
import topic_model
import utils

def compute_scores(questions, X, fv, scorer=similarity.Scorer.cosine,\
        topics=None, print_info=False, train=False, normalize=True):
    """
    """
    scores = np.zeros((4*len(questions), 2))
    if train:
        y = np.zeros((4*len(questions),)) # binary classification
    for ind, q in enumerate(questions):
        if print_info:
            print ("\ranswer {:>06d} of {:>06d}".format(ind+1, len(questions))),
        sys.stdout.flush()
 
        qi = fv[q['id']]
        ai = fv[q['answerA']]
        bi = fv[q['answerB']]
        ci = fv[q['answerC']]
        di = fv[q['answerD']]
        
        qv = X[qi,:] # efficient in CSR format
        av = X[ai,:]
        bv = X[bi,:]
        cv = X[ci,:]
        dv = X[di,:]
    
        # Similarity measure stuff
        scores[4*ind+0,0] = scorer(qv, av)
        scores[4*ind+1,0] = scorer(qv, bv)
        scores[4*ind+2,0] = scorer(qv, cv)
        scores[4*ind+3,0] = scorer(qv, dv)
        
        # Just use sklearn's standard scaler
        #if normalize:
        #    sm = sum(scores[4*ind+0:4*ind+3,0])
        #    scores[4*ind+0:4*ind+3,0] /= sm
 
        # LDA stuff
        if topics is not None:
            qt = topics[qi,:]
            at = topics[ai,:]
            bt = topics[bi,:]
            ct = topics[ci,:]
            dt = topics[di,:]

            # use the cosine measure to compare topics?
            
            scores[4*ind+0,1] = np.inner(qt, at) / np.linalg.norm(qt) / np.linalg.norm(at)
            scores[4*ind+1,1] = np.inner(qt, bt) / np.linalg.norm(qt) / np.linalg.norm(bt)
            scores[4*ind+2,1] = np.inner(qt, ct) / np.linalg.norm(qt) / np.linalg.norm(ct)
            scores[4*ind+3,1] = np.inner(qt, dt) / np.linalg.norm(qt) / np.linalg.norm(dt)

            #if normalize:
            #    sm = sum(scores[4*ind+0:4*ind+3,1])
            #    scores[4*ind+0:4*ind+3,1] /= sm
 

        if train:
            if 'correctAnswer' not in q:
                raise ValueError("You're not running on the training set.")
            
            ca = q['correctAnswer']
            if ca == 'A':
                y[4*ind+0] = 1
            elif ca == 'B':
                y[4*ind+1] = 1
            elif ca == 'C':
                y[4*ind+2] = 1
            elif ca == 'D':
                y[4*ind+3] = 1
   
    if print_info:
        print ()

    if train:
        return scores, y
    else:
        return scores


def make_sim_and_topic_dicts():
    pages_dict = pickle.load(open('../data/wiki_pages_dict.pkl', 'rb'))

    print ("Loading precomputed feature strings for trainx and testx:")
    #train, test, fs, fv, analyzer, feat = \
    #        pickle.load(open('../data/xval_feat_strings.pkl', 'rb'))
    #print ("Loading precomputed feature strings for real-deal train and test:")
    train, test, fs, fv, analyzer, feat = \
            pickle.load(open('../data/realdeal_feat_strings.pkl', 'rb'))


    ## Here we do some cross-validation
    X = feat.compute_feats(fs)
    X = X.tocsr() # might already be CSR
    X.sort_indices() # needed for cosine-type measures

    feat = None

    # try some LDA stuff
    print ("Training LDA topic model")
    topic_mod = lda.LDA(n_topics=20, n_iter=150)
    #topic_mod = lda.LDA(n_topics=20, n_iter=500)
    tm_analyzer = topic_model.Analyzer()
    tm_feat = topic_model.Featurizer(tm_analyzer, pages_dict) # use the same feature strings as similarity
    tm_fs = topic_model.add_wiki_categories(train+test, fs, fv, pages_dict)
    topic_X = tm_feat.compute_feats(tm_fs)
    topics = topic_mod.fit_transform(topic_X) # gives probabilities for each topic

    # compute similarity for each question and each answer (of 4)
    # use this as X (e.g. NLP similarity, LDA similarity)
    # binary classification with LR (i.e. is the answer right or not)
    
    print ("Evaluating train data:")
    X_lr_train = compute_scores(train, X, fv,\
            scorer=similarity.Scorer.cosine, topics=topics, train=False,\
            print_info=True)
    
    sim_prob_dict = dict()
    topic_prob_dict = dict()
    for ind, q in enumerate(train):
        arr = np.exp(X_lr_train[ind:ind+4,0]) # soft-max
        sim_prob_dict[q['id']] = arr / sum(arr)

        arr = np.exp(X_lr_train[ind:ind+4,1])
        topic_prob_dict[q['id']] = arr / sum(arr)
        #print (q['id'], sim_prob_dict[q['id']], topic_prob_dict[q['id']])

    print ("Evaluating test data:")
    X_lr_test = compute_scores(test, X, fv,\
            scorer=similarity.Scorer.cosine, topics=topics, train=False,\
            print_info=True)
    
    for ind, q in enumerate(test):
        arr = np.exp(X_lr_test[ind:ind+4,0]) # soft-max
        sim_prob_dict[q['id']] = arr / sum(arr)

        arr = np.exp(X_lr_test[ind:ind+4,1])
        topic_prob_dict[q['id']] = arr / sum(arr)
        #print (q['id'], sim_prob_dict[q['id']], topic_prob_dict[q['id']])
 
    pickle.dump(sim_prob_dict, open('../data/sim_prob_dict.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(topic_prob_dict, open('../data/topic_prob_dict.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    make_sim_and_topic_dicts()

