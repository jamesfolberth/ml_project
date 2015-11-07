
import sys
import numpy as np
import argparse
import random
import re
import string
import itertools

import cPickle as pickle

# our modules
import read_csv_data
import similarity
import utils

def test_xval(questions, X, fv,scorer=similarity.Scorer.cosine, print_info=False):
    """
    """
    num_correct = 0
    num_total = 0
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

        similarity_scores = []
        similarity_scores.append(scorer(qv, av))
        similarity_scores.append(scorer(qv, bv))
        similarity_scores.append(scorer(qv, cv))
        similarity_scores.append(scorer(qv, dv))
        
        our_answer = ['A','B','C','D'][similarity_scores.index(max(similarity_scores))]
        correct_answer = q['correctAnswer']
         
        #print (similarity_scores)
        #print ("our_answer, correct_answer = {}, {}".format(our_answer, correct_answer))
        
        num_total += 1
        if our_answer == correct_answer:
            num_correct += 1
    
    if print_info:
        print("\r")
    
    accuracy = float(num_correct)/float(num_total)
    return accuracy


def answer_xval(args):
    """
    Answer questions on a cross-validation dataset by doing the following:
        1. Extract (or load) feature strings for the training and test set
        2. Parse the feature strings to compute feature vectors.
        2. ???
        3. Profit

    Args:
        args: ArgumentParser arguments defined in __main__

    Returns:
        None
    """
    pages_dict = pickle.load(open('../data/wiki_pages_dict.pkl', 'rb'))
    
    if not args.load:
        train_reader, _ = read_csv_data.read_csv_data()
        train = list(train_reader)
        random.shuffle(train)

        # split train for X-val
        if args.limit > 0:
            train = train[0:args.limit]
        
        trainx, testx = utils.split_list(train, (args.split, 100.-args.split))
        print ("len(xval_train) = {}, len(xval_test) = {}"\
                .format(len(trainx), len(testx)))
 
        analyzer = similarity.Analyzer()
        feat = similarity.Featurizer(trainx, analyzer, pages_dict)
        
        print ("Computing feature strings:")
        fs, fv = feat.compute_feat_strings(trainx + testx, print_info=True)
        
        pickle.dump((trainx, testx, fs, fv, analyzer, feat),
                open('../data/xval_feat_strings.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    elif args.load: # load pre-comuted feature strings
        print ("Loading precomputed feature strings for trainx and testx:")
        trainx, testx, fs, fv, analyzer, feat = \
                pickle.load(open('../data/xval_feat_strings.pkl', 'rb'))
    
    ## Here we do some cross-validation
    X = feat.compute_feats(fs)
    X = X.tocsr() # might already be CSR
    X.sort_indices() # needed for cosine-type measures

    print ("Evaluating train data:")
    acc_trainx = test_xval(trainx, X, fv,\
            scorer=similarity.Scorer.cosine, print_info=True)
    print ("Train accuracy = {}\n".format(acc_trainx))

    print ("Evaluating test data:")
    acc_testx = test_xval(testx, X, fv,\
            scorer=similarity.Scorer.cosine, print_info=True)
    print ("Test accuracy = {}\n".format(acc_testx))


def answer_questions(args):
    raise NotImplementedError()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", help="Percent going to training set for\
            X-val", type=float, default=85., required=False)
    argparser.add_argument("--limit", help="Limit training size",
            type=int, default=-1, required=False)
    argparser.add_argument("--load", help="Load precomputed feature strings",
            action="store_true")
    argparser.add_argument("--test", help="Make predictions on the real-deal test set",
            type=bool, default=False, required=False)

    args = argparser.parse_args()
    
    # do the stuff!
    if not args.test:
        answer_xval(args)

    else:
        answer_questions(args)

