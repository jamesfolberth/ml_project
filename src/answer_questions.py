
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


def make_predictions(questions, X, fv, scorer=similarity.Scorer.cosine,\
        topics=None, print_info=False):
    our_answers = [] 
    eps = 1e-10 # to avoid division by zero
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
        similarity_scores = []
        similarity_scores.append(scorer(qv, av) + 1.)
        similarity_scores.append(scorer(qv, bv) + 1.)
        similarity_scores.append(scorer(qv, cv) + 1.)
        similarity_scores.append(scorer(qv, dv) + 1.)
        
        sm = sum(similarity_scores) + 4*eps
        similarity_scores = map(lambda x: (x+eps) / sm, similarity_scores)
        
        if sm == 4*eps:
            print ("similarity_scores == [0,0,0,0] (so topics definately helps)")
        #print (similarity_scores)
        #print ("our_answer, correct_answer = {}, {}".format(our_answer, correct_answer))

        # LDA stuff
        if topics is not None:
            qt = topics[qi,:]
            at = topics[ai,:]
            bt = topics[bi,:]
            ct = topics[ci,:]
            dt = topics[di,:]

            # use the cosine measure to compare topics?
            topic_scores = []
            topic_scores.append(np.inner(qt, at) / np.linalg.norm(qt) / np.linalg.norm(at) + 1.)
            topic_scores.append(np.inner(qt, bt) / np.linalg.norm(qt) / np.linalg.norm(bt) + 1.)
            topic_scores.append(np.inner(qt, ct) / np.linalg.norm(qt) / np.linalg.norm(ct) + 1.)
            topic_scores.append(np.inner(qt, dt) / np.linalg.norm(qt) / np.linalg.norm(dt) + 1.)

            topic_score_index = topic_scores.index(max(topic_scores))

            sm = sum(topic_scores) + 4*eps
            topic_scores = map(lambda x: (x+eps) / sm, topic_scores)
 

        else:
            topic_scores = [0,0,0,0]

        # use normalized scores to combine with weighted sum
        scores = map(lambda x: 0.75*x[0] + 0.25*x[1], itertools.izip(similarity_scores, topic_scores))

        our_answer = ['A', 'B', 'C', 'D'][scores.index(max(scores))]
        our_answers.append(our_answer)
         
    if print_info:
        print("\r")
    
    return our_answers


def test_xval(questions, X, fv, scorer=similarity.Scorer.cosine,\
        topics=None, print_info=False):
    """
    """
    our_answers = make_predictions(questions, X, fv, scorer=scorer,\
            topics=topics, print_info=print_info)

    num_correct = 0
    num_total = 0
    for q, oa in itertools.izip(questions, our_answers):
        correct_answer = q['correctAnswer']
        
        num_total += 1
        if oa == correct_answer:
            num_correct += 1
    
    accuracy = float(num_correct)/float(num_total)
    return accuracy


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


def lr_make_predictions(scores, lr):
    log_prob = lr.predict_log_proba(scores)
    answers = []
    for i in xrange(log_prob.shape[0]/4):
        ind = np.argmax(log_prob[4*i+0:4*i+4,1]) % 4
        #print (scores[4*i+0:4*i+4,:])
        #print (ind)
        answers.append(['A', 'B', 'C', 'D'][ind])

    return answers


def compute_accuracy(questions, our_answers):
    num_correct = 0
    num_total = 0
    for q, oa in itertools.izip(questions, our_answers):
        correct_answer = q['correctAnswer']
        
        num_total += 1
        if oa == correct_answer:
            num_correct += 1
    
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
        feat = similarity.Featurizer(analyzer, pages_dict)
        
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

    # try some LDA stuff
    print ("Training LDA topic model")
    topic_mod = lda.LDA(n_topics=20, n_iter=150)
    tm_analyzer = topic_model.Analyzer()
    tm_feat = topic_model.Featurizer(tm_analyzer, pages_dict) # use the same feature strings as similarity
    tm_fs = topic_model.add_wiki_categories(trainx+testx, fs, fv, pages_dict)
    topic_X = tm_feat.compute_feats(tm_fs)
    topics = topic_mod.fit_transform(topic_X) # gives probabilities for each topic

    print ("Evaluating train data:")
    acc_trainx = test_xval(trainx, X, fv,\
            #scorer=similarity.Scorer.cosine, print_info=True)
            scorer=similarity.Scorer.cosine, topics=topics, print_info=True)
    print ("Train accuracy = {}\n".format(acc_trainx))

    print ("Evaluating test data:")
    acc_testx = test_xval(testx, X, fv,\
            #scorer=similarity.Scorer.cosine, print_info=True)
            scorer=similarity.Scorer.cosine, topics=topics, print_info=True)
    print ("Test accuracy = {}\n".format(acc_testx))


def answer_xval_lr(args):
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
        feat = similarity.Featurizer(analyzer, pages_dict)
        
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

    #X_scaler = sklearn.preprocessing.StandardScaler(with_mean=False, with_std=True)
    #X = X_scaler.fit_transform(X)

    # try some LDA stuff
    print ("Training LDA topic model")
    topic_mod = lda.LDA(n_topics=20, n_iter=150)
    tm_analyzer = topic_model.Analyzer()
    tm_feat = topic_model.Featurizer(tm_analyzer, pages_dict) # use the same feature strings as similarity
    tm_fs = topic_model.add_wiki_categories(trainx+testx, fs, fv, pages_dict)
    topic_X = tm_feat.compute_feats(tm_fs)
    topics = topic_mod.fit_transform(topic_X) # gives probabilities for each topic

    #topics_scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    #topics = topics_scaler.fit_transform(topics)

    # compute similarity for each question and each answer (of 4)
    # use this as X (e.g. NLP similarity, LDA similarity)
    # binary classification with LR (i.e. is the answer right or not)
    
    print ("Evaluating train data:")
    X_lr_train, y_lr_train = compute_scores(trainx, X, fv,\
            scorer=similarity.Scorer.cosine, topics=topics, train=True,\
            print_info=True)
    print ("Training LR")
    # standardizing
    lr_scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
    X_lr_train = lr_scaler.fit_transform(X_lr_train)

    # alpha sets the weight on regularization term
    lr = sklearn.linear_model.SGDClassifier(loss='log', penalty='l2',\
            n_iter=100, shuffle=True, fit_intercept=True, class_weight={0:.1, 1:.9})
    lr.fit(X_lr_train, y_lr_train)
    #lr.coef_[0,0] = 0.75
    #lr.coef_[0,1] = 0.25
    #lr.intercept_[0] = 0.0
    print (lr.coef_)
    print (lr.intercept_)
    our_answers = lr_make_predictions(X_lr_train, lr)
    acc_trainx = compute_accuracy(trainx, our_answers)
    print ("Train accuracy = {}\n".format(acc_trainx))

    print ("Evaluating test data:")
    X_lr_test = compute_scores(testx, X, fv,\
            scorer=similarity.Scorer.cosine, topics=topics, print_info=True)
    X_lr_test = lr_scaler.transform(X_lr_test)
    our_answers = lr_make_predictions(X_lr_test, lr)
    acc_testx = compute_accuracy(testx, our_answers)
    print ("Test accuracy = {}\n".format(acc_testx))


def answer_questions(args):
    """
    Answer questions on the real-deal dataset by doing the following:
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
        train_reader, test_reader = read_csv_data.read_csv_data()
        train = list(train_reader); test = list(test_reader)
        
        print ("len(train) = {}, len(test) = {}"\
                .format(len(train), len(test)))
 
        analyzer = similarity.Analyzer()
        feat = similarity.Featurizer(analyzer, pages_dict)
        
        print ("Computing feature strings:")
        fs, fv = feat.compute_feat_strings(train + test, print_info=True)
        
        pickle.dump((train, test, fs, fv, analyzer, feat),
                open('../data/realdeal_feat_strings.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

    elif args.load: # load pre-comuted feature strings
        print ("Loading precomputed feature strings for real-deal train and test:")
        train, test, fs, fv, analyzer, feat = \
                pickle.load(open('../data/realdeal_feat_strings.pkl', 'rb'))
    
    ## Here we do some cross-validation
    X = feat.compute_feats(fs)
    X = X.tocsr() # might already be CSR
    X.sort_indices() # needed for cosine-type measures

    # running into memory issues, so release this guy
    feat = None

    # try some LDA stuff
    print ("Training LDA topic model")
    topic_mod = lda.LDA(n_topics=20, n_iter=500)
    tm_feat = topic_model.Featurizer(analyzer, pages_dict) # use the same feature strings as similarity
    tm_fs = topic_model.add_wiki_categories(train+test, fs, fv, pages_dict) # adding these seems to hurt public test performance.  Does slightly better on Xval
    topic_X = tm_feat.compute_feats(tm_fs)
    topics = topic_mod.fit_transform(topic_X) # gives probabilities for each topic

    print ("Evaluating train data (overfitting!):")
    acc_train = test_xval(train, X, fv,\
            #scorer=similarity.Scorer.cosine, print_info=True)
            scorer=similarity.Scorer.cosine, topics=topics, print_info=True)
    print ("Train accuracy = {}\n".format(acc_train))

    print ("Making predictions for test data:")
    our_answers = make_predictions(test, X, fv,\
            #scorer=similarity.Scorer.cosine, print_info=True)
            scorer=similarity.Scorer.cosine, topics=topics, print_info=True)
    answer_file = "../data/our_answers.csv"
    print ("Writing predictions to {}:".format(answer_file))
    o = csv.DictWriter(open(answer_file, 'w'), ["id", "correctAnswer"])
    o.writeheader()
    for q,a in itertools.izip(test, our_answers):
        d = {"id": q['id'], "correctAnswer": a}
        o.writerow(d)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", help="Percent going to training set for\
            X-val", type=float, default=87., required=False)
    argparser.add_argument("--limit", help="Limit training size",
            type=int, default=-1, required=False)
    argparser.add_argument("--load", help="Load precomputed feature strings",
            action="store_true")
    argparser.add_argument("--test", help="Make predictions on the real-deal test set",
            action="store_true")

    args = argparser.parse_args()
    
    # do the stuff!
    if not args.test:
        #answer_xval(args)
        answer_xval_lr(args)

    else:
        answer_questions(args)

