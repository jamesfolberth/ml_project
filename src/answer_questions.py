
import sys
import numpy as np
import argparse
import random
import re
import string

import cPickle as pickle

import read_csv_data
import similarity
import utils


def answer_questions(args):
    """
    Answer questions by doing the following:
        1. Extract features for the training set
        2. ???
        3. Profit

    Args:
        args: ArgumentParser arguments

    Returns:
        None
    """
    
    if not args.load and not args.loadex:
        train_reader, test_reader = read_csv_data.read_csv_data()
        train = list(train_reader); test = list(test_reader)
        #random.shuffle(train)
        pages_dict = pickle.load(open('../data/wiki_pages_dict.pkl', 'rb'))

        # split train for X-val
        if args.limit > 0:
            train = train[0:args.limit]
        
        trainx, testx = utils.split_list(train, (args.split, 100.-args.split))
        print ("len(xval_train) = {}, len(xval_test) = {}"\
                .format(len(trainx), len(testx)))
 
        
        analyzer = similarity.Analyzer()
        feat = similarity.Featurizer(trainx, analyzer, pages_dict)
        X_trainx, fv_trainx = feat.compute_features(trainx)
        #X_testx, fv_testx = feat.compute_features(testx)

        
        X_trainx = X_trainx.tocsr() # might already be CSR
        X_trainx.sort_indices() # needed for cosine measure
        for ind, q in enumerate(trainx):
            qi = fv_trainx[q['id']]
            ai = fv_trainx[q['answerA']]
            bi = fv_trainx[q['answerB']]
            ci = fv_trainx[q['answerC']]
            di = fv_trainx[q['answerD']]
            
            qv = X_trainx[qi,:] # efficient in CSR format
            av = X_trainx[ai,:]
            bv = X_trainx[bi,:]
            cv = X_trainx[ci,:]
            dv = X_trainx[di,:]

            similarity_scores = []
            similarity_scores.append(similarity.Scorer.cosine(qv, av))
            similarity_scores.append(similarity.Scorer.cosine(qv, bv))
            similarity_scores.append(similarity.Scorer.cosine(qv, cv))
            similarity_scores.append(similarity.Scorer.cosine(qv, dv))
            
            our_answer = ['A','B','C','D'][similarity_scores.index(max(similarity_scores))]
            correct_answer = q['correctAnswer']
            
            print (similarity_scores)
            print ("our_answer, correct_answer = {}, {}".format(our_answer, correct_answer))
            
            if ind > 10:
                raise SystemExit
        


        #train_pquestions = {}
        #train_pcontent = {}
        #train_psummary = {}
        #train_psections = {}
        #for ind, q in enumerate(trainx):
        #    print ("\rquestion {:>06d} of {:>06d}".format(ind+1, len(trainx))),
        #    sys.stdout.flush()
        #    
        #    if q['id'] not in train_pquestions:
        #        train_pquestions[q['id']] = feat.process_text(q['question'])
        #    else:
        #        print ("repeated question id: {}".format(q['id']))
        #        
        #    similarity_scores = [0,0,0,0]
        #    for score_ind, ans in enumerate((q['answerA'], q['answerB'], q['answerC'], q['answerD'])):
        #        ## process text
        #        #if ans not in train_pcontent:
        #        #    train_pcontent[ans] = feat.process_text(\
        #        #            pages_dict[ans]['content'])

        #        if ans not in train_psummary:
        #            train_psummary[ans] = feat.process_text(\
        #                    pages_dict[ans]['summary'])
        #        
        #        similarity_scores[score_ind] += similarity.Scorer.summary(\
        #                train_pquestions[q['id']], train_psummary[ans],\
        #                summary_text=pages_dict[ans]['summary'])


        #        #if ans not in train_psections:
        #        #    train_psections[ans] = feat.process_text(\
        #        #            pages_dict[ans]['sections'])


        #    our_answer = ['A','B','C','D'][similarity_scores.index(max(similarity_scores))]
        #    correct_answer = q['correctAnswer']
        #    
        #    print (similarity_scores)
        #    print ("our_answer, correct_answer = {}, {}".format(our_answer, correct_answer))
        #    
        #    if ind > 10:
        #        raise SystemExit
        #
        #print ("\r")
 
  

    elif args.loadex: # load only examples
        raise NotImplementedError("loading pre-computed examples not implemented")

    elif args.load: # load examples and feature vectors
        raise NotImplementedError("loading pre-computed features not implemented")




    if args.test:
        raise NotImplementedError("Answering test set questions not implemented")


    # {{{
        ## iterate over all answers and print the corresponding wiki page title
        ## as a check that pages_dict has all the right keys
        #for row in train_reader:
        #    for ans in (row['answerA'], row['answerB'], row['answerC'], row['answerD']):
        #        print (u"answer: {}\n  title: {}".format(ans, pages_dict[ans]['title']))
   
        #print (train[0]['question'])

        #print (pages_dict['Genetic drift']['content'].find("Moran"))
        #print (pages_dict['Genetic drift']['content'].find("Wright"))

        #print (pages_dict['Hamiltonian (quantum mechanics)']['content'].find("Moran"))
        #print (pages_dict['Hamiltonian (quantum mechanics)']['content'].find("Wright"))

        #print (pages_dict['Georg Wilhelm Friedrich Hegel']['content'].find("Moran"))
        #print (pages_dict['Georg Wilhelm Friedrich Hegel']['content'].find("Wright"))
    # }}} 

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", help="Percent going to training set for\
            X-val", type=float, default=85., required=False)
    argparser.add_argument("--limit", help="Limit training size",
            type=int, default=-1, required=False)
    argparser.add_argument("--load", help="Load precomputed features",
            type=bool, default=False, required=False)
    argparser.add_argument("--loadex", help="Load precomputed examples only",
            type=bool, default=False, required=False)
    argparser.add_argument("--test", help="Make predictions on the true test set",
            type=bool, default=False, required=False)

    args = argparser.parse_args()
    
    # do the stuff!
    answer_questions(args)

