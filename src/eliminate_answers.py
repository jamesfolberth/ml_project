
import sys
import numpy as np
import argparse
import random
import re
import string
import itertools
import csv
import cPickle as pickle

import pdb
#from nltk import word_tokenize, ne_chunk, pos_tag
import nltk

# our modules
import read_csv_data
# import similarity
import utils

#from pygoogle import pygoogle
import time


def question_features(question_data):
    #chunked = ne_chunk(pos_tag(word_tokenize(question_data['question'])))
    chunked = question_data['question'].split()
    found_this = 0
    answer_type = []
    for w in chunked:

        if (found_this):
            #pos = nltk.pos_tag(nltk.word_tokenize(w))
            #if ( (pos[0][1] != 'NN' or pos[0][1] != 'NNS') and (len(answer_type) > 0) ):
            answer_type = (w)
            break
            #pdb.set_trace()
        else:
            found_this = (w == 'this' or w == 'these' or w == 'This' or w == 'These')

    return answer_type

def question_features2(q_data):
    found_this = False
    answer_type = [0,0]
    for w in q_data['pos']:

        if (found_this) and len(w) > 1:
            if w[1] == 'NN':
                #answer_type = (w[0])
                answer_type = [q_data['id'] , w[0]]
                break
            #return [q_data['id'] , w[0]]
        else:
            found_this = (w[0] == 'this' or w[0] == 'these' or \
                w[0] == 'This' or w[0] == 'These')

    return answer_type

def answer_question(question_data, answer_type, wiki_data):
    ''' 'links' --> dont use
        'summary' -->
        'content' --> 2,1 gives 92,471 and 2,2 gives 82,989'''

    answer_list = ['answerA','answerB','answerC','answerD']
    correct_ans_list = ['A','B','C','D']
    count_list = []

    for ii in range(0,4):
        cnt = wiki_data[question_data[answer_list[ii]]]['content'].count(answer_type)
        count_list.append(cnt)

    max_ind = count_list.index(max(count_list))

    if max(count_list) > 3 :
        count_list.remove(max(count_list))
        if max(count_list) < 2:
            return correct_ans_list[max_ind]
    else:
        return []

def google_ans(q_data, ans_type):
    answer_list = ['answerA','answerB','answerC','answerD']
    correct_ans_list = ['A','B','C','D']
    g_counts = []
    for ii in range(len(answer_list)):
        g_query = '\"' + q_data[answer_list[ii]] + ' ' + ans_type + '\"'
        g = pygoogle(g_query)
        #pdb.set_trace()
        g_counts.append(g.get_result_count())
        time.sleep(10)

    return correct_ans_list[g_counts.index(max(g_counts))]

def pickle_ans(pred, questions):

    answer_probs = {}
    answer_dict = {'A':0,'B':1,'C':2,'D':3}
    for q in questions:
        answer_probs[q['id']] = [0,0,0,0]
        if q['id'] in pred:
            if pred[q['id']]:
                answer_probs[q['id']][answer_dict[pred[q['id']]]] = 1
        answer_probs[q['id']] = np.asarray(answer_probs[q['id']])

    return answer_probs


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
    train_pos = pickle.load(open('pos/train_pos.pkl', 'rb'))
    test_pos = pickle.load(open('pos/test_pos.pkl', 'rb'))
    all_pos = train_pos + test_pos

    example_d = pickle.load(open('pos/sim_prob_dict.pkl', 'rb'))

    row_num = 0
    old_ans = []
    with open('pos/our_answers.csv', 'rb') as csvfile:
         ans_reader = csv.reader(csvfile, delimiter=',')
         for row in ans_reader:
            if row_num > 0:
                old_ans.append({'id':row[0],'correctAnswer':row[1]})
            row_num += 1

    if not args.load:
        train_reader, test_reader = read_csv_data.read_csv_data()
        train = list(train_reader)
        test = list(test_reader)
        all_data = train + test
        random.shuffle(train)

        # split train for X-val
        if args.limit > 0:
            train = train[0:args.limit]
        
        trainx, testx = utils.split_list(train, (args.split, 100.-args.split))
        print ("len(xval_train) = {}, len(xval_test) = {}"\
                .format(len(trainx), len(testx)))
 
        #analyzer = similarity.Analyzer()
        #feat = similarity.Featurizer(analyzer, pages_dict)
        
        #print ("Computing feature strings:")
        #fs, fv = feat.compute_feat_strings(trainx + testx, print_info=True)

#####################################
        #use_data = train
        #use_pos = train_pos
        use_data = all_data
        use_pos = all_pos

        ind = 0
        num_this = 0
        ans_types = {}
        num_q = 0
        old_relevant = []
        #for kk in trainx:
        for kk in use_data:
            for kk_pos in use_pos:
            #for kk_pos in train_pos:
                if kk_pos['id'] == kk['id']:
                    break

            #for kk_old in old_ans:
            #    if kk_old['id'] == kk['id']:
            #        break
            #ans_types.append(question_features2(kk))
            #ans_types.append(question_features2(kk_pos))
            [k,t] = question_features2(kk_pos)
            if k != 0:
                ans_types[k] = t
                num_q += 1
            #old_relevant.append(kk_old)
            ind += 1
            sys.stdout.write("Parse Progress: %f%%   \r" % (ind*100/float(len(use_data))) )
            sys.stdout.flush()

        num_empty = 0
        for ans in ans_types:
            if not(ans):
                num_empty += 1

        pred_list = {}
        ind = 0
        max_ind = len(use_data)
        for kk in range(0,len(use_data)):
            #if ind > max_ind:
                #break
            if use_data[kk]['id'] in ans_types.keys():
                ind += 1
                pred_list[use_data[kk]['id']] = answer_question(use_data[kk], \
                    ans_types[use_data[kk]['id']], pages_dict)
            else:
                ind += 1
                pred_list[use_data[kk]['id']] = []
            sys.stdout.write("Parse Progress: %f%%   \r" % (ind*100/max_ind) )
            sys.stdout.flush()        


        '''
        for kk in range(0,len(ans_types)):
            if ind > max_ind:
                break
            if (ans_types[kk]):
                ind += 1
                #pred_list.append(google_ans(trainx[kk], ans_types[kk]))
                #pred_list.append(answer_question(trainx[kk], ans_types[kk], pages_dict))
                pred_list.append(answer_question(use_data[kk], ans_types[kk], pages_dict))

            else:
                ind += 1
                pred_list.append([])
            sys.stdout.write("Parse Progress: %f%%   \r" % (ind*100/max_ind) )
            sys.stdout.flush()        '''    

        corr = 0
        total = 0
        for p in range(0,len(train)):
            q_key = train[p]['id']
            if q_key in pred_list.keys():
                if pred_list[q_key]:
                    if pred_list[q_key] == train[p]['correctAnswer']:
                    #if pred_list[p] == old_relevant[p]['correctAnswer']:
                        corr += 1
                    total +=1

        print ('Performance: ' + str(corr/float(total)))
        print ('Fraction Answered: ' + str(float(total)/float(len(use_data))))

        final_answers = pickle_ans(pred_list, use_data)

        pdb.set_trace()

        filepath = 'pos/metric_dict_10_90.pkl'
        pickle.dump(final_answers,open(filepath, 'wb'))
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", help="Percent going to training set for\
            X-val", type=float, default=100., required=False)
    argparser.add_argument("--limit", help="Limit training size",
            type=int, default=-1, required=False)
    argparser.add_argument("--load", help="Load precomputed feature strings",
            action="store_true")
    argparser.add_argument("--test", help="Make predictions on the real-deal test set",
            action="store_true")

    args = argparser.parse_args()
    
    # do the stuff!
    if not args.test:
        answer_xval(args)

    else:
        answer_questions(args)