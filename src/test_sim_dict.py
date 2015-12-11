
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


def single_make_predictions(questions, prob_dict):
    our_answers = []
    for q in questions:
        arr = prob_dict[q['id']]
        ans = ['A','B','C','D'][np.argmax(arr)]
        our_answers.append(ans)

    return our_answers


def run_test():
    sim_prob_dict = pickle.load(open('../data/sim_prob_dict.pkl', 'rb'))
    topic_prob_dict = pickle.load(open('../data/topic_prob_dict.pkl', 'rb'))
 
    #print ("Loading precomputed feature strings for trainx and testx:")
    #train, test, fs, fv, analyzer, feat = \
    #        pickle.load(open('../data/xval_feat_strings.pkl', 'rb'))
 
    print ("Loading precomputed feature strings for real-deal train and test:")
    train, test, fs, fv, analyzer, feat = \
            pickle.load(open('../data/realdeal_feat_strings.pkl', 'rb'))

   
    
    train_ans = single_make_predictions(train, sim_prob_dict)
    #train_ans = single_make_predictions(train, topic_prob_dict)
    train_acc = compute_accuracy(train, train_ans)
    print ("Train accuracy = {}\n".format(train_acc))


if __name__ == '__main__':

    run_test()

